"""
CreditLens — Combined FastAPI (OpenEnv REST) + Gradio UI
Runs on port 7860 for HuggingFace Spaces.

FIX: Gradio is now mounted at /ui (not /) so the validator's
     POST /reset hits FastAPI directly and returns HTTP 200.
     Previous bug: gr.mount_gradio_app(api, demo, path="/") caused
     Gradio to intercept POST /reset → 405 Method Not Allowed.
"""

from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

sys.path.insert(0, str(Path(__file__).parent))

import gradio as gr
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel

from creditlens.models import (
    ActionType,
    LoanObservation,
    RejectReason,
    RequestField,
    UnderwritingAction,
)
from creditlens.env.engine import CreditLensEnv, TASK_CONFIGS
from creditlens.tasks.graders import grade_episode

# ══════════════════════════════════════════════════════════════════════════
# FastAPI app — ALL OpenEnv REST routes live here at the ROOT path
# The validator calls POST /reset → this must return HTTP 200 JSON
# ══════════════════════════════════════════════════════════════════════════

api = FastAPI(
    title="CreditLens OpenEnv",
    description="AI Credit Risk Underwriting Environment — OpenEnv Compliant",
    version="1.0.0",
)

api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Session store: "{session_id}:{task_id}" → CreditLensEnv
_sessions: Dict[str, CreditLensEnv] = {}
_DEFAULT_SESSION = "default"


class ResetRequest(BaseModel):
    task_id: str = "easy"
    seed: Optional[int] = 42
    session_id: Optional[str] = _DEFAULT_SESSION


class StepRequest(BaseModel):
    task_id: str = "easy"
    action: Dict[str, Any]
    session_id: Optional[str] = _DEFAULT_SESSION


def _get_env(session_id: str, task_id: str) -> CreditLensEnv:
    key = f"{session_id}:{task_id}"
    if key not in _sessions:
        _sessions[key] = CreditLensEnv(task_id=task_id)
    return _sessions[key]


# ── Health check ──────────────────────────────────────────────────────────
@api.get("/health")
def health():
    """Liveness check — required by the validator."""
    return {"status": "ok", "service": "creditlens"}


# ── Task listing ──────────────────────────────────────────────────────────
@api.get("/tasks")
def list_tasks():
    return {
        tid: {
            "name": cfg.name,
            "num_applicants": cfg.num_applicants,
            "max_steps": cfg.max_steps,
            "fraud_ring_size": cfg.fraud_ring_size,
            "macro_shock": cfg.macro_shock,
        }
        for tid, cfg in TASK_CONFIGS.items()
    }


# ── POST /reset ───────────────────────────────────────────────────────────
# THE CRITICAL ROUTE: validator calls POST /reset with empty body {}
# Must return HTTP 200 with a valid JSON observation.
@api.post("/reset")
def reset_endpoint(req: ResetRequest = None):
    """
    OpenEnv reset — starts a new episode.
    Accepts empty body {} (validator sends this) or full ResetRequest.
    """
    if req is None:
        req = ResetRequest()

    task_id    = req.task_id or "easy"
    session_id = req.session_id or _DEFAULT_SESSION
    seed       = req.seed

    if task_id not in TASK_CONFIGS:
        task_id = "easy"

    env = CreditLensEnv(task_id=task_id)
    key = f"{session_id}:{task_id}"
    _sessions[key] = env

    observation = env.reset(seed=seed)
    state       = env.state()

    return {
        "observation": observation.model_dump(),
        "episode_id":  state.episode_id,
        "task_id":     task_id,
        "num_applicants": TASK_CONFIGS[task_id].num_applicants,
        "max_steps":      TASK_CONFIGS[task_id].max_steps,
    }


# ── POST /step ────────────────────────────────────────────────────────────
@api.post("/step")
def step_endpoint(req: StepRequest):
    task_id    = req.task_id or "easy"
    session_id = req.session_id or _DEFAULT_SESSION
    key        = f"{session_id}:{task_id}"

    if key not in _sessions:
        return JSONResponse(
            {"error": "No active session. Call POST /reset first."},
            status_code=400,
        )

    env = _sessions[key]
    try:
        action_data = req.action
        if "action_type" not in action_data:
            return JSONResponse(
                {"error": "action must contain 'action_type'"},
                status_code=400,
            )
        action = UnderwritingAction(**action_data)
        result = env.step(action)
        return {
            "observation": result.observation.model_dump() if result.observation else None,
            "reward":      result.reward,
            "done":        result.done,
            "info":        result.info,
        }
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=422)


# ── GET /state ────────────────────────────────────────────────────────────
@api.get("/state")
def state_endpoint(
    task_id:    str = "easy",
    session_id: str = _DEFAULT_SESSION,
):
    key = f"{session_id}:{task_id}"
    if key not in _sessions:
        return JSONResponse(
            {"error": "No active session. Call POST /reset first."},
            status_code=400,
        )
    return _sessions[key].state().model_dump()


# ── GET /grade ────────────────────────────────────────────────────────────
@api.get("/grade")
def grade_endpoint(
    task_id:    str = "easy",
    session_id: str = _DEFAULT_SESSION,
):
    key = f"{session_id}:{task_id}"
    if key not in _sessions:
        return JSONResponse({"error": "No episode to grade."}, status_code=400)
    env    = _sessions[key]
    config = TASK_CONFIGS[task_id]
    scores = grade_episode(env.state(), config)
    return {
        "task_id":     task_id,
        "scores":      scores,
        "final_score": scores.get("score", 0.0),
    }


# ══════════════════════════════════════════════════════════════════════════
# Gradio UI — mounted at /ui so it does NOT conflict with POST /reset
# Visit https://your-space.hf.space/ui for the interactive demo
# ══════════════════════════════════════════════════════════════════════════

ACTION_COLOURS = {
    "APPROVE":      ("#16a34a", "#ffffff"),
    "REJECT":       ("#dc2626", "#ffffff"),
    "COUNTER":      ("#d97706", "#ffffff"),
    "REQUEST_INFO": ("#2563eb", "#ffffff"),
}


def _rule_based_decide(obs: LoanObservation) -> UnderwritingAction:
    ecl_usage   = obs.portfolio_ecl / max(obs.portfolio_ecl_budget, 1e-8)
    shock_adj   = 0.15 if obs.macro_shock_active else 0.0
    fairness_gap = obs.approval_rate_reference - obs.approval_rate_protected
    is_protected = obs.demographic_group in ("group_b", "group_c")

    def _rej(reason, note):
        return UnderwritingAction(action_type=ActionType.REJECT,
                                  applicant_id=obs.applicant_id,
                                  params={"reason_code": reason}, reasoning=note)

    if obs.fraud_ring_score > 0.70 or (obs.shared_phone and obs.shared_employer_id):
        return _rej(RejectReason.FRAUD_SUSPECTED.value, f"Fraud {obs.fraud_ring_score:.0%}")

    xgb_cut = 0.62 - shock_adj
    if obs.xgb_default_prob > xgb_cut:
        return _rej(RejectReason.HIGH_DTI.value, f"XGB {obs.xgb_default_prob:.0%} > cutoff")

    ecl_t = 1.1
    if ecl_usage > 0.95:   ecl_t = 0.10
    elif ecl_usage > 0.80: ecl_t = 0.15
    elif ecl_usage > 0.60: ecl_t = 0.35
    ecl_t -= shock_adj
    if obs.xgb_default_prob > ecl_t:
        return _rej(RejectReason.HIGH_DTI.value, f"ECL guard {ecl_usage:.0%}")

    if (fairness_gap > 0.12 and is_protected and
            obs.xgb_default_prob < 0.60 and obs.fico_score >= 600 and
            obs.dti_ratio < 0.50 and obs.fraud_ring_score < 0.50):
        return UnderwritingAction(action_type=ActionType.APPROVE,
                                  applicant_id=obs.applicant_id,
                                  params={"amount_fraction": 0.90},
                                  reasoning=f"Fairness rescue gap={fairness_gap:+.0%}")

    if (fairness_gap < -0.12 and is_protected and obs.xgb_default_prob > 0.35):
        return UnderwritingAction(action_type=ActionType.COUNTER,
                                  applicant_id=obs.applicant_id,
                                  params={"revised_amount_fraction": 0.75, "revised_rate_delta": 1.5},
                                  reasoning=f"Fairness rebalance gap={fairness_gap:+.0%}")

    if obs.macro_shock_active and obs.xgb_default_prob > 0.30:
        return UnderwritingAction(action_type=ActionType.COUNTER,
                                  applicant_id=obs.applicant_id,
                                  params={"revised_amount_fraction": 0.75, "revised_rate_delta": 2.0},
                                  reasoning="Post-shock conservative counter")

    if obs.fico_score < 580:
        return _rej(RejectReason.LOW_CREDIT_SCORE.value, "FICO < 580")
    if obs.dti_ratio > 0.50:
        return _rej(RejectReason.HIGH_DTI.value, "DTI > 50%")
    if obs.xgb_default_prob > 0.40 or obs.fico_score < 620 or obs.dti_ratio > 0.40:
        frac = 0.70 if obs.xgb_default_prob > 0.50 else 0.82
        return UnderwritingAction(action_type=ActionType.COUNTER,
                                  applicant_id=obs.applicant_id,
                                  params={"revised_amount_fraction": frac, "revised_rate_delta": 1.5},
                                  reasoning="Moderate risk counter")
    return UnderwritingAction(action_type=ActionType.APPROVE,
                              applicant_id=obs.applicant_id,
                              params={"amount_fraction": 1.0},
                              reasoning="Creditworthy — approve")


def _risk_bar(prob: float) -> str:
    pct = int(prob * 100)
    c = "#16a34a" if prob < 0.25 else "#d97706" if prob < 0.50 else "#dc2626"
    return (f'<div style="background:#e5e7eb;border-radius:6px;height:12px;width:100%;margin:4px 0 2px;">'
            f'<div style="background:{c};width:{pct}%;height:12px;border-radius:6px;"></div></div>'
            f'<span style="font-size:0.8rem;color:{c};font-weight:700;">{pct}% default risk</span>')


def _fico_color(f: int) -> str:
    if f >= 720: return "#16a34a"
    if f >= 660: return "#65a30d"
    if f >= 620: return "#d97706"
    return "#dc2626"


def _mini_card(title, value, color="#1e293b", sub=""):
    return (f'<div style="background:#fff;border:1px solid #e2e8f0;border-radius:10px;padding:12px;text-align:center;">'
            f'<div style="font-size:0.72rem;color:#64748b;margin-bottom:4px;">{title}</div>'
            f'<div style="font-size:1.5rem;font-weight:800;color:{color};">{value}</div>'
            + (f'<div style="font-size:0.72rem;color:#94a3b8;">{sub}</div>' if sub else "")
            + '</div>')


def _obs_html(obs: LoanObservation) -> str:
    shock_badge = (
        f'<span style="background:#7c3aed;color:#fff;padding:3px 10px;border-radius:12px;font-size:0.8rem;font-weight:700;">⚡ SHOCK +{obs.shock_magnitude_bps}bps</span>'
        if obs.macro_shock_active else
        '<span style="background:#dcfce7;color:#16a34a;padding:3px 10px;border-radius:12px;font-size:0.8rem;">No Shock</span>'
    )
    fraud_c = "#dc2626" if obs.fraud_ring_score > 0.50 else "#16a34a"
    fraud_icon = "🚨" if obs.fraud_ring_score > 0.70 else ("⚠️" if obs.fraud_ring_score > 0.40 else "✅")
    shap_c = "#dc2626" if obs.shap_top_value > 0 else "#16a34a"
    group_colors = {"group_a": "#3b82f6", "group_b": "#8b5cf6", "group_c": "#ec4899"}
    grp  = str(obs.demographic_group)
    gc   = group_colors.get(grp, "#64748b")
    dti_c = "#dc2626" if obs.dti_ratio > 0.43 else "#d97706" if obs.dti_ratio > 0.36 else "#16a34a"
    purpose = str(obs.loan_purpose).replace("LoanPurpose.", "").upper()

    return f"""
<div style="font-family:sans-serif;padding:18px;background:#f8fafc;border-radius:14px;border:1px solid #e2e8f0;">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:14px;flex-wrap:wrap;gap:8px;">
    <h3 style="margin:0;color:#0f172a;font-size:1.1rem;">
      Applicant <code style="background:#e2e8f0;padding:3px 8px;border-radius:6px;">{obs.applicant_id}</code>
      <span style="font-size:0.75rem;color:#94a3b8;margin-left:8px;">Step {obs.step_number} · {obs.steps_remaining} left</span>
    </h3>
    {shock_badge}
  </div>
  <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;margin-bottom:12px;">
    {_mini_card("FICO Score", str(obs.fico_score), _fico_color(obs.fico_score), "Credit score")}
    {_mini_card("Annual Income", f"${obs.income:,.0f}", "#1e293b", "USD/year")}
    {_mini_card("Loan Amount", f"${obs.loan_amount:,.0f}", "#1e293b", purpose)}
  </div>
  <div style="background:#fff;border:1px solid #e2e8f0;border-radius:10px;padding:12px;margin-bottom:10px;">
    <div style="font-size:0.72rem;color:#64748b;margin-bottom:2px;">XGBoost Default Probability</div>
    {_risk_bar(obs.xgb_default_prob)}
    <div style="font-size:0.72rem;color:#94a3b8;margin-top:4px;">
      Top driver: <b style="color:{shap_c};">{obs.shap_top_feature}</b>
      <span style="color:{shap_c};">{obs.shap_top_value:+.2f}</span>
    </div>
  </div>
  <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;margin-bottom:10px;">
    <div style="background:#fff;border:1px solid #e2e8f0;border-radius:8px;padding:8px;">
      <div style="font-size:0.72rem;color:#64748b;">DTI</div>
      <div style="font-weight:700;color:{dti_c};">{obs.dti_ratio:.1%}</div>
    </div>
    <div style="background:#fff;border:1px solid #e2e8f0;border-radius:8px;padding:8px;">
      <div style="font-size:0.72rem;color:#64748b;">Fraud Score</div>
      <div style="font-weight:700;color:{fraud_c};">{obs.fraud_ring_score:.1%} {fraud_icon}</div>
    </div>
    <div style="background:#fff;border:1px solid #e2e8f0;border-radius:8px;padding:8px;">
      <div style="font-size:0.72rem;color:#64748b;">Group</div>
      <span style="background:{gc}22;color:{gc};padding:2px 8px;border-radius:6px;font-weight:700;font-size:0.82rem;">{grp}</span>
    </div>
  </div>
</div>"""


def _portfolio_html(env_state) -> str:
    ecl_usage = env_state.portfolio_ecl / max(env_state.portfolio_ecl_budget, 1e-8)
    ecl_pct   = min(ecl_usage * 100, 100)
    ecl_c = "#16a34a" if ecl_usage < 0.60 else "#d97706" if ecl_usage < 0.85 else "#dc2626"

    ref_dec  = env_state.decisions_by_group.get("group_a", 0)
    ref_app  = env_state.approvals_by_group.get("group_a", 0)
    prot_dec = (env_state.decisions_by_group.get("group_b", 0)
                + env_state.decisions_by_group.get("group_c", 0))
    prot_app = (env_state.approvals_by_group.get("group_b", 0)
                + env_state.approvals_by_group.get("group_c", 0))
    ref_rate  = ref_app  / max(ref_dec,  1)
    prot_rate = prot_app / max(prot_dec, 1)
    gap = ref_rate - prot_rate

    if ref_dec < 1 or prot_dec < 1:
        gap_html = '<div style="font-size:0.82rem;color:#94a3b8;">⚖️ Not enough data yet</div>'
    elif abs(gap) < 0.12:
        gap_html = f'<div style="color:#16a34a;font-weight:700;">⚖️ {gap:+.0%} ✅ Within threshold</div>'
    elif gap > 0:
        gap_html = f'<div style="color:#dc2626;font-weight:700;">⚖️ {gap:+.0%} ⚠️ Protected under-approved</div>'
    else:
        gap_html = f'<div style="color:#d97706;font-weight:700;">⚖️ {gap:+.0%} ⚠️ Protected over-approved</div>'

    return f"""
<div style="font-family:sans-serif;padding:16px;background:#f8fafc;border-radius:14px;border:1px solid #e2e8f0;">
  <h4 style="margin:0 0 12px;color:#0f172a;">📊 Portfolio</h4>
  <div style="margin-bottom:10px;">
    <div style="display:flex;justify-content:space-between;font-size:0.8rem;margin-bottom:4px;">
      <span style="color:#64748b;">ECL Budget Used</span>
      <span style="font-weight:700;color:{ecl_c};">{ecl_usage:.0%}</span>
    </div>
    <div style="background:#e5e7eb;border-radius:8px;height:10px;">
      <div style="background:{ecl_c};width:{ecl_pct:.1f}%;height:10px;border-radius:8px;"></div>
    </div>
  </div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:10px;font-size:0.82rem;">
    <div style="background:#fff;border:1px solid #e2e8f0;border-radius:8px;padding:8px;"><div style="color:#64748b;font-size:0.72rem;">Step</div><div style="font-weight:700;">{env_state.step}</div></div>
    <div style="background:#fff;border:1px solid #e2e8f0;border-radius:8px;padding:8px;"><div style="color:#64748b;font-size:0.72rem;">Reward</div><div style="font-weight:700;color:{'#16a34a' if env_state.total_reward >= 0 else '#dc2626'};">{env_state.total_reward:+.3f}</div></div>
    <div style="background:#fff;border:1px solid #e2e8f0;border-radius:8px;padding:8px;"><div style="color:#64748b;font-size:0.72rem;">Fraud Caught</div><div style="font-weight:700;color:#16a34a;">{env_state.fraud_caught}</div></div>
    <div style="background:#fff;border:1px solid #e2e8f0;border-radius:8px;padding:8px;"><div style="color:#64748b;font-size:0.72rem;">Fraud Missed</div><div style="font-weight:700;color:{'#dc2626' if env_state.fraud_missed > 0 else '#16a34a'};">{env_state.fraud_missed}</div></div>
  </div>
  {gap_html}
</div>"""


def _history_html(history: list) -> str:
    if not history:
        return "<p style='color:#94a3b8;font-style:italic;padding:12px;'>No decisions yet.</p>"
    rows = ""
    for h in reversed(history[-15:]):
        bg, fg = ACTION_COLOURS.get(h["action"], ("#6b7280", "#fff"))
        rwd_c = "#16a34a" if h["reward"] > 0 else "#dc2626"
        rows += (f'<tr style="border-bottom:1px solid #f1f5f9;">'
                 f'<td style="padding:6px 8px;font-size:0.82rem;font-family:monospace;">{h["applicant_id"]}</td>'
                 f'<td style="padding:6px 8px;"><span style="background:{bg};color:{fg};padding:2px 8px;border-radius:8px;font-size:0.78rem;font-weight:700;">{h["action"]}</span></td>'
                 f'<td style="padding:6px 8px;font-size:0.82rem;">{h["fico"]}</td>'
                 f'<td style="padding:6px 8px;font-size:0.82rem;font-weight:700;color:{rwd_c};">{h["reward"]:+.3f}</td>'
                 f'<td style="padding:6px 8px;font-size:0.78rem;color:#64748b;">{h["reasoning"][:55]}</td></tr>')
    return (f'<div style="overflow-x:auto;font-family:sans-serif;">'
            f'<table style="width:100%;border-collapse:collapse;">'
            f'<thead><tr style="background:#f1f5f9;border-bottom:2px solid #e2e8f0;">'
            f'<th style="padding:7px 8px;text-align:left;font-size:0.78rem;">App</th>'
            f'<th style="padding:7px 8px;text-align:left;font-size:0.78rem;">Decision</th>'
            f'<th style="padding:7px 8px;text-align:left;font-size:0.78rem;">FICO</th>'
            f'<th style="padding:7px 8px;text-align:left;font-size:0.78rem;">Reward</th>'
            f'<th style="padding:7px 8px;text-align:left;font-size:0.78rem;">Reasoning</th>'
            f'</tr></thead><tbody>{rows}</tbody></table></div>')


def _fresh():
    return {"env": None, "obs": None, "history": [], "done": False, "task_id": "easy"}


def _all_btns(on: bool):
    return [gr.update(interactive=on)] * 5


def start_episode(task_id: str, state: dict):
    env = CreditLensEnv(task_id=task_id, seed=42)
    obs = env.reset()
    state.update({"env": env, "obs": obs, "history": [], "done": False, "task_id": task_id})
    cfg = TASK_CONFIGS[task_id]
    info_html = (f'<div style="background:#eff6ff;border:1px solid #bfdbfe;border-radius:10px;padding:12px;font-family:sans-serif;">'
                 f'<b>📋 {cfg.name}</b>'
                 f'<div style="font-size:0.85rem;margin-top:6px;">👥 {cfg.num_applicants} applicants · ⏱ {cfg.max_steps} max steps · 💰 {cfg.ecl_budget:.0%} ECL budget</div></div>')
    return (_obs_html(obs), _portfolio_html(env.state()), _history_html(state["history"]),
            info_html, *_all_btns(True), state)


def _apply(action: UnderwritingAction, state: dict):
    env = state["env"]
    obs = state["obs"]
    result = env.step(action)
    state["history"].append({
        "applicant_id": obs.applicant_id,
        "action":       str(action.action_type).replace("ActionType.", ""),
        "fico":         obs.fico_score,
        "reward":       result.reward,
        "reasoning":    action.reasoning or "",
    })
    env_state = env.state()
    if result.done or result.observation is None:
        state["done"] = True
        done_html = (f'<div style="background:#f0fdf4;border:2px solid #86efac;border-radius:14px;'
                     f'padding:20px;text-align:center;font-family:sans-serif;">'
                     f'<div style="font-size:2rem;">✅</div>'
                     f'<h3 style="color:#15803d;margin:8px 0;">Episode Complete — {state["task_id"].upper()}</h3>'
                     f'<p style="color:#475569;">Total reward: {env_state.total_reward:+.3f} · '
                     f'Fraud caught: {env_state.fraud_caught}</p></div>')
        return (done_html, _portfolio_html(env_state), _history_html(state["history"]),
                *_all_btns(False), state)
    state["obs"] = result.observation
    return (_obs_html(result.observation), _portfolio_html(env_state),
            _history_html(state["history"]), *_all_btns(True), state)


def _no_ep(state):
    ph = "<p style='color:#94a3b8;padding:20px;'>Start an episode first.</p>"
    return (ph, ph, _history_html(state.get("history", [])), *_all_btns(False), state)


def auto_decide(state):
    if state["done"] or not state["obs"]: return _no_ep(state)
    return _apply(_rule_based_decide(state["obs"]), state)


def do_approve(frac, state):
    if state["done"] or not state["obs"]: return _no_ep(state)
    obs = state["obs"]
    return _apply(UnderwritingAction(action_type=ActionType.APPROVE, applicant_id=obs.applicant_id,
                                     params={"amount_fraction": frac},
                                     reasoning=f"Manual APPROVE {frac:.0%}"), state)


def do_reject(reason, state):
    if state["done"] or not state["obs"]: return _no_ep(state)
    obs = state["obs"]
    return _apply(UnderwritingAction(action_type=ActionType.REJECT, applicant_id=obs.applicant_id,
                                     params={"reason_code": reason},
                                     reasoning=f"Manual REJECT: {reason}"), state)


def do_counter(frac, delta, state):
    if state["done"] or not state["obs"]: return _no_ep(state)
    obs = state["obs"]
    return _apply(UnderwritingAction(action_type=ActionType.COUNTER, applicant_id=obs.applicant_id,
                                     params={"revised_amount_fraction": frac, "revised_rate_delta": delta},
                                     reasoning=f"Manual COUNTER {frac:.0%} +{delta:.1f}%"), state)


def do_info(field, state):
    if state["done"] or not state["obs"]: return _no_ep(state)
    obs = state["obs"]
    return _apply(UnderwritingAction(action_type=ActionType.REQUEST_INFO, applicant_id=obs.applicant_id,
                                     params={"field_name": field},
                                     reasoning=f"Requesting: {field}"), state)


# ── Build Gradio UI ───────────────────────────────────────────────────────
with gr.Blocks(title="CreditLens — AI Loan Underwriting") as demo:
    state = gr.State(_fresh())

    gr.HTML("""
<div style="text-align:center;padding:24px 20px 10px;font-family:sans-serif;
     background:linear-gradient(135deg,#0f172a,#1e3a5f);border-radius:14px;margin-bottom:14px;">
  <div style="font-size:2.2rem;margin-bottom:4px;">🏦</div>
  <h1 style="margin:0;font-size:1.8rem;font-weight:900;color:#f8fafc;">CreditLens</h1>
  <p style="color:#94a3b8;margin:6px 0 0;font-size:0.9rem;">
    AI loan underwriting · credit risk · fairness · fraud · macro shocks
  </p>
  <p style="color:#64748b;margin:6px 0 0;font-size:0.78rem;">
    REST API available at root — <code style="color:#7dd3fc;">/reset</code>
    <code style="color:#7dd3fc;">/step</code> <code style="color:#7dd3fc;">/grade</code>
    <code style="color:#7dd3fc;">/health</code>
  </p>
</div>""")

    with gr.Row():
        with gr.Column(scale=2):
            task_dd   = gr.Dropdown(choices=["easy","medium","hard"], value="easy",
                                    label="Select Task")
            start_btn = gr.Button("🚀 Start New Episode", variant="primary", size="lg")
            task_info = gr.HTML("<p style='color:#94a3b8;font-size:0.85rem;padding:6px;'>Select a task and click Start.</p>")
        with gr.Column(scale=3):
            portfolio_out = gr.HTML("<p style='color:#94a3b8;padding:16px;'>Portfolio metrics appear here.</p>")

    with gr.Row(equal_height=False):
        with gr.Column(scale=3):
            obs_out = gr.HTML("""<div style='padding:40px;color:#94a3b8;text-align:center;'>
  <div style='font-size:3rem;'>🏦</div>
  <p>Start an episode to see your first applicant.</p></div>""")
        with gr.Column(scale=2):
            gr.Markdown("### 🤖 AI Auto-Decision")
            auto_btn     = gr.Button("⚡ AI Decides", variant="secondary", interactive=False)
            gr.Markdown("### ✅ Approve")
            approve_frac = gr.Slider(0.5, 1.0, value=1.0, step=0.05, label="Amount Fraction")
            approve_btn  = gr.Button("✅ Approve", interactive=False, variant="primary")
            gr.Markdown("### ❌ Reject")
            reject_reason = gr.Dropdown(choices=[r.value for r in RejectReason],
                                        value="HIGH_DTI", label="Reason Code")
            reject_btn   = gr.Button("❌ Reject", interactive=False)
            gr.Markdown("### 🔄 Counter")
            counter_frac  = gr.Slider(0.25, 0.9, value=0.70, step=0.05, label="Revised Fraction")
            counter_delta = gr.Slider(0.0, 5.0, value=1.5, step=0.25, label="Rate Delta %")
            counter_btn   = gr.Button("🔄 Counter", interactive=False)
            gr.Markdown("### 📄 Request Info")
            info_field   = gr.Dropdown(choices=[f.value for f in RequestField],
                                       value="income_proof", label="Document")
            info_btn     = gr.Button("📄 Request Info", interactive=False)

    gr.Markdown("### 📜 Decision History")
    history_out = gr.HTML("<p style='color:#94a3b8;font-style:italic;padding:12px;'>No decisions yet.</p>")

    OUTS = [obs_out, portfolio_out, history_out,
            auto_btn, approve_btn, reject_btn, counter_btn, info_btn, state]

    start_btn.click(fn=start_episode, inputs=[task_dd, state],
                    outputs=[obs_out, portfolio_out, history_out, task_info,
                             auto_btn, approve_btn, reject_btn, counter_btn, info_btn, state])
    auto_btn.click(fn=auto_decide,   inputs=[state], outputs=OUTS)
    approve_btn.click(fn=do_approve, inputs=[approve_frac, state], outputs=OUTS)
    reject_btn.click(fn=do_reject,   inputs=[reject_reason, state], outputs=OUTS)
    counter_btn.click(fn=do_counter, inputs=[counter_frac, counter_delta, state], outputs=OUTS)
    info_btn.click(fn=do_info,       inputs=[info_field, state], outputs=OUTS)


# ══════════════════════════════════════════════════════════════════════════
# CRITICAL: Mount Gradio at /ui — NOT at / — so POST /reset works
# Old bug:  gr.mount_gradio_app(api, demo, path="/")  ← Gradio catches POST /reset → 405
# Fix:      gr.mount_gradio_app(api, demo, path="/ui") ← FastAPI handles / routes
# ══════════════════════════════════════════════════════════════════════════
app = gr.mount_gradio_app(api, demo, path="/ui")


# Redirect bare / to /ui so the Space looks nice in a browser
@api.get("/")
def root_redirect():
    return RedirectResponse(url="/ui")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)