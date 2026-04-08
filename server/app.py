"""
CreditLens — server/app.py
============================
This is the file the OpenEnv validator looks for.

Validator requirements (ALL must pass):
  ✅ File exists at server/app.py
  ✅ Contains a callable main() function
  ✅ Has `if __name__ == '__main__': main()`
  ✅ Entry point is server.app:main (NOT creditlens.inference.service:app)

What this does:
  - Mounts the full OpenEnv REST API (POST /reset, POST /step, GET /state, GET /grade)
  - Mounts the Gradio UI at /
  - Starts uvicorn on port 7860 (HuggingFace Spaces default)
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

# ── Make the creditlens package importable ──────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

import uvicorn
import gradio as gr
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
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
# SECTION 1 — FastAPI: full OpenEnv REST API
# POST /reset is what the validator calls first.
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


@api.get("/health")
def health():
    """Health check — validator pings this first."""
    return {"status": "ok", "service": "creditlens"}


@api.get("/tasks")
def list_tasks():
    """List available tasks with config."""
    return {
        tid: {
            "name": cfg.name,
            "num_applicants": cfg.num_applicants,
            "max_steps": cfg.max_steps,
            "ecl_budget": cfg.ecl_budget,
            "fraud_ring_size": cfg.fraud_ring_size,
            "macro_shock": cfg.macro_shock,
        }
        for tid, cfg in TASK_CONFIGS.items()
    }


@api.post("/reset")
def reset_endpoint(req: ResetRequest = None):
    """
    OpenEnv /reset — start a new episode, return first observation.
    Validator calls: POST /reset with empty body or {"task_id": "easy"}.
    Must return HTTP 200 with JSON containing observation.
    """
    if req is None:
        req = ResetRequest()

    task_id = req.task_id if req.task_id in TASK_CONFIGS else "easy"
    session_id = req.session_id or _DEFAULT_SESSION

    env = CreditLensEnv(task_id=task_id)
    key = f"{session_id}:{task_id}"
    _sessions[key] = env

    obs = env.reset(seed=req.seed)
    state = env.state()

    return {
        "observation": obs.model_dump(),
        "episode_id": state.episode_id,
        "task_id": task_id,
        "num_applicants": TASK_CONFIGS[task_id].num_applicants,
        "max_steps": TASK_CONFIGS[task_id].max_steps,
    }


@api.post("/step")
def step_endpoint(req: StepRequest):
    """OpenEnv /step — process one agent action."""
    task_id = req.task_id if req.task_id in TASK_CONFIGS else "easy"
    session_id = req.session_id or _DEFAULT_SESSION
    key = f"{session_id}:{task_id}"

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
            "reward": result.reward,
            "reward_breakdown": result.reward_breakdown.model_dump(),
            "done": result.done,
            "info": result.info,
        }
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=422)


@api.get("/state")
def state_endpoint(task_id: str = "easy", session_id: str = _DEFAULT_SESSION):
    """OpenEnv /state — return current episode state."""
    key = f"{session_id}:{task_id}"
    if key not in _sessions:
        return JSONResponse(
            {"error": "No active session. Call POST /reset first."},
            status_code=400,
        )
    return _sessions[key].state().model_dump()


@api.get("/grade")
def grade_endpoint(task_id: str = "easy", session_id: str = _DEFAULT_SESSION):
    """Grade the current episode — returns 0.0–1.0 with component breakdown."""
    key = f"{session_id}:{task_id}"
    if key not in _sessions:
        return JSONResponse({"error": "No episode to grade."}, status_code=400)
    env = _sessions[key]
    config = TASK_CONFIGS[task_id]
    scores = grade_episode(env.state(), config)
    return {
        "task_id": task_id,
        "scores": scores,
        "final_score": scores.get("score", 0.0),
    }


# ══════════════════════════════════════════════════════════════════════════
# SECTION 2 — Gradio UI
# ══════════════════════════════════════════════════════════════════════════

ACTION_COLOURS = {
    "APPROVE":      ("#16a34a", "#ffffff"),
    "REJECT":       ("#dc2626", "#ffffff"),
    "COUNTER":      ("#d97706", "#ffffff"),
    "REQUEST_INFO": ("#2563eb", "#ffffff"),
}


def _rule_based_decide(obs: LoanObservation) -> UnderwritingAction:
    ecl_usage = obs.portfolio_ecl / max(obs.portfolio_ecl_budget, 1e-8)
    shock_adj = 0.15 if obs.macro_shock_active else 0.0
    fairness_gap = obs.approval_rate_reference - obs.approval_rate_protected
    is_protected = obs.demographic_group in ("group_b", "group_c")

    def _rej(reason, note):
        return UnderwritingAction(action_type=ActionType.REJECT,
                                  applicant_id=obs.applicant_id,
                                  params={"reason_code": reason},
                                  reasoning=note)

    # P1 Fraud
    if obs.fraud_ring_score > 0.70 or (obs.shared_phone and obs.shared_employer_id):
        return _rej(RejectReason.FRAUD_SUSPECTED.value,
                    f"Fraud score {obs.fraud_ring_score:.0%} — ring detected")
    # P2 XGB cutoff
    xgb_cut = 0.62 - shock_adj
    if obs.xgb_default_prob > xgb_cut:
        return _rej(RejectReason.HIGH_DTI.value,
                    f"XGB {obs.xgb_default_prob:.0%} > cutoff {xgb_cut:.0%}")
    # P3 ECL guard
    ecl_t = 1.1
    if ecl_usage > 0.95:   ecl_t = 0.10
    elif ecl_usage > 0.80: ecl_t = 0.15
    elif ecl_usage > 0.60: ecl_t = 0.35
    ecl_t -= shock_adj
    if obs.xgb_default_prob > ecl_t:
        return _rej(RejectReason.HIGH_DTI.value,
                    f"ECL {ecl_usage:.0%} used — portfolio guard")
    # P4 Fairness rescue (protected only)
    if (fairness_gap > 0.12 and is_protected and
            obs.xgb_default_prob < 0.60 and obs.fico_score >= 600 and
            obs.dti_ratio < 0.50 and obs.fraud_ring_score < 0.50):
        return UnderwritingAction(action_type=ActionType.APPROVE,
                                  applicant_id=obs.applicant_id,
                                  params={"amount_fraction": 0.90},
                                  reasoning=f"Fairness rescue (gap={fairness_gap:+.0%})")
    if (fairness_gap < -0.12 and is_protected and obs.xgb_default_prob > 0.35):
        return UnderwritingAction(action_type=ActionType.COUNTER,
                                  applicant_id=obs.applicant_id,
                                  params={"revised_amount_fraction": 0.75, "revised_rate_delta": 1.5},
                                  reasoning=f"Fairness rebalance (gap={fairness_gap:+.0%})")
    # P5 Post-shock
    if obs.macro_shock_active and obs.xgb_default_prob > 0.30:
        return UnderwritingAction(action_type=ActionType.COUNTER,
                                  applicant_id=obs.applicant_id,
                                  params={"revised_amount_fraction": 0.75, "revised_rate_delta": 2.0},
                                  reasoning="Post-shock conservative counter")
    # P6 Standard rules
    if obs.fico_score < 580:
        return _rej(RejectReason.LOW_CREDIT_SCORE.value, "FICO < 580")
    if obs.dti_ratio > 0.50:
        return _rej(RejectReason.HIGH_DTI.value, "DTI > 50%")
    if obs.xgb_default_prob > 0.40 or obs.fico_score < 620 or obs.dti_ratio > 0.40:
        frac = 0.70 if obs.xgb_default_prob > 0.50 else 0.82
        return UnderwritingAction(action_type=ActionType.COUNTER,
                                  applicant_id=obs.applicant_id,
                                  params={"revised_amount_fraction": frac, "revised_rate_delta": 1.5},
                                  reasoning="Moderate risk — counter offer")
    return UnderwritingAction(action_type=ActionType.APPROVE,
                              applicant_id=obs.applicant_id,
                              params={"amount_fraction": 1.0},
                              reasoning="Creditworthy — full approval")


def _risk_bar(prob: float) -> str:
    pct = int(prob * 100)
    c = "#16a34a" if prob < 0.25 else "#d97706" if prob < 0.50 else "#dc2626"
    return (f'<div style="background:#e5e7eb;border-radius:6px;height:12px;width:100%;margin:4px 0 2px;">'
            f'<div style="background:{c};width:{pct}%;height:12px;border-radius:6px;"></div></div>'
            f'<span style="font-size:0.8rem;color:{c};font-weight:700;">{pct}% default probability</span>')


def _fico_color(f):
    if f >= 720: return "#16a34a"
    if f >= 660: return "#65a30d"
    if f >= 620: return "#d97706"
    return "#dc2626"


def _card(title, value, color="#1e293b", sub=""):
    return (f'<div style="background:#fff;border:1px solid #e2e8f0;border-radius:10px;padding:12px;text-align:center;">'
            f'<div style="font-size:0.72rem;color:#64748b;margin-bottom:4px;">{title}</div>'
            f'<div style="font-size:1.5rem;font-weight:800;color:{color};">{value}</div>'
            + (f'<div style="font-size:0.72rem;color:#94a3b8;">{sub}</div>' if sub else "")
            + '</div>')


def _obs_html(obs: LoanObservation) -> str:
    shock_badge = (
        f'<span style="background:#7c3aed;color:#fff;padding:3px 10px;border-radius:12px;font-size:0.8rem;font-weight:700;">'
        f'⚡ RATE SHOCK +{obs.shock_magnitude_bps}bps</span>'
        if obs.macro_shock_active else
        '<span style="background:#dcfce7;color:#16a34a;padding:3px 10px;border-radius:12px;font-size:0.8rem;">No Shock</span>'
    )
    fraud_c = "#dc2626" if obs.fraud_ring_score > 0.50 else "#16a34a"
    fraud_icon = "🚨" if obs.fraud_ring_score > 0.70 else ("⚠️" if obs.fraud_ring_score > 0.40 else "✅")
    shap_c = "#dc2626" if obs.shap_top_value > 0 else "#16a34a"
    grp = str(obs.demographic_group)
    gc = {"group_a": "#3b82f6", "group_b": "#8b5cf6", "group_c": "#ec4899"}.get(grp, "#64748b")
    is_prot = grp in ("group_b", "group_c")
    dti_c = "#dc2626" if obs.dti_ratio > 0.43 else "#d97706" if obs.dti_ratio > 0.36 else "#16a34a"
    dti_lbl = "⚠️ High" if obs.dti_ratio > 0.43 else ("📊 Borderline" if obs.dti_ratio > 0.36 else "✅ OK")
    purpose = str(obs.loan_purpose).replace("LoanPurpose.", "").upper()

    return f"""
<div style="font-family:sans-serif;padding:18px;background:#f8fafc;border-radius:14px;border:1px solid #e2e8f0;">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:14px;flex-wrap:wrap;gap:8px;">
    <h3 style="margin:0;color:#0f172a;font-size:1.1rem;">
      Applicant <code style="background:#e2e8f0;padding:3px 8px;border-radius:6px;">{obs.applicant_id}</code>
      <span style="font-size:0.75rem;color:#94a3b8;margin-left:6px;">Step {obs.step_number} · {obs.steps_remaining} left</span>
    </h3>
    {shock_badge}
  </div>
  <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;margin-bottom:12px;">
    {_card("FICO Score", str(obs.fico_score), _fico_color(obs.fico_score), "Credit score")}
    {_card("Annual Income", f"${obs.income:,.0f}", "#1e293b", "USD/year")}
    {_card("Loan Amount", f"${obs.loan_amount:,.0f}", "#1e293b", purpose)}
  </div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:12px;">
    <div style="background:#fff;border:1px solid #e2e8f0;border-radius:10px;padding:12px;">
      <div style="font-size:0.72rem;color:#64748b;margin-bottom:2px;">DTI Ratio</div>
      <div style="font-size:1.2rem;font-weight:700;color:{dti_c};">{obs.dti_ratio:.1%} <span style="font-size:0.8rem;">{dti_lbl}</span></div>
    </div>
    <div style="background:#fff;border:1px solid #e2e8f0;border-radius:10px;padding:12px;">
      <div style="font-size:0.72rem;color:#64748b;margin-bottom:2px;">Employment</div>
      <div style="font-size:1.2rem;font-weight:700;">{obs.employment_years:.1f} <span style="font-size:0.8rem;color:#64748b;">yrs</span></div>
    </div>
    <div style="background:#fff;border:1px solid #e2e8f0;border-radius:10px;padding:12px;">
      <div style="font-size:0.72rem;color:#64748b;margin-bottom:2px;">Derogatory Marks</div>
      <div style="font-size:1.2rem;font-weight:700;color:{'#dc2626' if obs.num_derogatory_marks > 1 else '#d97706' if obs.num_derogatory_marks == 1 else '#16a34a'};">
        {obs.num_derogatory_marks} {'❌' if obs.num_derogatory_marks > 1 else '⚠️' if obs.num_derogatory_marks == 1 else '✅'}
      </div>
    </div>
    <div style="background:#fff;border:1px solid #e2e8f0;border-radius:10px;padding:12px;">
      <div style="font-size:0.72rem;color:#64748b;margin-bottom:2px;">Demographic Group</div>
      <span style="background:{gc}22;color:{gc};padding:3px 10px;border-radius:8px;font-weight:700;font-size:0.85rem;">{grp}</span>
      <span style="font-size:0.7rem;color:#94a3b8;margin-left:4px;">{'protected' if is_prot else 'reference'}</span>
    </div>
  </div>
  <div style="background:#fff;border:1px solid #e2e8f0;border-radius:10px;padding:12px;margin-bottom:10px;">
    <div style="font-size:0.72rem;color:#64748b;margin-bottom:2px;">XGBoost Default Probability</div>
    {_risk_bar(obs.xgb_default_prob)}
    <div style="font-size:0.72rem;color:#94a3b8;margin-top:4px;">
      Top driver: <b style="color:{shap_c};">{obs.shap_top_feature}</b>
      <span style="color:{shap_c};">{obs.shap_top_value:+.2f}</span>
    </div>
  </div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;">
    <div style="background:#fff;border:1px solid #e2e8f0;border-radius:10px;padding:12px;">
      <div style="font-size:0.72rem;color:#64748b;margin-bottom:4px;">Fraud Ring Score</div>
      <div style="font-weight:700;color:{fraud_c};">{obs.fraud_ring_score:.1%} {fraud_icon}</div>
    </div>
    <div style="background:#fff;border:1px solid #e2e8f0;border-radius:10px;padding:12px;">
      <div style="font-size:0.72rem;color:#64748b;margin-bottom:4px;">Credit Utilization</div>
      <div style="font-weight:700;color:{'#dc2626' if obs.credit_utilization > 0.75 else '#16a34a'};">{obs.credit_utilization:.0%}</div>
    </div>
  </div>
</div>"""


def _portfolio_html(env_state) -> str:
    ecl_usage = env_state.portfolio_ecl / max(env_state.portfolio_ecl_budget, 1e-8)
    ecl_pct = min(ecl_usage * 100, 100)
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
    both = ref_dec >= 1 and prot_dec >= 1
    if not both:
        gap_html = ('<div style="background:#fefce8;border:1px solid #fde68a;border-radius:8px;'
                    'padding:10px;font-size:0.82rem;color:#92400e;">'
                    '⚖️ Fairness Gap: not enough data yet — need ≥1 decision per group</div>')
    else:
        gc = "#16a34a" if abs(gap) < 0.12 else ("#dc2626" if gap > 0 else "#d97706")
        gl = (f"{gap:+.0%} ✅ OK" if abs(gap) < 0.12
              else (f"{gap:+.0%} ⚠️ Protected under-approved" if gap > 0
                    else f"{gap:+.0%} ⚠️ Protected over-approved"))
        gap_html = (f'<div style="background:#fff;border:1px solid #e2e8f0;border-radius:8px;padding:10px;">'
                    f'<div style="font-size:0.72rem;color:#64748b;margin-bottom:4px;">⚖️ Fairness Gap</div>'
                    f'<div style="font-weight:700;color:{gc};">{gl}</div>'
                    f'<div style="font-size:0.7rem;color:#94a3b8;margin-top:3px;">'
                    f'Ref: {ref_rate:.0%} ({ref_dec} dec) · Prot: {prot_rate:.0%} ({prot_dec} dec)'
                    f'</div></div>')
    return f"""
<div style="font-family:sans-serif;padding:16px;background:#f8fafc;border-radius:14px;border:1px solid #e2e8f0;">
  <h4 style="margin:0 0 12px;color:#0f172a;">📊 Portfolio Dashboard</h4>
  <div style="margin-bottom:12px;">
    <div style="display:flex;justify-content:space-between;font-size:0.8rem;margin-bottom:4px;">
      <span style="color:#64748b;">ECL Budget Used</span>
      <span style="font-weight:700;color:{ecl_c};">{ecl_usage:.0%} of {env_state.portfolio_ecl_budget:.0%}</span>
    </div>
    <div style="background:#e5e7eb;border-radius:8px;height:10px;">
      <div style="background:{ecl_c};width:{ecl_pct:.1f}%;height:10px;border-radius:8px;"></div>
    </div>
    <div style="font-size:0.7rem;color:#94a3b8;margin-top:2px;">ECL: {env_state.portfolio_ecl:.4%}</div>
  </div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:10px;font-size:0.82rem;">
    <div style="background:#fff;border:1px solid #e2e8f0;border-radius:8px;padding:8px;"><div style="color:#64748b;font-size:0.72rem;">Step</div><div style="font-weight:700;">{env_state.step}</div></div>
    <div style="background:#fff;border:1px solid #e2e8f0;border-radius:8px;padding:8px;"><div style="color:#64748b;font-size:0.72rem;">Total Reward</div><div style="font-weight:700;color:{'#16a34a' if env_state.total_reward >= 0 else '#dc2626'};">{env_state.total_reward:+.3f}</div></div>
    <div style="background:#fff;border:1px solid #e2e8f0;border-radius:8px;padding:8px;"><div style="color:#64748b;font-size:0.72rem;">Fraud Caught</div><div style="font-weight:700;color:#16a34a;">{env_state.fraud_caught} 🎯</div></div>
    <div style="background:#fff;border:1px solid #e2e8f0;border-radius:8px;padding:8px;"><div style="color:#64748b;font-size:0.72rem;">Fraud Missed</div><div style="font-weight:700;color:{'#dc2626' if env_state.fraud_missed > 0 else '#16a34a'};">{env_state.fraud_missed}</div></div>
  </div>
  {gap_html}
</div>"""


def _history_html(history: list) -> str:
    if not history:
        return "<p style='color:#94a3b8;font-style:italic;padding:12px;'>No decisions yet.</p>"
    rows = ""
    for h in reversed(history[-20:]):
        bg, fg = ACTION_COLOURS.get(h["action"], ("#6b7280", "#fff"))
        rwd_c = "#16a34a" if h["reward"] > 0 else "#dc2626"
        rows += (f'<tr style="border-bottom:1px solid #f1f5f9;">'
                 f'<td style="padding:7px 10px;font-size:0.82rem;color:#475569;font-family:monospace;">{h["applicant_id"]}</td>'
                 f'<td style="padding:7px 10px;"><span style="background:{bg};color:{fg};padding:3px 10px;border-radius:10px;font-size:0.78rem;font-weight:700;">{h["action"]}</span></td>'
                 f'<td style="padding:7px 10px;font-size:0.82rem;font-weight:600;">{h["fico"]}</td>'
                 f'<td style="padding:7px 10px;font-size:0.82rem;font-weight:700;color:{rwd_c};">{h["reward"]:+.3f}</td>'
                 f'<td style="padding:7px 10px;font-size:0.78rem;color:#64748b;">{str(h["reasoning"])[:60]}</td></tr>')
    return (f'<div style="overflow-x:auto;font-family:sans-serif;"><table style="width:100%;border-collapse:collapse;">'
            f'<thead><tr style="background:#f1f5f9;border-bottom:2px solid #e2e8f0;">'
            f'<th style="padding:8px 10px;text-align:left;font-size:0.78rem;color:#475569;">Applicant</th>'
            f'<th style="padding:8px 10px;text-align:left;font-size:0.78rem;color:#475569;">Decision</th>'
            f'<th style="padding:8px 10px;text-align:left;font-size:0.78rem;color:#475569;">FICO</th>'
            f'<th style="padding:8px 10px;text-align:left;font-size:0.78rem;color:#475569;">Reward</th>'
            f'<th style="padding:8px 10px;text-align:left;font-size:0.78rem;color:#475569;">Reasoning</th>'
            f'</tr></thead><tbody>{rows}</tbody></table></div>')


def _end_screen(env_state, task_id: str) -> str:
    ecl_usage = env_state.portfolio_ecl / max(env_state.portfolio_ecl_budget, 1e-8)
    ref_dec   = env_state.decisions_by_group.get("group_a", 0)
    prot_dec  = (env_state.decisions_by_group.get("group_b", 0)
                 + env_state.decisions_by_group.get("group_c", 0))
    ref_rate  = env_state.approvals_by_group.get("group_a", 0) / max(ref_dec, 1)
    prot_rate = ((env_state.approvals_by_group.get("group_b", 0)
                  + env_state.approvals_by_group.get("group_c", 0)) / max(prot_dec, 1))
    gap = abs(ref_rate - prot_rate)
    checks = [
        ("ECL Budget",      f"{ecl_usage:.0%} used",                           ecl_usage <= 1.0),
        ("Fraud Detection", f"{env_state.fraud_caught} caught / {env_state.fraud_missed} missed", env_state.fraud_missed == 0),
        ("Fairness Gap",    f"{gap:.0%} (target <12%)",                         gap <= 0.12),
        ("Total Reward",    f"{env_state.total_reward:+.3f}",                   env_state.total_reward > 0),
    ]
    rows = "".join(
        f'<div style="display:flex;justify-content:space-between;padding:8px 0;border-bottom:1px solid #f1f5f9;">'
        f'<span style="color:#475569;">{n}</span>'
        f'<span style="font-weight:700;color:{"#16a34a" if ok else "#dc2626"};">{"✅" if ok else "❌"} {v}</span></div>'
        for n, v, ok in checks
    )
    stars = sum(1 for *_, ok in checks if ok)
    star_str = "🌟🌟🌟" if stars == 4 else "⭐⭐" if stars >= 3 else "⭐"
    return (f'<div style="background:linear-gradient(135deg,#f0fdf4,#eff6ff);border:2px solid #86efac;'
            f'border-radius:16px;padding:24px;text-align:center;font-family:sans-serif;">'
            f'<div style="font-size:2.5rem;margin-bottom:8px;">{star_str}</div>'
            f'<h2 style="color:#15803d;margin:0 0 4px;">Episode Complete!</h2>'
            f'<p style="color:#475569;margin:0 0 16px;">Task: <b>{task_id.upper()}</b> · Steps: {env_state.step}</p>'
            f'<div style="background:#fff;border-radius:10px;padding:12px 16px;margin-bottom:16px;text-align:left;">{rows}</div>'
            f'<p style="color:#94a3b8;font-size:0.85rem;">Click <b>Start New Episode</b> to play again!</p>'
            f'</div>')


def _fresh():
    return {"env": None, "obs": None, "history": [], "done": False, "task_id": "easy"}


def _all_btns(on: bool):
    return [gr.update(interactive=on)] * 5


def _start_episode(task_id: str, state: dict):
    env = CreditLensEnv(task_id=task_id, seed=42)
    obs = env.reset()
    state.update({"env": env, "obs": obs, "history": [], "done": False, "task_id": task_id})
    cfg = TASK_CONFIGS[task_id]
    task_html = (f'<div style="background:#eff6ff;border:1px solid #bfdbfe;border-radius:10px;'
                 f'padding:12px 14px;font-family:sans-serif;">'
                 f'<b style="color:#1e40af;">📋 {cfg.name}</b>'
                 f'<div style="font-size:0.82rem;color:#475569;margin-top:4px;">'
                 f'👥 {cfg.num_applicants} applicants · ⏱️ {cfg.max_steps} max steps · '
                 f'💰 {cfg.ecl_budget:.0%} ECL budget · '
                 f'{"🚨 fraud rings" if cfg.fraud_ring_size else "✅ no fraud"} · '
                 f'{"⚡ rate shock" if cfg.macro_shock else "✅ no shock"}'
                 f'</div></div>')
    return (_obs_html(obs), _portfolio_html(env.state()), _history_html([]),
            task_html, *_all_btns(True), state)


def _apply(action: UnderwritingAction, state: dict):
    env = state["env"]
    obs = state["obs"]
    result = env.step(action)
    state["history"].append({
        "applicant_id": obs.applicant_id,
        "action": str(action.action_type).replace("ActionType.", ""),
        "fico": obs.fico_score,
        "reward": result.reward,
        "reasoning": action.reasoning or "",
    })
    env_state = env.state()
    if result.done or result.observation is None:
        state["done"] = True
        return (_end_screen(env_state, state["task_id"]),
                _portfolio_html(env_state), _history_html(state["history"]),
                *_all_btns(False), state)
    state["obs"] = result.observation
    return (_obs_html(result.observation), _portfolio_html(env_state),
            _history_html(state["history"]), *_all_btns(True), state)


def _no_ep(state):
    ph = "<p style='color:#94a3b8;padding:20px;'>Start an episode first.</p>"
    return (ph, ph, _history_html(state.get("history", [])), *_all_btns(False), state)


def _auto_decide(state):
    if state["done"] or not state["obs"]: return _no_ep(state)
    return _apply(_rule_based_decide(state["obs"]), state)

def _do_approve(frac, state):
    if state["done"] or not state["obs"]: return _no_ep(state)
    obs = state["obs"]
    return _apply(UnderwritingAction(action_type=ActionType.APPROVE, applicant_id=obs.applicant_id,
                                     params={"amount_fraction": frac},
                                     reasoning=f"Manual APPROVE {frac:.0%}"), state)

def _do_reject(reason, state):
    if state["done"] or not state["obs"]: return _no_ep(state)
    obs = state["obs"]
    return _apply(UnderwritingAction(action_type=ActionType.REJECT, applicant_id=obs.applicant_id,
                                     params={"reason_code": reason},
                                     reasoning=f"Manual REJECT: {reason}"), state)

def _do_counter(frac, delta, state):
    if state["done"] or not state["obs"]: return _no_ep(state)
    obs = state["obs"]
    return _apply(UnderwritingAction(action_type=ActionType.COUNTER, applicant_id=obs.applicant_id,
                                     params={"revised_amount_fraction": frac, "revised_rate_delta": delta},
                                     reasoning=f"Manual COUNTER {frac:.0%} +{delta:.1f}%"), state)

def _do_info(field, state):
    if state["done"] or not state["obs"]: return _no_ep(state)
    obs = state["obs"]
    return _apply(UnderwritingAction(action_type=ActionType.REQUEST_INFO, applicant_id=obs.applicant_id,
                                     params={"field_name": field},
                                     reasoning=f"Requesting: {field}"), state)


with gr.Blocks(title="CreditLens — AI Loan Underwriting") as demo:
    _state = gr.State(_fresh())

    gr.HTML("""
<div style="text-align:center;padding:28px 20px 12px;font-family:sans-serif;
     background:linear-gradient(135deg,#0f172a,#1e3a5f);border-radius:16px;margin-bottom:16px;">
  <div style="font-size:2.5rem;margin-bottom:6px;">🏦</div>
  <h1 style="margin:0;font-size:2rem;font-weight:900;color:#f8fafc;">CreditLens</h1>
  <p style="color:#94a3b8;margin:8px 0 0;font-size:0.95rem;">
    AI loan underwriting — balance <b style="color:#60a5fa;">credit risk</b>,
    <b style="color:#a78bfa;">fairness</b>, <b style="color:#f87171;">fraud detection</b>
    &amp; <b style="color:#fbbf24;">macro shocks</b>
  </p>
</div>""")

    with gr.Row():
        with gr.Column(scale=2):
            _task_dd = gr.Dropdown(choices=["easy","medium","hard"], value="easy",
                                   label="📋 Select Task",
                                   info="Easy: baseline | Medium: rate shock | Hard: fraud ring + shock")
            _start_btn = gr.Button("🚀 Start New Episode", variant="primary", size="lg")
            _task_info = gr.HTML("<p style='color:#94a3b8;font-size:0.85rem;padding:6px;'>Select a task and click Start.</p>")
        with gr.Column(scale=3):
            _portfolio_out = gr.HTML("<p style='color:#94a3b8;padding:16px;'>Portfolio metrics appear here.</p>")

    gr.HTML("<hr style='border:none;border-top:1px solid #e2e8f0;margin:6px 0;'>")

    with gr.Row(equal_height=False):
        with gr.Column(scale=3):
            _obs_out = gr.HTML("""<div style='padding:40px;color:#94a3b8;text-align:center;font-family:sans-serif;'>
  <div style='font-size:3.5rem;'>🏦</div>
  <p style='font-size:1rem;margin-top:8px;'>Start an episode to see your first loan applicant.</p>
</div>""")
        with gr.Column(scale=2):
            gr.Markdown("### 🤖 AI Auto-Decision")
            _auto_btn = gr.Button("⚡ AI Decides (Rule-Based Agent)", variant="secondary",
                                  interactive=False, size="lg")
            gr.Markdown("### ✅ Approve")
            _approve_frac = gr.Slider(0.5, 1.0, value=1.0, step=0.05, label="Amount Fraction")
            _approve_btn  = gr.Button("✅ Approve", interactive=False, variant="primary")
            gr.Markdown("### ❌ Reject")
            _reject_reason = gr.Dropdown(choices=[r.value for r in RejectReason],
                                         value="HIGH_DTI", label="Reason Code (ECOA compliant)")
            _reject_btn = gr.Button("❌ Reject", interactive=False)
            gr.Markdown("### 🔄 Counter Offer")
            _counter_frac  = gr.Slider(0.25, 0.9, value=0.70, step=0.05, label="Revised Amount Fraction")
            _counter_delta = gr.Slider(0.0,  5.0, value=1.5,  step=0.25, label="Rate Delta (%)")
            _counter_btn   = gr.Button("🔄 Counter", interactive=False)
            gr.Markdown("### 📄 Request Info")
            _info_field = gr.Dropdown(choices=[f.value for f in RequestField],
                                      value="income_proof", label="Document (costs 1 step)")
            _info_btn = gr.Button("📄 Request Info", interactive=False)

    gr.HTML("<hr style='border:none;border-top:1px solid #e2e8f0;margin:6px 0;'>")
    gr.Markdown("### 📜 Decision History")
    _history_out = gr.HTML("<p style='color:#94a3b8;font-style:italic;padding:12px;'>No decisions yet.</p>")

    _OUTS = [_obs_out, _portfolio_out, _history_out,
             _auto_btn, _approve_btn, _reject_btn, _counter_btn, _info_btn, _state]

    _start_btn.click(fn=_start_episode, inputs=[_task_dd, _state],
                     outputs=[_obs_out, _portfolio_out, _history_out, _task_info,
                              _auto_btn, _approve_btn, _reject_btn, _counter_btn, _info_btn, _state])
    _auto_btn.click(fn=_auto_decide,   inputs=[_state],                               outputs=_OUTS)
    _approve_btn.click(fn=_do_approve, inputs=[_approve_frac, _state],                outputs=_OUTS)
    _reject_btn.click(fn=_do_reject,   inputs=[_reject_reason, _state],               outputs=_OUTS)
    _counter_btn.click(fn=_do_counter, inputs=[_counter_frac, _counter_delta, _state], outputs=_OUTS)
    _info_btn.click(fn=_do_info,       inputs=[_info_field, _state],                  outputs=_OUTS)


# ══════════════════════════════════════════════════════════════════════════
# SECTION 3 — Mount Gradio into FastAPI, expose main()
#
# The validator checks ALL of:
#   [x] server/app.py exists
#   [x] main() function is defined and callable
#   [x] if __name__ == '__main__': main()   ← bottom of file
#   [x] Entry point NOT referencing creditlens.inference.service:app
# ══════════════════════════════════════════════════════════════════════════

# Mount Gradio UI at root; FastAPI REST routes stay at /reset, /step, etc.
app = gr.mount_gradio_app(api, demo, path="/")


def main():
    """
    Entry point called by the OpenEnv validator and by direct execution.
    Starts uvicorn serving both the FastAPI REST API and the Gradio UI.
    """
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(
        app,            # the combined FastAPI + Gradio ASGI app
        host="0.0.0.0",
        port=port,
        log_level="info",
    )


if __name__ == "__main__":
    main()