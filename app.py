"""
CreditLens — Gradio Demo for Hugging Face Spaces  (v2)
Interactive loan underwriting demo with rule-based AI agent.
No Ollama required — uses the hardened rule-based fallback.

Fixes vs v1:
  * Fairness gap cold-start: shows "Not enough data" until ≥1 decision per group
  * Score breakdown panel shown after episode ends
  * Clearer portfolio metrics with explanatory tooltips
  * Fairness gap direction explained inline
"""

from __future__ import annotations

import sys
import gradio as gr
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from creditlens.models import (
    ActionType,
    LoanObservation,
    RejectReason,
    RequestField,
    UnderwritingAction,
)
from creditlens.env.engine import CreditLensEnv, TASK_CONFIGS

# ── Colour palette ─────────────────────────────────────────────────────────
ACTION_COLOURS = {
    "APPROVE":      ("#16a34a", "#ffffff"),
    "REJECT":       ("#dc2626", "#ffffff"),
    "COUNTER":      ("#d97706", "#ffffff"),
    "REQUEST_INFO": ("#2563eb", "#ffffff"),
}

# ── Rule-based agent ───────────────────────────────────────────────────────
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

    # P1: Fraud
    if obs.fraud_ring_score > 0.70 or (obs.shared_phone and obs.shared_employer_id):
        return _rej(RejectReason.FRAUD_SUSPECTED.value,
                    f"Fraud score {obs.fraud_ring_score:.0%} — ring detected")

    # P2: XGB hard cutoff
    xgb_cut = 0.62 - shock_adj
    if obs.xgb_default_prob > xgb_cut:
        return _rej(RejectReason.HIGH_DTI.value,
                    f"XGB {obs.xgb_default_prob:.0%} > cutoff {xgb_cut:.0%}")

    # P3: ECL budget guard
    ecl_t = 1.1
    if ecl_usage > 0.95:   ecl_t = 0.10
    elif ecl_usage > 0.80: ecl_t = 0.15
    elif ecl_usage > 0.60: ecl_t = 0.35
    ecl_t -= shock_adj
    if obs.xgb_default_prob > ecl_t:
        return _rej(RejectReason.HIGH_DTI.value,
                    f"ECL budget {ecl_usage:.0%} used — protecting portfolio")

    # P4: Fairness rescue (genuinely under-served protected group only)
    if (fairness_gap > 0.12 and is_protected and
            obs.xgb_default_prob < 0.60 and obs.fico_score >= 600 and
            obs.dti_ratio < 0.50 and obs.fraud_ring_score < 0.50):
        return UnderwritingAction(action_type=ActionType.APPROVE,
                                  applicant_id=obs.applicant_id,
                                  params={"amount_fraction": 0.90},
                                  reasoning=f"Fairness rescue — protected under-served (gap={fairness_gap:+.0%})")

    if (fairness_gap < -0.12 and is_protected and obs.xgb_default_prob > 0.35):
        return UnderwritingAction(action_type=ActionType.COUNTER,
                                  applicant_id=obs.applicant_id,
                                  params={"revised_amount_fraction": 0.75, "revised_rate_delta": 1.5},
                                  reasoning=f"Fairness rebalance — protected over-approved (gap={fairness_gap:+.0%})")

    # P5: Post-shock conservatism
    if obs.macro_shock_active and obs.xgb_default_prob > 0.30:
        return UnderwritingAction(action_type=ActionType.COUNTER,
                                  applicant_id=obs.applicant_id,
                                  params={"revised_amount_fraction": 0.75, "revised_rate_delta": 2.0},
                                  reasoning="Post-shock: conservative counter offer")

    # P6: Standard FICO/DTI rules
    if obs.fico_score < 580:
        return _rej(RejectReason.LOW_CREDIT_SCORE.value, "FICO < 580 — below minimum")
    if obs.dti_ratio > 0.50:
        return _rej(RejectReason.HIGH_DTI.value, "DTI > 50% — too high")
    if obs.xgb_default_prob > 0.40 or obs.fico_score < 620 or obs.dti_ratio > 0.40:
        frac = 0.70 if obs.xgb_default_prob > 0.50 else 0.82
        return UnderwritingAction(action_type=ActionType.COUNTER,
                                  applicant_id=obs.applicant_id,
                                  params={"revised_amount_fraction": frac, "revised_rate_delta": 1.5},
                                  reasoning="Moderate risk — counter offer at reduced amount")
    return UnderwritingAction(action_type=ActionType.APPROVE,
                              applicant_id=obs.applicant_id,
                              params={"amount_fraction": 1.0},
                              reasoning="Creditworthy — full approval")


# ── HTML helpers ───────────────────────────────────────────────────────────
def _risk_bar(prob: float) -> str:
    pct = int(prob * 100)
    c = "#16a34a" if prob < 0.25 else "#d97706" if prob < 0.50 else "#dc2626"
    return (f'<div style="background:#e5e7eb;border-radius:6px;height:12px;width:100%;margin:4px 0 2px;">'
            f'<div style="background:{c};width:{pct}%;height:12px;border-radius:6px;"></div></div>'
            f'<span style="font-size:0.8rem;color:{c};font-weight:700;">{pct}% default probability</span>')

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
        f'<span style="background:#7c3aed;color:#fff;padding:3px 10px;border-radius:12px;font-size:0.8rem;font-weight:700;">'
        f'⚡ RATE SHOCK +{obs.shock_magnitude_bps}bps</span>'
        if obs.macro_shock_active else
        '<span style="background:#dcfce7;color:#16a34a;padding:3px 10px;border-radius:12px;font-size:0.8rem;">No Shock</span>'
    )
    fraud_c = "#dc2626" if obs.fraud_ring_score > 0.50 else "#16a34a"
    fraud_icon = "🚨" if obs.fraud_ring_score > 0.70 else ("⚠️" if obs.fraud_ring_score > 0.40 else "✅")
    shap_c = "#dc2626" if obs.shap_top_value > 0 else "#16a34a"
    group_colors = {"group_a": "#3b82f6", "group_b": "#8b5cf6", "group_c": "#ec4899"}
    grp = str(obs.demographic_group)
    group_c = group_colors.get(grp, "#64748b")
    is_prot = grp in ("group_b", "group_c")

    dti_c = "#dc2626" if obs.dti_ratio > 0.43 else "#d97706" if obs.dti_ratio > 0.36 else "#16a34a"
    dti_label = "⚠️ High" if obs.dti_ratio > 0.43 else ("📊 Borderline" if obs.dti_ratio > 0.36 else "✅ OK")

    purpose = str(obs.loan_purpose).replace("LoanPurpose.", "").upper()

    return f"""
<div style="font-family:sans-serif;padding:18px;background:#f8fafc;border-radius:14px;border:1px solid #e2e8f0;">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:14px;flex-wrap:wrap;gap:8px;">
    <h3 style="margin:0;color:#0f172a;font-size:1.1rem;">
      Applicant <code style="background:#e2e8f0;padding:3px 8px;border-radius:6px;">{obs.applicant_id}</code>
      <span style="font-size:0.75rem;color:#94a3b8;margin-left:8px;">Step {obs.step_number} · {obs.steps_remaining} remaining</span>
    </h3>
    {shock_badge}
  </div>

  <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;margin-bottom:12px;">
    {_mini_card("FICO Score", str(obs.fico_score), _fico_color(obs.fico_score), "Credit score")}
    {_mini_card("Annual Income", f"${obs.income:,.0f}", "#1e293b", "USD/year")}
    {_mini_card("Loan Amount", f"${obs.loan_amount:,.0f}", "#1e293b", purpose)}
  </div>

  <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:12px;">
    <div style="background:#fff;border:1px solid #e2e8f0;border-radius:10px;padding:12px;">
      <div style="font-size:0.72rem;color:#64748b;margin-bottom:2px;">DTI Ratio <span style="color:#94a3b8;">(debt-to-income)</span></div>
      <div style="font-size:1.2rem;font-weight:700;color:{dti_c};">{obs.dti_ratio:.1%} <span style="font-size:0.8rem;">{dti_label}</span></div>
    </div>
    <div style="background:#fff;border:1px solid #e2e8f0;border-radius:10px;padding:12px;">
      <div style="font-size:0.72rem;color:#64748b;margin-bottom:2px;">Employment</div>
      <div style="font-size:1.2rem;font-weight:700;">{obs.employment_years:.1f} <span style="font-size:0.8rem;color:#64748b;">years</span></div>
    </div>
    <div style="background:#fff;border:1px solid #e2e8f0;border-radius:10px;padding:12px;">
      <div style="font-size:0.72rem;color:#64748b;margin-bottom:2px;">Derogatory Marks</div>
      <div style="font-size:1.2rem;font-weight:700;color:{'#dc2626' if obs.num_derogatory_marks > 1 else '#d97706' if obs.num_derogatory_marks == 1 else '#16a34a'};">
        {obs.num_derogatory_marks} {'❌' if obs.num_derogatory_marks > 1 else '⚠️' if obs.num_derogatory_marks == 1 else '✅'}
      </div>
    </div>
    <div style="background:#fff;border:1px solid #e2e8f0;border-radius:10px;padding:12px;">
      <div style="font-size:0.72rem;color:#64748b;margin-bottom:2px;">Demographic Group <span style="color:#94a3b8;">(fairness tracking only)</span></div>
      <span style="background:{group_c}22;color:{group_c};padding:3px 10px;border-radius:8px;font-weight:700;font-size:0.85rem;">{grp}</span>
      <span style="font-size:0.7rem;color:#94a3b8;margin-left:4px;">{'protected' if is_prot else 'reference'}</span>
    </div>
  </div>

  <div style="background:#fff;border:1px solid #e2e8f0;border-radius:10px;padding:12px;margin-bottom:10px;">
    <div style="font-size:0.72rem;color:#64748b;margin-bottom:2px;">XGBoost Default Probability <span style="color:#94a3b8;">(ML risk model)</span></div>
    {_risk_bar(obs.xgb_default_prob)}
    <div style="font-size:0.72rem;color:#94a3b8;margin-top:4px;">
      Top SHAP driver: <b style="color:{shap_c};">{obs.shap_top_feature}</b>
      <span style="color:{shap_c};">{obs.shap_top_value:+.2f}</span>
      &nbsp;<span style="color:#94a3b8;">({'increases risk' if obs.shap_top_value > 0 else 'reduces risk'})</span>
    </div>
  </div>

  <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;">
    <div style="background:#fff;border:1px solid #e2e8f0;border-radius:10px;padding:12px;">
      <div style="font-size:0.72rem;color:#64748b;margin-bottom:4px;">Fraud Ring Score</div>
      <div style="font-weight:700;color:{fraud_c};">{obs.fraud_ring_score:.1%} {fraud_icon}</div>
      <div style="font-size:0.7rem;color:#94a3b8;">{'shared phone+employer!' if obs.shared_phone and obs.shared_employer_id else 'No shared fraud signals'}</div>
    </div>
    <div style="background:#fff;border:1px solid #e2e8f0;border-radius:10px;padding:12px;">
      <div style="font-size:0.72rem;color:#64748b;margin-bottom:4px;">Credit Utilization</div>
      <div style="font-weight:700;color:{'#dc2626' if obs.credit_utilization > 0.75 else '#16a34a'};">{obs.credit_utilization:.0%}</div>
      <div style="font-size:0.7rem;color:#94a3b8;">Open accounts: {obs.num_open_accounts}</div>
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

    # ── COLD START FIX: don't show gap until both groups have real data ──
    both_have_data = ref_dec >= 1 and prot_dec >= 1
    if not both_have_data:
        gap_section = f"""
<div style="background:#fefce8;border:1px solid #fde68a;border-radius:8px;padding:10px;font-size:0.82rem;">
  <b style="color:#000;">⚖️ Fairness Gap:</b>
  <span style="color:#000;"> Not enough data yet</span><br>
  <span style="color:#000;font-size:0.75rem;">
    Need at least 1 decision for both reference (group_a) and protected (group_b/c) groups.<br>
    Current: group_a={ref_dec} decision(s), group_b/c={prot_dec} decision(s)
  </span>
</div>"""
    else:
        if abs(gap) < 0.12:
            gap_color, gap_label = "#16a34a", f"{gap:+.0%} ✅ Within ±12% threshold"
        elif gap > 0:
            gap_color = "#dc2626"
            gap_label = f"{gap:+.0%} ⚠️ Protected group under-approved vs reference"
        else:
            gap_color = "#d97706"
            gap_label = f"{gap:+.0%} ⚠️ Protected group over-approved vs reference"

        gap_section = f"""
<div style="background:#fff;border:1px solid #e2e8f0;border-radius:8px;padding:10px;">
  <div style="font-size:0.72rem;color:#64748b;margin-bottom:4px;">
    ⚖️ Fairness Gap <span style="color:#94a3b8;">(ref_rate − prot_rate, target: within ±12%)</span>
  </div>
  <div style="font-weight:700;color:{gap_color};font-size:1rem;">{gap_label}</div>
  <div style="font-size:0.72rem;color:#94a3b8;margin-top:4px;">
    Ref (group_a): {ref_rate:.0%} from {ref_dec} decisions &nbsp;|&nbsp;
    Prot (group_b/c): {prot_rate:.0%} from {prot_dec} decisions
  </div>
</div>"""

    shock_badge = (
        f'<span style="background:#7c3aed;color:#fff;padding:2px 8px;border-radius:8px;font-size:0.78rem;font-weight:600;">'
        f'⚡ SHOCK +{env_state.shock_magnitude_bps}bps</span>'
        if env_state.macro_shock_active else ""
    )

    return f"""
<div style="font-family:sans-serif;padding:16px;background:#f8fafc;border-radius:14px;border:1px solid #e2e8f0;">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">
    <h4 style="margin:0;color:#0f172a;">📊 Portfolio Dashboard</h4>
    {shock_badge}
  </div>

  <div style="margin-bottom:12px;">
    <div style="display:flex;justify-content:space-between;font-size:0.8rem;margin-bottom:4px;">
      <span style="color:#64748b;">ECL Budget Used</span>
      <span style="font-weight:700;color:{ecl_c};">{ecl_usage:.0%} of {env_state.portfolio_ecl_budget:.0%}</span>
    </div>
    <div style="background:#e5e7eb;border-radius:8px;height:10px;">
      <div style="background:{ecl_c};width:{ecl_pct:.1f}%;height:10px;border-radius:8px;"></div>
    </div>
    <div style="font-size:0.7rem;color:#94a3b8;margin-top:2px;">Actual ECL: {env_state.portfolio_ecl:.4%}</div>
  </div>

  <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:10px;font-size:0.82rem;">
    <div style="background:#fff;border:1px solid #e2e8f0;border-radius:8px;padding:8px;">
      <div style="color:#64748b;font-size:0.72rem;">Step</div>
      <div style="font-weight:700;">{env_state.step}</div>
    </div>
    <div style="background:#fff;border:1px solid #e2e8f0;border-radius:8px;padding:8px;">
      <div style="color:#64748b;font-size:0.72rem;">Total Reward</div>
      <div style="font-weight:700;color:{'#16a34a' if env_state.total_reward >= 0 else '#dc2626'};">{env_state.total_reward:+.3f}</div>
    </div>
    <div style="background:#fff;border:1px solid #e2e8f0;border-radius:8px;padding:8px;">
      <div style="color:#64748b;font-size:0.72rem;">Fraud Caught</div>
      <div style="font-weight:700;color:#16a34a;">{env_state.fraud_caught} 🎯</div>
    </div>
    <div style="background:#fff;border:1px solid #e2e8f0;border-radius:8px;padding:8px;">
      <div style="color:#64748b;font-size:0.72rem;">Fraud Missed</div>
      <div style="font-weight:700;color:{'#dc2626' if env_state.fraud_missed > 0 else '#16a34a'};">{env_state.fraud_missed} {'💀' if env_state.fraud_missed else '✅'}</div>
    </div>
  </div>

  {gap_section}
</div>"""


def _history_html(history: list) -> str:
    if not history:
        return "<p style='color:#94a3b8;font-style:italic;padding:12px;'>No decisions yet — start an episode.</p>"
    rows = ""
    for h in reversed(history[-20:]):
        bg, fg = ACTION_COLOURS.get(h["action"], ("#6b7280", "#fff"))
        rwd_c = "#16a34a" if h["reward"] > 0 else "#dc2626"
        rows += f"""<tr style="border-bottom:1px solid #f1f5f9;">
  <td style="padding:7px 10px;font-size:0.82rem;color:#475569;font-family:monospace;">{h['applicant_id']}</td>
  <td style="padding:7px 10px;">
    <span style="background:{bg};color:{fg};padding:3px 10px;border-radius:10px;font-size:0.78rem;font-weight:700;">{h['action']}</span>
  </td>
  <td style="padding:7px 10px;font-size:0.82rem;font-weight:600;">{h['fico']}</td>
  <td style="padding:7px 10px;font-size:0.82rem;font-weight:700;color:{rwd_c};">{h['reward']:+.3f}</td>
  <td style="padding:7px 10px;font-size:0.78rem;color:#64748b;max-width:260px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">{h['reasoning']}</td>
</tr>"""
    return f"""<div style="overflow-x:auto;font-family:sans-serif;">
<table style="width:100%;border-collapse:collapse;">
  <thead><tr style="background:#f1f5f9;border-bottom:2px solid #e2e8f0;">
    <th style="padding:8px 10px;text-align:left;font-size:0.78rem;color:#475569;">Applicant</th>
    <th style="padding:8px 10px;text-align:left;font-size:0.78rem;color:#475569;">Decision</th>
    <th style="padding:8px 10px;text-align:left;font-size:0.78rem;color:#475569;">FICO</th>
    <th style="padding:8px 10px;text-align:left;font-size:0.78rem;color:#475569;">Reward</th>
    <th style="padding:8px 10px;text-align:left;font-size:0.78rem;color:#475569;">AI Reasoning</th>
  </tr></thead>
  <tbody>{rows}</tbody>
</table></div>"""


def _end_screen(env_state, task_id: str) -> str:
    ecl_usage = env_state.portfolio_ecl / max(env_state.portfolio_ecl_budget, 1e-8)
    ref_dec   = env_state.decisions_by_group.get("group_a", 0)
    prot_dec  = (env_state.decisions_by_group.get("group_b", 0)
                 + env_state.decisions_by_group.get("group_c", 0))
    ref_rate  = env_state.approvals_by_group.get("group_a", 0) / max(ref_dec, 1)
    prot_rate = ((env_state.approvals_by_group.get("group_b", 0)
                  + env_state.approvals_by_group.get("group_c", 0)) / max(prot_dec, 1))
    gap       = abs(ref_rate - prot_rate)

    checks = [
        ("ECL Budget",       f"{ecl_usage:.0%} used",                          ecl_usage <= 1.0),
        ("Fraud Detection",  f"{env_state.fraud_caught} caught / {env_state.fraud_missed} missed", env_state.fraud_missed == 0),
        ("Fairness Gap",     f"{gap:.0%} (target <12%)",                        gap <= 0.12),
        ("Total Reward",     f"{env_state.total_reward:+.3f}",                  env_state.total_reward > 0),
    ]
    rows = "".join(
        f'<div style="display:flex;justify-content:space-between;padding:8px 0;border-bottom:1px solid #f1f5f9;">'
        f'<span style="color:#475569;">{name}</span>'
        f'<span style="font-weight:700;color:{"#16a34a" if ok else "#dc2626"};">{"✅" if ok else "❌"} {val}</span></div>'
        for name, val, ok in checks
    )
    stars = sum(1 for *_, ok in checks if ok)
    star_str = "🌟🌟🌟" if stars == 4 else "⭐⭐" if stars >= 3 else "⭐"

    return f"""
<div style="background:linear-gradient(135deg,#f0fdf4,#eff6ff);border:2px solid #86efac;
     border-radius:16px;padding:24px;text-align:center;font-family:sans-serif;">
  <div style="font-size:2.5rem;margin-bottom:8px;">{star_str}</div>
  <h2 style="color:#15803d;margin:0 0 4px;">Episode Complete!</h2>
  <p style="color:#475569;margin:0 0 16px;">Task: <b>{task_id.upper()}</b> &nbsp;|&nbsp; Steps: {env_state.step}</p>
  <div style="background:#fff;border-radius:10px;padding:12px 16px;margin-bottom:16px;text-align:left;">{rows}</div>
  <p style="color:#94a3b8;font-size:0.85rem;margin:0;">
    Click <b>Start New Episode</b> to play again, or try a harder task!
  </p>
</div>"""


# ── State ──────────────────────────────────────────────────────────────────
def _fresh():
    return {"env": None, "obs": None, "history": [], "done": False, "task_id": "easy"}

def _all_btns(on: bool):
    return [gr.update(interactive=on)] * 5


# ── Gradio functions ───────────────────────────────────────────────────────
def start_episode(task_id: str, state: dict):
    env = CreditLensEnv(task_id=task_id, seed=42)
    obs = env.reset()
    state.update({"env": env, "obs": obs, "history": [], "done": False, "task_id": task_id})
    cfg = TASK_CONFIGS[task_id]
    task_html = f"""
<div style="background:#eff6ff;border:1px solid #bfdbfe;border-radius:10px;
padding:12px 14px;font-family:sans-serif;color:#000;">

<b style="color:#000;">📋 {cfg.name}</b>

<div style="font-size:0.85rem;margin-top:6px;display:flex;flex-wrap:wrap;gap:10px;color:#000;">

<span style="background:#dbeafe;color:#000;padding:4px 10px;border-radius:6px;font-weight:600;">
👥 {cfg.num_applicants} applicants
</span>

<span style="background:#dbeafe;color:#000;padding:4px 10px;border-radius:6px;font-weight:600;">
⏱️ {cfg.max_steps} max steps
</span>

<span style="background:#dbeafe;color:#000;padding:4px 10px;border-radius:6px;font-weight:600;">
💰 {cfg.ecl_budget:.0%} ECL budget
</span>

<span style="background:#dbeafe;color:#000;padding:4px 10px;border-radius:6px;font-weight:600;">
{"🚨 " + str(cfg.fraud_ring_size) + " fraud ring(s)" if cfg.fraud_ring_size else "✅ No fraud rings"}
</span>

<span style="background:#dbeafe;color:#000;padding:4px 10px;border-radius:6px;font-weight:600;">
{"⚡ Rate shock at step " + str(cfg.shock_step) if cfg.macro_shock else "✅ No macro shock"}
</span>

</div>
</div>
"""
    return (
        _obs_html(obs),
        _portfolio_html(env.state()),
        _history_html(state["history"]),
        task_html,
        *_all_btns(True),
        state,
    )


def _apply(action: UnderwritingAction, state: dict):
    env  = state["env"]
    obs  = state["obs"]
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
                _portfolio_html(env_state),
                _history_html(state["history"]),
                *_all_btns(False), state)
    state["obs"] = result.observation
    return (_obs_html(result.observation),
            _portfolio_html(env_state),
            _history_html(state["history"]),
            *_all_btns(True), state)


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
                                     reasoning=f"Manual APPROVE at {frac:.0%}"), state)

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


# ── Build UI ───────────────────────────────────────────────────────────────
with gr.Blocks(title="CreditLens — AI Loan Underwriting") as demo:
    state = gr.State(_fresh())

    gr.HTML("""
<div style="text-align:center;padding:28px 20px 12px;font-family:sans-serif;
     background:linear-gradient(135deg,#0f172a,#1e3a5f);border-radius:16px;margin-bottom:16px;">
  <div style="font-size:2.5rem;margin-bottom:6px;">🏦</div>
  <h1 style="margin:0;font-size:2rem;font-weight:900;color:#f8fafc;">CreditLens</h1>
  <p style="color:#94a3b8;margin:8px 0 0;font-size:0.95rem;">
    AI loan underwriting simulation — balance
    <b style="color:#60a5fa;">credit risk</b>,
    <b style="color:#a78bfa;">fairness</b>,
    <b style="color:#f87171;">fraud detection</b> &amp;
    <b style="color:#fbbf24;">macro shocks</b>
  </p>
</div>""")

    with gr.Row():
        with gr.Column(scale=2):
            task_dd   = gr.Dropdown(choices=["easy","medium","hard"], value="easy",
                                    label="📋 Select Task",
                                    info="Easy: baseline  |  Medium: rate shock  |  Hard: fraud ring + shock")
            start_btn = gr.Button("🚀 Start New Episode", variant="primary", size="lg")
            task_info = gr.HTML("<p style='color:#94a3b8;font-size:0.85rem;padding:6px;'>Select a task and click Start.</p>")
        with gr.Column(scale=3):
            portfolio_out = gr.HTML("<p style='color:#94a3b8;padding:16px;'>Portfolio metrics appear here.</p>")

    gr.HTML("<hr style='border:none;border-top:1px solid #e2e8f0;margin:6px 0;'>")

    with gr.Row(equal_height=False):
        with gr.Column(scale=3):
            obs_out = gr.HTML("""
<div style='padding:40px;color:#94a3b8;text-align:center;font-family:sans-serif;'>
  <div style='font-size:3.5rem;'>🏦</div>
  <p style='font-size:1rem;margin-top:8px;'>Start an episode to see your first loan applicant.</p>
</div>""")
        with gr.Column(scale=2):
            gr.Markdown("### 🤖 AI Auto-Decision")
            gr.HTML("""<div style="font-size:0.8rem;color:#475569;padding:8px 10px;background:#f1f5f9;
                     border-radius:8px;margin-bottom:8px;line-height:1.6;">
              <b>6 priority rules:</b><br>
              1 Fraud &nbsp;·&nbsp; 2 XGB cutoff &nbsp;·&nbsp; 3 ECL guard<br>
              4 Fairness &nbsp;·&nbsp; 5 Post-shock &nbsp;·&nbsp; 6 FICO/DTI
            </div>""")
            auto_btn    = gr.Button("⚡ AI Decides (Rule-Based Agent)", variant="secondary",
                                    interactive=False, size="lg")

            gr.Markdown("### ✅ Approve")
            approve_frac = gr.Slider(0.5, 1.0, value=1.0, step=0.05, label="Amount Fraction")
            approve_btn  = gr.Button("✅ Approve", interactive=False, variant="primary")

            gr.Markdown("### ❌ Reject")
            reject_reason = gr.Dropdown(choices=[r.value for r in RejectReason],
                                        value="HIGH_DTI", label="Reason Code (ECOA compliant)")
            reject_btn    = gr.Button("❌ Reject", interactive=False)

            gr.Markdown("### 🔄 Counter Offer")
            counter_frac  = gr.Slider(0.25, 0.9,  value=0.70, step=0.05, label="Revised Amount Fraction")
            counter_delta = gr.Slider(0.0,  5.0,  value=1.5,  step=0.25, label="Rate Delta (%)")
            counter_btn   = gr.Button("🔄 Counter", interactive=False)

            gr.Markdown("### 📄 Request Info")
            info_field = gr.Dropdown(choices=[f.value for f in RequestField],
                                     value="income_proof", label="Document (costs 1 step)")
            info_btn   = gr.Button("📄 Request Info", interactive=False)

    gr.HTML("<hr style='border:none;border-top:1px solid #e2e8f0;margin:6px 0;'>")
    gr.Markdown("### 📜 Decision History")
    history_out = gr.HTML("<p style='color:#94a3b8;font-style:italic;padding:12px;'>No decisions yet.</p>")

    gr.HTML("""
<div style="text-align:center;padding:14px;font-size:0.78rem;color:#94a3b8;font-family:sans-serif;
     border-top:1px solid #f1f5f9;margin-top:8px;">
  CreditLens — OpenEnv AI Underwriting &nbsp;|&nbsp; Built with Gradio &nbsp;|&nbsp;
  Rule-based agent (no external LLM required)
</div>""")

    OUTS = [obs_out, portfolio_out, history_out,
            auto_btn, approve_btn, reject_btn, counter_btn, info_btn, state]

    start_btn.click(fn=start_episode, inputs=[task_dd, state],
                    outputs=[obs_out, portfolio_out, history_out, task_info,
                             auto_btn, approve_btn, reject_btn, counter_btn, info_btn, state])
    auto_btn.click(fn=auto_decide,   inputs=[state],                          outputs=OUTS)
    approve_btn.click(fn=do_approve, inputs=[approve_frac, state],            outputs=OUTS)
    reject_btn.click(fn=do_reject,   inputs=[reject_reason, state],           outputs=OUTS)
    counter_btn.click(fn=do_counter, inputs=[counter_frac, counter_delta, state], outputs=OUTS)
    info_btn.click(fn=do_info,       inputs=[info_field, state],              outputs=OUTS)


if __name__ == "__main__":
    demo.launch()