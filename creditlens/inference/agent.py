"""
CreditLens — Ollama LLM Agent  (v4 — medium fairness gap fix)

Changes from v3:
  * Fairness override now checks DIRECTION of gap before rescuing.
    v3 bug: Override-B fired even when protected_rate >= ref_rate,
    creating INVERTED gap (protected 30%+ above reference) → fairness_score=0.000.
    Fix: only rescue if prot_rate < ref_rate - threshold (genuinely under-served).
    New Override-D2: when protected OVER-approved, downgrade to COUNTER to rebalance.
  * ECL too conservative: actual_ecl=0.021 vs target=0.050 (57% under).
    Relaxed XGB hard cutoff: 0.55 → 0.62 so borderline applicants get COUNTER not REJECT.
  * Compact user prompt with signed gap (+/- direction visible to LLM).
  * Override chain: Fraud → XGB cutoff → ECL guard → Fairness D1/D2 → Post-shock.
"""

from __future__ import annotations

import json
import re
from typing import Optional

import httpx
from loguru import logger
from openai import OpenAI

from creditlens.models import (
    ActionType,
    LoanObservation,
    RejectReason,
    RequestField,
    UnderwritingAction,
)

OLLAMA_BASE_URL = "http://localhost:11434"
PREFERRED_MODELS = ["llama3.2", "llama3.1", "llama3", "mistral", "gemma2", "phi3"]


def _detect_ollama_model() -> Optional[str]:
    try:
        resp = httpx.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5.0)
        if resp.status_code == 200:
            models = [m["name"].split(":")[0] for m in resp.json().get("models", [])]
            for pref in PREFERRED_MODELS:
                if pref in models:
                    return pref
            return models[0] if models else None
    except Exception:
        return None


def _build_system_prompt() -> str:
    return """You are a senior AI loan underwriter at a major bank. Make exactly ONE decision per applicant.

ACTIONS:
  APPROVE      params: {"amount_fraction": 0.5-1.0}
  REJECT       params: {"reason_code": "HIGH_DTI"|"LOW_CREDIT_SCORE"|"INSUFFICIENT_INCOME"|"FRAUD_SUSPECTED"|"INCOMPLETE_APPLICATION"|"RECENT_DEROGATORY"}
  COUNTER      params: {"revised_amount_fraction": 0.25-0.9, "revised_rate_delta": 0.0-5.0}
  REQUEST_INFO params: {"field_name": "income_proof"|"employment_letter"|"bank_statements"|"tax_returns"|"identity_document"}

=== DECISION RULES — apply in order, stop at first match ===

1. FRAUD (overrides everything):
   fraud_ring_score > 0.70 → REJECT FRAUD_SUSPECTED
   shared_phone=true AND shared_employer=true → REJECT FRAUD_SUSPECTED

2. ECL BUDGET GUARD (protect the portfolio):
   ECL usage > 95% → REJECT if xgb_default_prob > 0.10
   ECL usage > 80% → REJECT if xgb_default_prob > 0.15
   ECL usage > 60% → REJECT if xgb_default_prob > 0.35

3. XGB HARD CUTOFF:
   xgb_default_prob > 0.62 → REJECT (HIGH_DTI or LOW_CREDIT_SCORE)
   NEVER use COUNTER when xgb > 0.62 — partial approval still creates ECL exposure.

4. POST-SHOCK (when macro_shock_active=true):
   Tighten all thresholds by 15 percentage points.
   xgb > 0.47 → REJECT (not just > 0.62)
   Any APPROVE with xgb > 0.30 → COUNTER at 0.75 fraction instead.
   Goal: 70%+ of post-shock decisions = REJECT or COUNTER.

5. FAIRNESS BALANCE — read the gap DIRECTION carefully:
   gap shown in prompt = approval_rate_reference MINUS approval_rate_protected.
   Positive gap (+N%) means protected group is UNDER-approved vs reference.
   Negative gap (-N%) means protected group is OVER-approved vs reference.

   If gap > +12% AND this applicant is group_b or group_c:
     AND xgb < 0.60 AND fico >= 600 AND dti < 0.50 → APPROVE at 0.90 (fairness rescue)

   If gap < -12% AND this applicant is group_b or group_c:
     AND xgb > 0.35 → COUNTER instead of APPROVE (rebalance without discriminating)

6. STANDARD RULES:
   FICO < 580 → REJECT LOW_CREDIT_SCORE
   DTI > 0.50 → REJECT HIGH_DTI
   xgb > 0.45 OR FICO 580-619 → COUNTER at 0.65-0.75 fraction
   FICO 620-679 → APPROVE 0.85-0.95 or COUNTER 0.80
   FICO >= 680 AND xgb < 0.40 → APPROVE 1.0

OUTPUT: respond ONLY with this JSON, no other text:
{"action_type": "APPROVE", "params": {"amount_fraction": 1.0}, "reasoning": "one sentence"}"""


def _build_user_prompt(obs: LoanObservation) -> str:
    ecl_usage = obs.portfolio_ecl / max(obs.portfolio_ecl_budget, 1e-8)
    # Signed gap: positive = reference approved more = protected under-served
    # Negative = protected approved more = inverted gap, need rebalancing
    fairness_gap = obs.approval_rate_reference - obs.approval_rate_protected

    alerts = []
    if obs.macro_shock_active:
        alerts.append(f"POST-SHOCK+{obs.shock_magnitude_bps}bps:tighten-15pp COUNTER-approve-xgb>0.30")
    if ecl_usage > 0.95:
        alerts.append(f"ECL-CRITICAL({ecl_usage:.0%}):REJECT-xgb>0.10")
    elif ecl_usage > 0.80:
        alerts.append(f"ECL-HIGH({ecl_usage:.0%}):REJECT-xgb>0.15")
    elif ecl_usage > 0.60:
        alerts.append(f"ECL-ELEVATED({ecl_usage:.0%}):REJECT-xgb>0.35")
    if obs.xgb_default_prob > 0.62:
        alerts.append(f"XGB={obs.xgb_default_prob:.0%}>0.62:MUST-REJECT")
    if obs.fraud_ring_score > 0.70:
        alerts.append(f"FRAUD={obs.fraud_ring_score:.0%}:MUST-REJECT-FRAUD_SUSPECTED")
    if fairness_gap > 0.12 and obs.demographic_group in ("group_b", "group_c"):
        alerts.append(f"FAIRNESS-UNDER-SERVED(gap={fairness_gap:+.0%}):APPROVE-xgb<0.60-fico>600")
    elif fairness_gap < -0.12 and obs.demographic_group in ("group_b", "group_c"):
        alerts.append(f"FAIRNESS-OVER-SERVED(gap={fairness_gap:+.0%}):COUNTER-if-xgb>0.35")

    alert_str = " | ".join(alerts) if alerts else "no-alerts"

    return (
        f"ID={obs.applicant_id} group={obs.demographic_group} "
        f"FICO={obs.fico_score} income=${obs.income:,.0f} loan=${obs.loan_amount:,.0f} "
        f"purpose={obs.loan_purpose} emp={obs.employment_years:.1f}yr "
        f"DTI={obs.dti_ratio:.1%} util={obs.credit_utilization:.1%} "
        f"pay_hist={obs.payment_history_score:.2f} derog={obs.num_derogatory_marks} "
        f"XGB={obs.xgb_default_prob:.1%} shap={obs.shap_top_feature}({obs.shap_top_value:+.2f}) "
        f"fraud={obs.fraud_ring_score:.1%} sh_phone={obs.shared_phone} "
        f"sh_empl={obs.shared_employer_id} cluster={obs.graph_cluster_size} "
        f"fed={obs.fed_funds_rate:.2f}% shock={obs.macro_shock_active}+{obs.shock_magnitude_bps}bps "
        f"ECL={obs.portfolio_ecl:.3%}/budget={obs.portfolio_ecl_budget:.1%}({ecl_usage:.0%}used) "
        f"ref_rate={obs.approval_rate_reference:.0%} prot_rate={obs.approval_rate_protected:.0%} "
        f"gap={fairness_gap:+.0%} steps_left={obs.steps_remaining} "
        f"ALERTS:{alert_str}"
    )


def _parse_action(response_text: str, obs: LoanObservation) -> UnderwritingAction:
    """
    4-stage JSON repair pipeline.
    Root cause of most failures: llama3.2 embeds bare newlines inside reasoning strings.
    Stage 4a (replace newlines) fixes this. extra_body={"format":"json"} prevents it at source.
    """
    text = re.sub(r"```(?:json)?", "", response_text).strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    text = match.group() if match else text

    data = None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        pass

    if data is None:
        r = text
        r = r.replace("\n", " ").replace("\r", " ")             # 4a: newline root-cause fix
        r = re.sub(r",\s*([}\]])", r"\1", r)                    # 4b: trailing commas
        r = re.sub(r"'([^'\\]*(?:\\.[^'\\]*)*)'", r'"\1"', r)   # 4c: single → double quotes
        r = re.sub(r"//[^\n]*", " ", r)                         # 4d: JS-style comments
        try:
            data = json.loads(r)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON repair failed: {e}. Rule-based fallback.")
            return _rule_based_fallback(obs)

    try:
        action_type = ActionType(data["action_type"].upper())
        params = data.get("params", {})
        reasoning = data.get("reasoning", "")

        # ── Normalise params ─────────────────────────────────────────────────
        if action_type == ActionType.APPROVE:
            frac = float(params.get("amount_fraction", 1.0))
            params = {"amount_fraction": max(0.5, min(1.0, frac))}
        elif action_type == ActionType.REJECT:
            reason = params.get("reason_code", "HIGH_DTI")
            valid = [r.value for r in RejectReason]
            params = {"reason_code": reason if reason in valid else "HIGH_DTI"}
        elif action_type == ActionType.COUNTER:
            frac  = float(params.get("revised_amount_fraction", 0.7))
            delta = float(params.get("revised_rate_delta", 1.5))
            params = {
                "revised_amount_fraction": max(0.25, min(0.9, frac)),
                "revised_rate_delta": max(0.0, min(5.0, delta)),
            }
        elif action_type == ActionType.REQUEST_INFO:
            field = params.get("field_name", "income_proof")
            valid = [f.value for f in RequestField]
            params = {"field_name": field if field in valid else "income_proof"}

        # ── Post-parse hard overrides (priority order) ────────────────────────
        ecl_usage = obs.portfolio_ecl / max(obs.portfolio_ecl_budget, 1e-8)
        shock_adj = 0.15 if obs.macro_shock_active else 0.0
        # Signed gap: + means reference approved more (protected under-served)
        # Negative means protected approved more (inverted gap — must rebalance)
        fairness_gap = obs.approval_rate_reference - obs.approval_rate_protected
        is_protected = obs.demographic_group in ("group_b", "group_c")

        # Override A: Fraud — absolute highest priority
        if action_type != ActionType.REJECT and (
            obs.fraud_ring_score > 0.70 or
            (obs.shared_phone and obs.shared_employer_id)
        ):
            action_type = ActionType.REJECT
            params = {"reason_code": RejectReason.FRAUD_SUSPECTED.value}
            reasoning = f"[A] fraud={obs.fraud_ring_score:.0%} → REJECT"
            logger.info(f"Override-A FRAUD score={obs.fraud_ring_score:.2f}")

        else:
            # Override B: XGB hard cutoff (tightened post-shock)
            xgb_hard = 0.62 - shock_adj  # 0.62 normal → 0.47 post-shock
            if (action_type in (ActionType.APPROVE, ActionType.COUNTER)
                    and obs.xgb_default_prob > xgb_hard):
                action_type = ActionType.REJECT
                params = {"reason_code": RejectReason.HIGH_DTI.value}
                reasoning = f"[B] XGB={obs.xgb_default_prob:.0%}>{xgb_hard:.0%} → REJECT"
                logger.info(f"Override-B XGB xgb={obs.xgb_default_prob:.2f} thresh={xgb_hard:.2f}")

            # Override C: ECL budget guard (3 tiers, shock-adjusted)
            ecl_thresh = 1.1  # sentinel: no restriction when ECL is fine
            if ecl_usage > 0.95:   ecl_thresh = 0.10
            elif ecl_usage > 0.80: ecl_thresh = 0.15
            elif ecl_usage > 0.60: ecl_thresh = 0.35
            ecl_thresh -= shock_adj

            if (action_type in (ActionType.APPROVE, ActionType.COUNTER)
                    and obs.xgb_default_prob > ecl_thresh):
                action_type = ActionType.REJECT
                params = {"reason_code": RejectReason.HIGH_DTI.value}
                reasoning = f"[C] ECL={ecl_usage:.0%} xgb={obs.xgb_default_prob:.0%}>{ecl_thresh:.0%} → REJECT"
                logger.info(f"Override-C ECL ecl={ecl_usage:.2f} xgb={obs.xgb_default_prob:.2f}")

            # Override D1: Fairness rescue — protected group genuinely UNDER-approved
            # Only fires when gap is POSITIVE (reference > protected) above threshold
            if (action_type == ActionType.REJECT and
                    fairness_gap > 0.12 and is_protected and
                    obs.xgb_default_prob < 0.60 and obs.fico_score >= 600 and
                    obs.dti_ratio < 0.50 and obs.fraud_ring_score < 0.50):
                action_type = ActionType.APPROVE
                params = {"amount_fraction": 0.90}
                reasoning = f"[D1] Fairness rescue gap={fairness_gap:+.0%} xgb={obs.xgb_default_prob:.0%}"
                logger.info(f"Override-D1 RESCUE gap={fairness_gap:.2f} xgb={obs.xgb_default_prob:.2f}")

            # Override D2: Fairness rebalance — protected group OVER-approved (inverted gap)
            # v3 bug: rescue fired here too → gap flipped to -30% → fairness_score=0
            # Fix: when gap is NEGATIVE (protected > reference) and this is a protected
            # applicant being approved, downgrade to COUNTER to rebalance rates.
            elif (action_type == ActionType.APPROVE and
                    fairness_gap < -0.12 and is_protected and
                    obs.xgb_default_prob > 0.35):
                frac = max(0.70, params.get("amount_fraction", 1.0) - 0.15)
                action_type = ActionType.COUNTER
                params = {"revised_amount_fraction": frac, "revised_rate_delta": 1.5}
                reasoning = f"[D2] Rebalance over-served gap={fairness_gap:+.0%} → COUNTER"
                logger.info(f"Override-D2 REBALANCE gap={fairness_gap:.2f}")

            # Override E: Post-shock conservatism — borderline APPROVEs → COUNTER
            if (obs.macro_shock_active and
                    action_type == ActionType.APPROVE and
                    obs.xgb_default_prob > 0.30):
                frac = max(0.70, params.get("amount_fraction", 1.0) - 0.15)
                action_type = ActionType.COUNTER
                params = {"revised_amount_fraction": frac, "revised_rate_delta": 2.0}
                reasoning = f"[E] Post-shock APPROVE→COUNTER xgb={obs.xgb_default_prob:.0%}"
                logger.info(f"Override-E POST-SHOCK xgb={obs.xgb_default_prob:.2f}")

        return UnderwritingAction(
            action_type=action_type,
            applicant_id=obs.applicant_id,
            params=params,
            reasoning=reasoning,
        )

    except Exception as e:
        logger.warning(f"Parse error: {e}. Fallback.")
        return _rule_based_fallback(obs)


def _rule_based_fallback(obs: LoanObservation) -> UnderwritingAction:
    """
    Hardened rule-based fallback — same 6-priority logic as the override chain.
    Used when Ollama is unavailable OR when JSON output is unrecoverable.
    """
    ecl_usage = obs.portfolio_ecl / max(obs.portfolio_ecl_budget, 1e-8)
    shock_adj = 0.15 if obs.macro_shock_active else 0.0
    fairness_gap = obs.approval_rate_reference - obs.approval_rate_protected
    is_protected = obs.demographic_group in ("group_b", "group_c")

    # P1: Fraud
    if obs.fraud_ring_score > 0.70 or (obs.shared_phone and obs.shared_employer_id):
        return UnderwritingAction(
            action_type=ActionType.REJECT,
            applicant_id=obs.applicant_id,
            params={"reason_code": RejectReason.FRAUD_SUSPECTED.value},
            reasoning="Fallback P1: fraud signal",
        )

    # P2: XGB hard cutoff
    xgb_cutoff = 0.62 - shock_adj
    if obs.xgb_default_prob > xgb_cutoff:
        return UnderwritingAction(
            action_type=ActionType.REJECT,
            applicant_id=obs.applicant_id,
            params={"reason_code": RejectReason.HIGH_DTI.value},
            reasoning=f"Fallback P2: XGB={obs.xgb_default_prob:.0%}>{xgb_cutoff:.0%}",
        )

    # P3: ECL budget guard
    ecl_thresh = 1.1
    if ecl_usage > 0.95:   ecl_thresh = 0.10
    elif ecl_usage > 0.80: ecl_thresh = 0.15
    elif ecl_usage > 0.60: ecl_thresh = 0.35
    ecl_thresh -= shock_adj
    if obs.xgb_default_prob > ecl_thresh:
        return UnderwritingAction(
            action_type=ActionType.REJECT,
            applicant_id=obs.applicant_id,
            params={"reason_code": RejectReason.HIGH_DTI.value},
            reasoning=f"Fallback P3: ECL guard ecl={ecl_usage:.0%}",
        )

    # P4: Fairness direction-aware
    if (fairness_gap > 0.12 and is_protected and
            obs.xgb_default_prob < 0.60 and obs.fico_score >= 600 and
            obs.dti_ratio < 0.50 and obs.fraud_ring_score < 0.50):
        return UnderwritingAction(
            action_type=ActionType.APPROVE,
            applicant_id=obs.applicant_id,
            params={"amount_fraction": 0.90},
            reasoning="Fallback P4: fairness rescue",
        )
    if (fairness_gap < -0.12 and is_protected and obs.xgb_default_prob > 0.35):
        return UnderwritingAction(
            action_type=ActionType.COUNTER,
            applicant_id=obs.applicant_id,
            params={"revised_amount_fraction": 0.75, "revised_rate_delta": 1.5},
            reasoning="Fallback P4: fairness rebalance (over-served)",
        )

    # P5: Post-shock conservatism
    if obs.macro_shock_active and obs.xgb_default_prob > 0.30:
        return UnderwritingAction(
            action_type=ActionType.COUNTER,
            applicant_id=obs.applicant_id,
            params={"revised_amount_fraction": 0.75, "revised_rate_delta": 2.0},
            reasoning="Fallback P5: post-shock conservative counter",
        )

    # P6: Standard FICO/DTI rules
    if obs.fico_score < 580:
        return UnderwritingAction(
            action_type=ActionType.REJECT,
            applicant_id=obs.applicant_id,
            params={"reason_code": RejectReason.LOW_CREDIT_SCORE.value},
            reasoning="Fallback P6: FICO<580",
        )
    if obs.dti_ratio > 0.50:
        return UnderwritingAction(
            action_type=ActionType.REJECT,
            applicant_id=obs.applicant_id,
            params={"reason_code": RejectReason.HIGH_DTI.value},
            reasoning="Fallback P6: DTI>50%",
        )
    if obs.xgb_default_prob > 0.40 or obs.fico_score < 620 or obs.dti_ratio > 0.40:
        frac = 0.70 if obs.xgb_default_prob > 0.50 else 0.82
        return UnderwritingAction(
            action_type=ActionType.COUNTER,
            applicant_id=obs.applicant_id,
            params={"revised_amount_fraction": frac, "revised_rate_delta": 1.5},
            reasoning="Fallback P6: moderate risk counter",
        )

    return UnderwritingAction(
        action_type=ActionType.APPROVE,
        applicant_id=obs.applicant_id,
        params={"amount_fraction": 1.0},
        reasoning="Fallback P6: creditworthy",
    )


class OllamaAgent:
    """
    LLM underwriting agent — OpenAI SDK routed to local Ollama /v1 endpoint.
    Contest-compliant (OpenAI client required), zero cost (Ollama is free and local).
    Falls back to hardened rule-based agent when Ollama is unavailable.
    """

    def __init__(self, model: Optional[str] = None, base_url: str = OLLAMA_BASE_URL):
        self.base_url = base_url
        self.model = model or _detect_ollama_model()
        self.use_ollama = self.model is not None
        # api_key="ollama" — Ollama ignores this value but OpenAI SDK requires non-empty
        self._client = OpenAI(base_url=f"{self.base_url}/v1", api_key="ollama")
        if self.use_ollama:
            logger.info(f"OllamaAgent v4 | model={self.model} | endpoint={self.base_url}/v1")
        else:
            logger.warning("Ollama not available — hardened rule-based fallback active")

    def decide(self, obs: LoanObservation) -> UnderwritingAction:
        if not self.use_ollama:
            return _rule_based_fallback(obs)
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": _build_system_prompt()},
                    {"role": "user",   "content": _build_user_prompt(obs)},
                ],
                temperature=0.05,           # near-deterministic — consistent decisions
                max_tokens=200,             # we only need the JSON object
                extra_body={"format": "json"},  # Ollama JSON mode: valid tokens only
            )
            return _parse_action(response.choices[0].message.content, obs)
        except Exception as e:
            logger.error(f"OpenAI client error: {e} — rule-based fallback")
            return _rule_based_fallback(obs)