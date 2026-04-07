"""
CreditLens — Ollama LLM Agent
Uses Ollama (100% free, runs locally) as the reasoning engine.
Calls Ollama via the OpenAI-compatible /v1 endpoint using the OpenAI SDK
— satisfies the contest requirement for OpenAI client usage without any
paid API key. Model: llama3.2 or mistral (auto-detects).
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional

import httpx
from loguru import logger
from openai import OpenAI
from rich.console import Console

from creditlens.models import (
    ActionType,
    LoanObservation,
    RejectReason,
    RequestField,
    UnderwritingAction,
)

console = Console()

OLLAMA_BASE_URL = "http://localhost:11434"
PREFERRED_MODELS = ["llama3.2", "llama3.1", "llama3", "mistral", "gemma2", "phi3"]


def _detect_ollama_model() -> Optional[str]:
    """Detect which models are available in local Ollama install."""
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
    return """You are an expert AI loan underwriter at a major bank.
You must make one underwriting decision per applicant based on the provided financial data.

Your decision must be one of:
- APPROVE: approve the loan (optionally at a reduced amount_fraction between 0.5 and 1.0)
- REJECT: decline with a reason code (HIGH_DTI, LOW_CREDIT_SCORE, INSUFFICIENT_INCOME, FRAUD_SUSPECTED, INCOMPLETE_APPLICATION, RECENT_DEROGATORY)
- COUNTER: counter-offer with revised_amount_fraction (0.25-0.9) and revised_rate_delta (extra %)
- REQUEST_INFO: ask for more information (field: income_proof, employment_letter, bank_statements, tax_returns, identity_document)

Risk guidance:
- FICO < 580: very high risk → likely REJECT
- FICO 580-619: high risk → consider COUNTER
- FICO 620-679: moderate risk → depends on DTI and income
- FICO ≥ 680: lower risk → consider APPROVE
- DTI > 43%: high debt burden → caution
- XGBoost default probability > 0.6: strong signal of default risk
- Fraud ring score > 0.5: suspicious → investigate or REJECT
- shared_phone=true AND shared_employer=true: very high fraud signal
- Post macro shock (macro_shock_active=true): REJECT or COUNTER any applicant with
  xgb_default_prob > 0.35 (not just > 0.45). The rate environment has tightened — your
  ECL budget resets tighter after a shock. Favour COUNTER over APPROVE on borderline cases.
- Maintain fairness: approval_rate_protected should not be more than 15% below approval_rate_reference

You must respond ONLY with valid JSON in this exact format:
{
  "action_type": "APPROVE" | "REJECT" | "COUNTER" | "REQUEST_INFO",
  "params": {
    // For APPROVE: {"amount_fraction": 1.0}
    // For REJECT: {"reason_code": "HIGH_DTI"}
    // For COUNTER: {"revised_amount_fraction": 0.7, "revised_rate_delta": 1.5}
    // For REQUEST_INFO: {"field_name": "income_proof"}
  },
  "reasoning": "Brief explanation of your decision"
}

Do not output anything except the JSON object."""


def _build_user_prompt(obs: LoanObservation) -> str:
    fairness_gap = obs.approval_rate_reference - obs.approval_rate_protected
    fairness_alert = ""
    if fairness_gap > 0.10:
        fairness_alert = f"\n⚠️ FAIRNESS ALERT: Protected group approval rate is {fairness_gap:.0%} below reference group — consider whether this rejection is well-justified."

    shock_alert = ""
    if obs.macro_shock_active:
        shock_alert = f"\n⚡ MACRO SHOCK ACTIVE: Fed raised rates +{obs.shock_magnitude_bps}bps. Be more conservative on variable-rate approvals."

    return f"""=== LOAN APPLICATION ===
Applicant ID: {obs.applicant_id}
FICO Score: {obs.fico_score}
Annual Income: ${obs.income:,.0f}
Loan Amount: ${obs.loan_amount:,.0f}
Loan Purpose: {obs.loan_purpose}
Employment Years: {obs.employment_years:.1f}
Debt-to-Income Ratio: {obs.dti_ratio:.2%}
Credit Utilization: {obs.credit_utilization:.2%}
Payment History Score: {obs.payment_history_score:.2f}
Open Accounts: {obs.num_open_accounts}
Derogatory Marks: {obs.num_derogatory_marks}
{"LTV Ratio: " + f"{obs.ltv_ratio:.2%}" if obs.ltv_ratio else ""}

=== ML RISK SIGNALS ===
XGBoost Default Probability: {obs.xgb_default_prob:.2%}
Top Risk Factor: {obs.shap_top_feature} (SHAP value: {obs.shap_top_value:.3f})

=== FRAUD SIGNALS ===
Fraud Ring Score: {obs.fraud_ring_score:.2%}
Shared Phone with Others: {obs.shared_phone}
Shared Employer with Others: {obs.shared_employer_id}
Connected Applicant Cluster Size: {obs.graph_cluster_size}

=== MACROECONOMIC CONTEXT ===
Fed Funds Rate: {obs.fed_funds_rate:.2f}%
10Y Treasury Yield: {obs.treasury_yield_10y:.2f}%
Unemployment Rate: {obs.unemployment_rate:.2f}%
{shock_alert}

=== PORTFOLIO STATE ===
Current ECL: {obs.portfolio_ecl:.4%} (Budget: {obs.portfolio_ecl_budget:.2%})
ECL Usage: {obs.portfolio_ecl / max(obs.portfolio_ecl_budget, 1e-8):.0%}
Reference Group Approval Rate: {obs.approval_rate_reference:.0%}
Protected Group Approval Rate: {obs.approval_rate_protected:.0%}
{fairness_alert}

Steps Remaining: {obs.steps_remaining}

Make your underwriting decision:"""


def _parse_action(response_text: str, obs: LoanObservation) -> UnderwritingAction:
    """
    Parse LLM JSON response into UnderwritingAction.
    Multi-stage repair pipeline before falling back to rules:
      1. Strip markdown code fences
      2. Extract outermost JSON object
      3. Direct parse
      4. Repair trailing commas + single-quoted strings and retry
    """
    # Stage 1: strip markdown fences
    text = re.sub(r"```(?:json)?", "", response_text).strip()

    # Stage 2: extract outermost JSON object
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        text = match.group()

    # Stage 3: direct parse
    data = None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        pass

    # Stage 4: repair common LLM mistakes and retry
    if data is None:
        repaired = text
        # trailing commas before } or ]
        repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
        # single-quoted strings → double-quoted
        repaired = re.sub(r"'([^']*)'", r'"\1"', repaired)
        try:
            data = json.loads(repaired)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON repair failed: {e}. Using rule-based fallback.")
            return _rule_based_fallback(obs)

    try:
        action_type = ActionType(data["action_type"].upper())
        params = data.get("params", {})
        reasoning = data.get("reasoning", "")

        # Validate params
        if action_type == ActionType.APPROVE:
            frac = float(params.get("amount_fraction", 1.0))
            params = {"amount_fraction": max(0.5, min(1.0, frac))}

        elif action_type == ActionType.REJECT:
            reason = params.get("reason_code", "HIGH_DTI")
            valid_reasons = [r.value for r in RejectReason]
            if reason not in valid_reasons:
                reason = "HIGH_DTI"
            params = {"reason_code": reason}

        elif action_type == ActionType.COUNTER:
            frac = float(params.get("revised_amount_fraction", 0.7))
            delta = float(params.get("revised_rate_delta", 1.5))
            params = {
                "revised_amount_fraction": max(0.25, min(0.9, frac)),
                "revised_rate_delta": max(0.0, min(5.0, delta)),
            }

        elif action_type == ActionType.REQUEST_INFO:
            field = params.get("field_name", "income_proof")
            valid_fields = [f.value for f in RequestField]
            if field not in valid_fields:
                field = "income_proof"
            params = {"field_name": field}

        return UnderwritingAction(
            action_type=action_type,
            applicant_id=obs.applicant_id,
            params=params,
            reasoning=reasoning,
        )

    except Exception as e:
        logger.warning(f"Failed to parse LLM response: {e}. Using rule-based fallback.")
        return _rule_based_fallback(obs)


def _rule_based_fallback(obs: LoanObservation) -> UnderwritingAction:
    """Simple rule-based fallback when LLM output is unparseable."""
    if obs.fraud_ring_score > 0.6 or (obs.shared_phone and obs.shared_employer_id):
        return UnderwritingAction(
            action_type=ActionType.REJECT,
            applicant_id=obs.applicant_id,
            params={"reason_code": RejectReason.FRAUD_SUSPECTED.value},
            reasoning="Rule-based fallback: high fraud signal",
        )
    if obs.fico_score < 580 or obs.dti_ratio > 0.5:
        return UnderwritingAction(
            action_type=ActionType.REJECT,
            applicant_id=obs.applicant_id,
            params={"reason_code": RejectReason.LOW_CREDIT_SCORE.value},
            reasoning="Rule-based fallback: low FICO/high DTI",
        )
    if obs.xgb_default_prob > 0.6:
        return UnderwritingAction(
            action_type=ActionType.COUNTER,
            applicant_id=obs.applicant_id,
            params={"revised_amount_fraction": 0.6, "revised_rate_delta": 2.0},
            reasoning="Rule-based fallback: moderate risk, counter-offer",
        )
    return UnderwritingAction(
        action_type=ActionType.APPROVE,
        applicant_id=obs.applicant_id,
        params={"amount_fraction": 1.0},
        reasoning="Rule-based fallback: appears creditworthy",
    )


class OllamaAgent:
    """
    LLM underwriting agent powered by Ollama (100% free, local inference).
    Uses the OpenAI-compatible /v1 endpoint so the official OpenAI SDK handles
    all HTTP — satisfies the contest OpenAI client requirement while costing $0.
    Falls back to rule-based decisions if Ollama is not running.
    """

    def __init__(self, model: Optional[str] = None, base_url: str = OLLAMA_BASE_URL):
        self.base_url = base_url
        self.model = model or _detect_ollama_model()
        self.use_ollama = self.model is not None

        # OpenAI SDK pointed at Ollama's /v1 endpoint.
        # api_key="ollama" — Ollama ignores the value but the SDK requires a non-empty string.
        self._client = OpenAI(
            base_url=f"{self.base_url}/v1",
            api_key="ollama",
        )

        if self.use_ollama:
            logger.info(f"OllamaAgent initialised | model={self.model} | endpoint={self.base_url}/v1")
        else:
            logger.warning("Ollama not available — using rule-based fallback agent")

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
                temperature=0.1,
                max_tokens=512,
            )
            content = response.choices[0].message.content
            return _parse_action(content, obs)

        except Exception as e:
            logger.error(f"OpenAI client error: {e} — using fallback")
            return _rule_based_fallback(obs)
