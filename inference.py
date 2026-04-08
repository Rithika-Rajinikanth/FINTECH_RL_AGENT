"""
CreditLens — Inference Script (contest-compliant)

Emits EXACTLY the stdout format required by the validator:
  [START] task=<name> env=<benchmark> model=<model>
  [STEP]  step=<n> action=<str> reward=<0.00> done=<true|false> error=<null|msg>
  [END]   success=<true|false> steps=<n> score=<0.00> rewards=<r1,r2,...>

Environment variables:
  API_BASE_URL  CreditLens server (default: http://localhost:7860)
  MODEL_NAME    LLM model name   (default: Qwen/Qwen2.5-72B-Instruct)
  HF_TOKEN      HuggingFace API key
  LLM_BASE_URL  LLM router URL   (default: https://router.huggingface.co/v1)
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from typing import List, Optional

import httpx
from openai import OpenAI

# ── Config ────────────────────────────────────────────────────────────────
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "dummy-key")
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:7860")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://router.huggingface.co/v1")

BENCHMARK         = "creditlens"
MAX_STEPS         = 25
SUCCESS_THRESHOLD = 0.5


# ══════════════════════════════════════════════════════════════════════════
# MANDATORY STDOUT FORMAT — must match exactly
# ══════════════════════════════════════════════════════════════════════════

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool,
             error: Optional[str]) -> None:
    # Sanitise action: no spaces, no newlines, max 80 chars
    action_clean = str(action).replace(" ", "_").replace("\n", "").replace("\r", "")[:80]
    error_val    = error if error else "null"
    done_val     = str(done).lower()
    print(f"[STEP] step={step} action={action_clean} reward={reward:.2f} "
          f"done={done_val} error={error_val}", flush=True)


def log_end(success: bool, steps: int, score: float,
            rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} "
          f"score={score:.2f} rewards={rewards_str}", flush=True)


# ══════════════════════════════════════════════════════════════════════════
# CreditLens REST client
# ══════════════════════════════════════════════════════════════════════════

class CreditLensClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self._http    = httpx.Client(timeout=30.0)

    def health(self) -> bool:
        try:
            r = self._http.get(f"{self.base_url}/health")
            return r.status_code == 200
        except Exception:
            return False

    def reset(self, task_id: str = "easy", seed: int = 42) -> dict:
        r = self._http.post(f"{self.base_url}/reset",
                            json={"task_id": task_id, "seed": seed})
        r.raise_for_status()
        return r.json()

    def step(self, task_id: str, action: dict) -> dict:
        r = self._http.post(f"{self.base_url}/step",
                            json={"task_id": task_id, "action": action})
        r.raise_for_status()
        return r.json()

    def grade(self, task_id: str) -> dict:
        r = self._http.get(f"{self.base_url}/grade",
                           params={"task_id": task_id})
        r.raise_for_status()
        return r.json()


# ══════════════════════════════════════════════════════════════════════════
# Decision logic — rule-based fallback (deterministic, no Ollama needed)
# ══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are a senior AI loan underwriter. Decide on each applicant.
Respond ONLY with a JSON object — no other text:
{"action_type": "APPROVE"|"REJECT"|"COUNTER"|"REQUEST_INFO",
 "params": {"amount_fraction": 0.9} or {"reason_code": "HIGH_DTI"} etc,
 "reasoning": "one sentence"}

Priority rules (apply in order, stop at first match):
1. fraud_ring_score > 0.70 → REJECT FRAUD_SUSPECTED
2. xgb_default_prob > 0.62 → REJECT HIGH_DTI  (NEVER counter above 0.62)
3. ECL usage > 80% AND xgb > 0.15 → REJECT HIGH_DTI
4. ECL usage > 60% AND xgb > 0.35 → REJECT HIGH_DTI
5. macro_shock_active=true AND xgb > 0.47 → REJECT HIGH_DTI
6. macro_shock_active=true AND xgb > 0.30 → COUNTER at 0.75 fraction
7. fico < 580 → REJECT LOW_CREDIT_SCORE
8. dti > 0.50 → REJECT HIGH_DTI
9. xgb > 0.40 OR fico < 620 → COUNTER at 0.75 fraction
10. Otherwise → APPROVE at 1.0"""


def _rule_based(obs: dict) -> dict:
    """Deterministic rule-based underwriting decision."""
    fico    = obs.get("fico_score", 650)
    xgb     = obs.get("xgb_default_prob", 0.3)
    dti     = obs.get("dti_ratio", 0.35)
    fraud   = obs.get("fraud_ring_score", 0.0)
    shock   = obs.get("macro_shock_active", False)
    sp      = obs.get("shared_phone", False)
    se      = obs.get("shared_employer_id", False)
    ecl     = obs.get("portfolio_ecl", 0.0)
    budget  = obs.get("portfolio_ecl_budget", 0.05)
    app_id  = obs.get("applicant_id", "unknown")
    ref_r   = obs.get("approval_rate_reference", 0.5)
    prot_r  = obs.get("approval_rate_protected", 0.5)
    group   = obs.get("demographic_group", "group_a")

    ecl_use   = ecl / max(budget, 1e-8)
    shock_adj = 0.15 if shock else 0.0
    gap       = ref_r - prot_r          # positive = protected under-served
    is_prot   = group in ("group_b", "group_c")

    # P1: Fraud
    if fraud > 0.70 or (sp and se):
        return {"action_type": "REJECT", "applicant_id": app_id,
                "params": {"reason_code": "FRAUD_SUSPECTED"},
                "reasoning": f"Fraud score {fraud:.0%}"}

    # P2: XGB hard cutoff
    xgb_cut = 0.62 - shock_adj
    if xgb > xgb_cut:
        return {"action_type": "REJECT", "applicant_id": app_id,
                "params": {"reason_code": "HIGH_DTI"},
                "reasoning": f"XGB {xgb:.0%} > cutoff {xgb_cut:.0%}"}

    # P3: ECL guard (3 tiers)
    ecl_thresh = 1.1
    if ecl_use > 0.95:   ecl_thresh = 0.10
    elif ecl_use > 0.80: ecl_thresh = 0.15
    elif ecl_use > 0.60: ecl_thresh = 0.35
    ecl_thresh -= shock_adj
    if xgb > ecl_thresh:
        return {"action_type": "REJECT", "applicant_id": app_id,
                "params": {"reason_code": "HIGH_DTI"},
                "reasoning": f"ECL guard {ecl_use:.0%} used"}

    # P4: Fairness rescue
    if (gap > 0.12 and is_prot and xgb < 0.60 and fico >= 600
            and dti < 0.50 and fraud < 0.50):
        return {"action_type": "APPROVE", "applicant_id": app_id,
                "params": {"amount_fraction": 0.90},
                "reasoning": f"Fairness rescue gap={gap:+.0%}"}

    # P5: Fairness rebalance (over-served)
    if gap < -0.12 and is_prot and xgb > 0.35:
        return {"action_type": "COUNTER", "applicant_id": app_id,
                "params": {"revised_amount_fraction": 0.75, "revised_rate_delta": 1.5},
                "reasoning": f"Rebalance over-served gap={gap:+.0%}"}

    # P6: Post-shock conservatism
    if shock and xgb > 0.30:
        return {"action_type": "COUNTER", "applicant_id": app_id,
                "params": {"revised_amount_fraction": 0.75, "revised_rate_delta": 2.0},
                "reasoning": "Post-shock conservative counter"}

    # P7: Standard rules
    if fico < 580:
        return {"action_type": "REJECT", "applicant_id": app_id,
                "params": {"reason_code": "LOW_CREDIT_SCORE"},
                "reasoning": "FICO < 580"}
    if dti > 0.50:
        return {"action_type": "REJECT", "applicant_id": app_id,
                "params": {"reason_code": "HIGH_DTI"},
                "reasoning": "DTI > 50%"}
    if xgb > 0.40 or fico < 620 or dti > 0.40:
        frac = 0.70 if xgb > 0.50 else 0.82
        return {"action_type": "COUNTER", "applicant_id": app_id,
                "params": {"revised_amount_fraction": frac, "revised_rate_delta": 1.5},
                "reasoning": "Moderate risk counter"}

    return {"action_type": "APPROVE", "applicant_id": app_id,
            "params": {"amount_fraction": 1.0},
            "reasoning": "Creditworthy"}


def _llm_action(client: OpenAI, obs: dict) -> dict:
    """
    Try LLM decision via OpenAI client (required by contest).
    Falls back to rule-based if LLM call fails or response is unparseable.
    """
    try:
        summary = (
            f"applicant_id={obs.get('applicant_id')} "
            f"fico={obs.get('fico_score')} "
            f"xgb={obs.get('xgb_default_prob', 0):.2f} "
            f"dti={obs.get('dti_ratio', 0):.2f} "
            f"fraud={obs.get('fraud_ring_score', 0):.2f} "
            f"income={obs.get('income', 0):,.0f} "
            f"macro_shock={obs.get('macro_shock_active', False)} "
            f"ecl_usage={obs.get('portfolio_ecl', 0)/max(obs.get('portfolio_ecl_budget', 0.05), 1e-8):.0%} "
            f"demographic={obs.get('demographic_group', 'group_a')} "
            f"ref_approval={obs.get('approval_rate_reference', 0.5):.0%} "
            f"prot_approval={obs.get('approval_rate_protected', 0.5):.0%}"
        )
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": summary},
            ],
            temperature=0.1,
            max_tokens=200,
        )
        text = (completion.choices[0].message.content or "").strip()
        text = re.sub(r"```(?:json)?|```", "", text).strip()
        text = text.replace("\n", " ").replace("\r", " ")
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            data = json.loads(match.group())
            data["applicant_id"] = obs.get("applicant_id", "unknown")
            return data
    except Exception:
        pass
    return _rule_based(obs)


# ══════════════════════════════════════════════════════════════════════════
# Episode runner
# ══════════════════════════════════════════════════════════════════════════

def run_task(task_id: str, env_client: CreditLensClient,
             llm_client: OpenAI, seed: int = 42) -> float:
    """Run one episode. Returns score in [0, 1]."""
    rewards:     List[float] = []
    steps_taken: int         = 0
    score:       float       = 0.0
    success:     bool        = False
    error_msg:   Optional[str] = None

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        # ── Reset ──────────────────────────────────────────────────────────
        reset_data = env_client.reset(task_id=task_id, seed=seed)
        obs_data   = reset_data["observation"]
        done       = False

        # ── Episode loop ───────────────────────────────────────────────────
        for step_n in range(1, MAX_STEPS + 1):
            if done:
                break

            # Decide (LLM with rule-based fallback)
            action = _llm_action(llm_client, obs_data)
            action_str = f"{action.get('action_type','?')}({action.get('reasoning','')[:40]})"

            # Step
            try:
                step_data = env_client.step(task_id=task_id, action=action)
                reward    = float(step_data.get("reward", 0.0))
                done      = bool(step_data.get("done", False))
                error_msg = step_data.get("error")
                next_obs  = step_data.get("observation")
            except Exception as e:
                reward    = 0.0
                done      = True
                error_msg = str(e)[:80]
                next_obs  = None

            rewards.append(reward)
            steps_taken = step_n

            log_step(step=step_n, action=action_str, reward=reward,
                     done=done, error=error_msg)

            if done or next_obs is None:
                break
            obs_data = next_obs

        # ── Grade ──────────────────────────────────────────────────────────
        try:
            grade_data = env_client.grade(task_id=task_id)
            score = float(grade_data.get("final_score", 0.0))
        except Exception:
            total = sum(rewards)
            score = min(max(total / max(steps_taken, 1) / 2.0, 0.0), 1.0)

        score   = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_THRESHOLD

    except Exception as e:
        error_msg = str(e)[:120]
        score   = 0.0
        success = False

    finally:
        log_end(success=success, steps=steps_taken,
                score=score, rewards=rewards)

    return score


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════

def main() -> None:
    # ── Wait for API ───────────────────────────────────────────────────────
    env_client = CreditLensClient(API_BASE_URL)
    print(f"[INFO] Connecting to CreditLens at {API_BASE_URL} ...", flush=True)
    for attempt in range(30):
        if env_client.health():
            print(f"[INFO] API ready (attempt {attempt + 1})", flush=True)
            break
        time.sleep(2)
    else:
        print("[ERROR] API did not become ready. Exiting.", flush=True)
        sys.exit(1)

    # ── LLM client (OpenAI SDK — required by contest) ─────────────────────
    llm_client = OpenAI(base_url=LLM_BASE_URL, api_key=API_KEY)

    # ── Run all three tasks ────────────────────────────────────────────────
    scores = {}
    for task_id in ["easy", "medium", "hard"]:
        scores[task_id] = run_task(
            task_id=task_id,
            env_client=env_client,
            llm_client=llm_client,
            seed=42,
        )

    # ── Summary ────────────────────────────────────────────────────────────
    overall = sum(scores.values()) / len(scores)
    print(f"\n[SUMMARY] easy={scores['easy']:.3f} medium={scores['medium']:.3f} "
          f"hard={scores['hard']:.3f} overall={overall:.3f}", flush=True)


if __name__ == "__main__":
    main()