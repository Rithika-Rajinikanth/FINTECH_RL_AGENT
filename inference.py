"""
CreditLens — Inference Script (Phase 2 compliant)

Stdout format (exact — validator parses this):
  [START] task=<name> env=<benchmark> model=<model>
  [STEP]  step=<n> action=<str> reward=<0.00> done=<true|false> error=<null|msg>
  [END]   success=<true|false> steps=<n> score=<0.00> rewards=<r1,r2,...>

Environment variables injected by the contest validator:
  API_BASE_URL   LiteLLM proxy URL for LLM calls  ← validator injects this
  API_KEY        Proxy API key                     ← validator injects this
  MODEL_NAME     LLM model identifier

Additional variable for the CreditLens environment server:
  ENV_URL        CreditLens FastAPI URL (default: http://localhost:7860)

KEY FIX: API_KEY must be read from os.getenv("API_KEY") first.
  The old code read HF_TOKEN as fallback, but the validator injects
  API_KEY specifically. Reading HF_TOKEN was causing 401 errors because
  the personal HF token is rejected by the validator's LiteLLM proxy.
  Now: API_KEY env var is used directly with no HF_TOKEN fallback.
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

# ── CRITICAL: validator injects these exact names ──────────────────────────
# Do NOT rename. Do NOT fall back to HF_TOKEN for the LLM client.
API_BASE_URL = os.environ["API_BASE_URL"]
API_KEY = os.environ["API_KEY"]
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

# ENV_URL is the CreditLens environment server (separate from the LLM proxy)
ENV_URL   = os.getenv("ENV_URL", "http://localhost:7860")

BENCHMARK         = "creditlens"
MAX_STEPS         = 25
SUCCESS_THRESHOLD = 0.5


# ══════════════════════════════════════════════════════════════════════════
# Mandatory stdout format — validator parses these exact strings
# ══════════════════════════════════════════════════════════════════════════

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float,
             done: bool, error: Optional[str]) -> None:
    a = str(action).replace(" ", "_").replace("\n", "").replace("\r", "")[:80]
    e = str(error)[:80] if error else "null"
    print(f"[STEP] step={step} action={a} reward={reward:.2f} "
          f"done={str(done).lower()} error={e}", flush=True)


def log_end(success: bool, steps: int, score: float,
            rewards: List[float]) -> None:
    r = ",".join(f"{x:.2f}" for x in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} "
          f"score={score:.2f} rewards={r}", flush=True)


# ══════════════════════════════════════════════════════════════════════════
# CreditLens REST client (connects to ENV_URL, NOT the LLM proxy)
# ══════════════════════════════════════════════════════════════════════════

class CreditLensClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self._http    = httpx.Client(timeout=30.0)

    def health(self) -> bool:
        try:
            return self._http.get(
                f"{self.base_url}/health", timeout=5.0
            ).status_code == 200
        except Exception:
            return False

    def reset(self, task_id: str = "easy", seed: int = 42) -> dict:
        r = self._http.post(
            f"{self.base_url}/reset",
            json={"task_id": task_id, "seed": seed},
        )
        r.raise_for_status()
        return r.json()

    def step(self, task_id: str, action: dict) -> dict:
        r = self._http.post(
            f"{self.base_url}/step",
            json={"task_id": task_id, "action": action},
        )
        try:
            r.raise_for_status()
            return r.json()
        except Exception as e:
            return {"reward": 0.0, "done": True,
                    "error": str(e)[:60], "observation": None}

    def grade(self, task_id: str) -> dict:
        r = self._http.get(
            f"{self.base_url}/grade", params={"task_id": task_id}
        )
        r.raise_for_status()
        return r.json()


# ══════════════════════════════════════════════════════════════════════════
# Rule-based fallback (used only when LLM is unavailable or unparseable)
# ══════════════════════════════════════════════════════════════════════════

def _rule_based(obs: dict) -> dict:
    fico   = obs.get("fico_score", 650)
    xgb    = obs.get("xgb_default_prob", 0.3)
    dti    = obs.get("dti_ratio", 0.35)
    fraud  = obs.get("fraud_ring_score", 0.0)
    shock  = obs.get("macro_shock_active", False)
    sp     = obs.get("shared_phone", False)
    se     = obs.get("shared_employer_id", False)
    ecl    = obs.get("portfolio_ecl", 0.0)
    budget = obs.get("portfolio_ecl_budget", 0.05)
    app_id = obs.get("applicant_id", "unknown")
    ref_r  = obs.get("approval_rate_reference", 0.5)
    prot_r = obs.get("approval_rate_protected", 0.5)
    group  = obs.get("demographic_group", "group_a")

    ecl_use   = ecl / max(budget, 1e-8)
    shock_adj = 0.15 if shock else 0.0
    gap       = ref_r - prot_r
    is_prot   = group in ("group_b", "group_c")

    def rej(reason: str) -> dict:
        return {"action_type": "REJECT", "applicant_id": app_id,
                "params": {"reason_code": reason}, "reasoning": reason}

    # P1 Fraud
    if fraud > 0.70 or (sp and se):
        return rej("FRAUD_SUSPECTED")

    # P2 XGB cutoff
    xgb_cut = 0.62 - shock_adj
    if xgb > xgb_cut:
        return rej("HIGH_DTI")

    # P3 ECL guard
    ecl_thresh = 1.1
    if ecl_use > 0.95:   ecl_thresh = 0.10
    elif ecl_use > 0.80: ecl_thresh = 0.15
    elif ecl_use > 0.60: ecl_thresh = 0.35
    ecl_thresh -= shock_adj
    if xgb > ecl_thresh:
        return rej("HIGH_DTI")

    # P4 Fairness rescue (protected under-served only)
    if (gap > 0.12 and is_prot and xgb < 0.60
            and fico >= 600 and dti < 0.50 and fraud < 0.50):
        return {"action_type": "APPROVE", "applicant_id": app_id,
                "params": {"amount_fraction": 0.90},
                "reasoning": f"fairness_rescue_gap={gap:+.0%}"}

    # P4b Fairness rebalance (protected over-served)
    if gap < -0.12 and is_prot and xgb > 0.35:
        return {"action_type": "COUNTER", "applicant_id": app_id,
                "params": {"revised_amount_fraction": 0.75, "revised_rate_delta": 1.5},
                "reasoning": f"fairness_rebalance_gap={gap:+.0%}"}

    # P5 Post-shock
    if shock and xgb > 0.30:
        return {"action_type": "COUNTER", "applicant_id": app_id,
                "params": {"revised_amount_fraction": 0.75, "revised_rate_delta": 2.0},
                "reasoning": "post_shock_counter"}

    # P6 Standard rules
    if fico < 580:
        return rej("LOW_CREDIT_SCORE")
    if dti > 0.50:
        return rej("HIGH_DTI")
    if xgb > 0.40 or fico < 620 or dti > 0.40:
        frac = 0.70 if xgb > 0.50 else 0.82
        return {"action_type": "COUNTER", "applicant_id": app_id,
                "params": {"revised_amount_fraction": frac, "revised_rate_delta": 1.5},
                "reasoning": "moderate_risk_counter"}

    return {"action_type": "APPROVE", "applicant_id": app_id,
            "params": {"amount_fraction": 1.0}, "reasoning": "creditworthy"}


# ══════════════════════════════════════════════════════════════════════════
# LLM decision via validator-injected proxy
# ══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are a senior AI loan underwriter. Decide on each applicant.
Respond ONLY with valid JSON — no other text, no markdown fences:
{"action_type":"APPROVE"|"REJECT"|"COUNTER"|"REQUEST_INFO","params":{...},"reasoning":"one sentence"}

Decision rules (apply in order, stop at first match):
1. fraud_ring_score > 0.70 OR (shared_phone=true AND shared_employer=true) → REJECT FRAUD_SUSPECTED
2. xgb_default_prob > 0.62 → REJECT HIGH_DTI  (never COUNTER — still creates ECL)
3. ECL_usage > 80% AND xgb > 0.15 → REJECT HIGH_DTI
4. ECL_usage > 60% AND xgb > 0.35 → REJECT HIGH_DTI
5. macro_shock=true AND xgb > 0.47 → REJECT HIGH_DTI
6. macro_shock=true AND xgb > 0.30 → COUNTER revised_amount_fraction=0.75 revised_rate_delta=2.0
7. fico < 580 → REJECT LOW_CREDIT_SCORE
8. dti > 0.50 → REJECT HIGH_DTI
9. xgb > 0.40 OR fico < 620 → COUNTER revised_amount_fraction=0.75 revised_rate_delta=1.5
10. Otherwise → APPROVE amount_fraction=1.0"""


def _llm_action(llm_client: OpenAI, obs: dict) -> dict:
    """
    Call the LLM via the validator-injected API_BASE_URL proxy.
    Always makes an HTTP call — the validator monitors traffic to API_BASE_URL.
    Rule-based is used only if the JSON response is completely unparseable.
    """
    ecl_use = obs.get("portfolio_ecl", 0) / max(
        obs.get("portfolio_ecl_budget", 0.05), 1e-8
    )
    user_msg = (
        f"applicant_id={obs.get('applicant_id')} "
        f"fico={obs.get('fico_score')} "
        f"xgb={obs.get('xgb_default_prob', 0):.2f} "
        f"dti={obs.get('dti_ratio', 0):.2f} "
        f"fraud={obs.get('fraud_ring_score', 0):.2f} "
        f"shared_phone={obs.get('shared_phone', False)} "
        f"shared_employer={obs.get('shared_employer_id', False)} "
        f"shock={obs.get('macro_shock_active', False)} "
        f"ecl_usage={ecl_use:.0%} "
        f"group={obs.get('demographic_group', 'group_a')} "
        f"ref_approval={obs.get('approval_rate_reference', 0.5):.0%} "
        f"prot_approval={obs.get('approval_rate_protected', 0.5):.0%} "
        f"income={obs.get('income', 0):,.0f}"
    )

    response_text = ""
    try:
        completion = llm_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
            temperature=0.1,
            max_tokens=200,
        )
        response_text = (completion.choices[0].message.content or "").strip()
        print(f"[LLM] model={MODEL_NAME} tokens={len(response_text)}", flush=True)
    except Exception as e:
        print(f"[LLM_ERROR] {type(e).__name__}: {str(e)[:120]}", flush=True)
        return _rule_based(obs)

    # Parse LLM JSON response
    try:
        text = re.sub(r"```(?:json)?|```", "", response_text).strip()
        text = text.replace("\n", " ").replace("\r", " ")
        text = re.sub(r",\s*([}\]])", r"\1", text)   # trailing commas
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            data = json.loads(match.group())
            data.setdefault("applicant_id", obs.get("applicant_id", "unknown"))
            if data.get("action_type") in ("APPROVE", "REJECT", "COUNTER", "REQUEST_INFO"):
                return data
    except Exception as e:
        print(f"[LLM_PARSE_ERROR] {e}", flush=True)

    print("[LLM_FALLBACK] unparseable response — using rule-based", flush=True)
    return _rule_based(obs)


# ══════════════════════════════════════════════════════════════════════════
# Episode runner
# ══════════════════════════════════════════════════════════════════════════

def run_task(
    task_id: str,
    env_client: CreditLensClient,
    llm_client: OpenAI,
    seed: int = 42,
) -> float:
    rewards:     List[float] = []
    steps_taken: int         = 0
    score:       float       = 0.0
    success:     bool        = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        reset_data = env_client.reset(task_id=task_id, seed=seed)
        obs_data   = reset_data.get("observation") or reset_data
        done       = False

        for step_n in range(1, MAX_STEPS + 1):
            if done:
                break

            # Always call LLM first (validator monitors API_BASE_URL traffic)
            action = _llm_action(llm_client, obs_data)

            action_str = (
                f"{action.get('action_type', '?')}"
                f"({str(action.get('reasoning', ''))[:40]})"
            )

            step_error: Optional[str] = None
            try:
                sd         = env_client.step(task_id=task_id, action=action)
                reward     = float(sd.get("reward", 0.0))
                done       = bool(sd.get("done", False))
                step_error = sd.get("error")
                next_obs   = sd.get("observation")
            except Exception as e:
                reward     = 0.0
                done       = True
                step_error = str(e)[:60]
                next_obs   = None

            rewards.append(reward)
            steps_taken = step_n

            log_step(step=step_n, action=action_str, reward=reward,
                     done=done, error=step_error)

            if done or next_obs is None:
                break
            obs_data = next_obs

        # Grade the completed episode
        try:
            gd    = env_client.grade(task_id=task_id)
            score = float(
                gd.get("final_score")
                or gd.get("scores", {}).get("score", 0.0)
                or 0.0
            )
        except Exception:
            total = sum(rewards)
            score = min(max(total / max(steps_taken * 0.5, 1), 0.0), 1.0)

        # Ensure strictly within (0, 1) — matches grader contract
        score   = round(max(0.01, min(0.99, float(score))), 4)
        success = score >= SUCCESS_THRESHOLD

    except Exception as e:
        print(f"[ERROR] run_task({task_id}): {e}", flush=True)
        score   = 0.01
        success = False

    finally:
        log_end(success=success, steps=steps_taken,
                score=score, rewards=rewards)

    return score


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════

def main() -> int:
    # Print config — helps debug validator environment
    print(f"[INFO] API_BASE_URL={API_BASE_URL}", flush=True)
    print(f"[INFO] ENV_URL={ENV_URL}", flush=True)
    print(f"[INFO] MODEL_NAME={MODEL_NAME}", flush=True)
    # NOTE: API_KEY presence logged without revealing the value
    print(f"[INFO] API_KEY={'set (' + str(len(API_KEY)) + ' chars)' if API_KEY else 'NOT SET — LLM calls will fail'}", flush=True)

    # Wait for environment server
    env_client = CreditLensClient(ENV_URL)
    print(f"[INFO] Waiting for CreditLens env at {ENV_URL} ...", flush=True)
    ready = False
    for attempt in range(30):
        if env_client.health():
            print(f"[INFO] Env ready (attempt {attempt + 1})", flush=True)
            ready = True
            break
        time.sleep(2)

    if not ready:
        print(f"[ERROR] Env not ready after 60s at {ENV_URL}", flush=True)
        for t in ["easy", "medium", "hard"]:
            log_start(task=t, env=BENCHMARK, model=MODEL_NAME)
            log_end(success=False, steps=0, score=0.01, rewards=[])
        return 1

    # Initialise LLM client pointing at API_BASE_URL (validator's proxy)
    # This is the key requirement: base_url=API_BASE_URL, api_key=API_KEY
    # Using a dummy key "none" when API_KEY is unset — the LLM call will
    # fail gracefully and fall back to rule-based decisions.
    effective_key = API_KEY if API_KEY else "none"
    print(f"[INFO] Initialising LLM client → {API_BASE_URL}", flush=True)
    llm_client = OpenAI(base_url=API_BASE_URL, api_key=effective_key)

    # Run all three tasks
    scores: dict = {}
    for task_id in ["easy", "medium", "hard"]:
        try:
            scores[task_id] = run_task(
                task_id=task_id,
                env_client=env_client,
                llm_client=llm_client,
                seed=42,
            )
        except Exception as e:
            print(f"[ERROR] outer({task_id}): {e}", flush=True)
            log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)
            log_end(success=False, steps=0, score=0.01, rewards=[])
            scores[task_id] = 0.01

    # Summary
    overall = sum(scores.values()) / max(len(scores), 1)
    print(
        f"\n[SUMMARY] easy={scores.get('easy', 0):.3f} "
        f"medium={scores.get('medium', 0):.3f} "
        f"hard={scores.get('hard', 0):.3f} "
        f"overall={overall:.3f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"[FATAL] {e}", flush=True)
        sys.exit(0)