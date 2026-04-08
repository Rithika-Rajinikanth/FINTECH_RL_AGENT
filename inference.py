"""
CreditLens — Inference Script (OpenEnv compliant)
Produces EXACTLY the [START]/[STEP]/[END] stdout format required by the validator.

Usage:
    python inference.py

Environment variables:
    API_BASE_URL   CreditLens API endpoint (default: http://localhost:7860)
    MODEL_NAME     Model identifier (default: Qwen/Qwen2.5-72B-Instruct)
    HF_TOKEN       HuggingFace / API key
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import List, Optional

# ── Path setup ────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

import httpx
from openai import OpenAI

# ── Config — all from environment variables ───────────────────────────────
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "dummy-key")
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:7860")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://router.huggingface.co/v1")

BENCHMARK    = "creditlens"
MAX_STEPS    = 25   # safety cap per task episode
SUCCESS_THRESHOLD = 0.5


# ══════════════════════════════════════════════════════════════════════════
# MANDATORY STDOUT FORMAT
# [START] task=<name> env=<benchmark> model=<model>
# [STEP]  step=<n> action=<str> reward=<0.00> done=<true|false> error=<msg|null>
# [END]   success=<true|false> steps=<n> score=<0.00> rewards=<r1,r2,...>
# ══════════════════════════════════════════════════════════════════════════

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val  = str(done).lower()
    # Sanitise action string — no spaces allowed in the token
    action_clean = str(action).replace(" ", "_").replace("\n", "")[:80]
    print(
        f"[STEP] step={step} action={action_clean} reward={reward:.2f} "
        f"done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


# ══════════════════════════════════════════════════════════════════════════
# CreditLens REST Client
# ══════════════════════════════════════════════════════════════════════════

class CreditLensClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self._http = httpx.Client(timeout=30.0)

    def health(self) -> bool:
        try:
            r = self._http.get(f"{self.base_url}/health")
            return r.status_code == 200
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
        r.raise_for_status()
        return r.json()

    def grade(self, task_id: str) -> dict:
        r = self._http.get(f"{self.base_url}/grade", params={"task_id": task_id})
        r.raise_for_status()
        return r.json()


# ══════════════════════════════════════════════════════════════════════════
# Rule-based agent (no external LLM needed for baseline)
# Uses OpenAI client as required by the contest spec — pointed at HF router
# Falls back to deterministic rules if LLM call fails
# ══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are a senior AI loan underwriter. Decide on each applicant.
Respond ONLY with a JSON object — no other text:
{"action_type": "APPROVE"|"REJECT"|"COUNTER"|"REQUEST_INFO",
 "params": {"amount_fraction": 0.9} or {"reason_code": "HIGH_DTI"} etc,
 "reasoning": "one sentence"}

Rules (apply in order):
1. fraud_ring_score > 0.7 → REJECT FRAUD_SUSPECTED
2. xgb_default_prob > 0.62 → REJECT HIGH_DTI
3. fico_score < 580 → REJECT LOW_CREDIT_SCORE
4. dti_ratio > 0.5 → REJECT HIGH_DTI
5. xgb_default_prob > 0.4 → COUNTER at 0.75 fraction
6. Otherwise → APPROVE at 1.0"""


def _rule_based_action(obs_data: dict) -> dict:
    """Deterministic fallback — always produces a valid action."""
    fico  = obs_data.get("fico_score", 650)
    xgb   = obs_data.get("xgb_default_prob", 0.3)
    dti   = obs_data.get("dti_ratio", 0.35)
    fraud = obs_data.get("fraud_ring_score", 0.0)
    applicant_id = obs_data.get("applicant_id", "unknown")
    shared_phone = obs_data.get("shared_phone", False)
    shared_employer = obs_data.get("shared_employer_id", False)
    macro_shock = obs_data.get("macro_shock_active", False)

    shock_adj = 0.15 if macro_shock else 0.0
    xgb_cut = 0.62 - shock_adj

    if fraud > 0.70 or (shared_phone and shared_employer):
        return {"action_type": "REJECT",
                "applicant_id": applicant_id,
                "params": {"reason_code": "FRAUD_SUSPECTED"},
                "reasoning": f"Fraud score {fraud:.0%}"}

    if xgb > xgb_cut:
        return {"action_type": "REJECT",
                "applicant_id": applicant_id,
                "params": {"reason_code": "HIGH_DTI"},
                "reasoning": f"XGB {xgb:.0%} above cutoff"}

    if fico < 580:
        return {"action_type": "REJECT",
                "applicant_id": applicant_id,
                "params": {"reason_code": "LOW_CREDIT_SCORE"},
                "reasoning": "FICO below minimum"}

    if dti > 0.50:
        return {"action_type": "REJECT",
                "applicant_id": applicant_id,
                "params": {"reason_code": "HIGH_DTI"},
                "reasoning": "DTI too high"}

    if xgb > 0.40 or fico < 620 or dti > 0.40:
        frac = 0.70 if xgb > 0.50 else 0.82
        return {"action_type": "COUNTER",
                "applicant_id": applicant_id,
                "params": {"revised_amount_fraction": frac, "revised_rate_delta": 1.5},
                "reasoning": "Moderate risk — counter offer"}

    return {"action_type": "APPROVE",
            "applicant_id": applicant_id,
            "params": {"amount_fraction": 1.0},
            "reasoning": "Creditworthy"}


def _llm_action(client: OpenAI, obs_data: dict) -> dict:
    """Try LLM decision; fall back to rules on any error."""
    import json, re
    try:
        obs_summary = (
            f"applicant_id={obs_data.get('applicant_id')} "
            f"fico={obs_data.get('fico_score')} "
            f"xgb_default_prob={obs_data.get('xgb_default_prob', 0):.2f} "
            f"dti={obs_data.get('dti_ratio', 0):.2f} "
            f"fraud={obs_data.get('fraud_ring_score', 0):.2f} "
            f"income={obs_data.get('income', 0):,.0f} "
            f"macro_shock={obs_data.get('macro_shock_active', False)}"
        )
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": obs_summary},
            ],
            temperature=0.1,
            max_tokens=200,
        )
        text = (completion.choices[0].message.content or "").strip()
        # Strip markdown fences if present
        text = re.sub(r"```(?:json)?|```", "", text).strip()
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            data = json.loads(match.group())
            data["applicant_id"] = obs_data.get("applicant_id", "unknown")
            return data
    except Exception as e:
        pass
    return _rule_based_action(obs_data)


# ══════════════════════════════════════════════════════════════════════════
# Episode runner
# ══════════════════════════════════════════════════════════════════════════

def run_task(
    task_id: str,
    client_env: CreditLensClient,
    client_llm: OpenAI,
    seed: int = 42,
) -> float:
    """Run one complete task episode. Returns final score in [0, 1]."""

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    error_msg = None

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        # ── Reset ──────────────────────────────────────────────────────────
        reset_data = client_env.reset(task_id=task_id, seed=seed)
        obs_data   = reset_data["observation"]
        done       = False

        # ── Episode loop ───────────────────────────────────────────────────
        for step_n in range(1, MAX_STEPS + 1):
            if done:
                break

            # Decide
            action = _llm_action(client_llm, obs_data)
            action_str = f"{action.get('action_type','?')}({action.get('params',{})})"

            # Step
            try:
                step_data = client_env.step(task_id=task_id, action=action)
                reward    = float(step_data.get("reward", 0.0))
                done      = bool(step_data.get("done", False))
                error_msg = step_data.get("error", None)
                next_obs  = step_data.get("observation")
            except Exception as e:
                reward    = 0.0
                done      = True
                error_msg = str(e)
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
            grade_data = client_env.grade(task_id=task_id)
            score = float(grade_data.get("final_score", 0.0))
        except Exception:
            # Fallback: normalise cumulative reward
            total = sum(rewards)
            score = min(max(total / max(steps_taken, 1) / 2.0, 0.0), 1.0)

        score   = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_THRESHOLD

    except Exception as e:
        error_msg = str(e)
        score = 0.0
        success = False

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════

def main() -> None:
    # ── Wait for server ────────────────────────────────────────────────────
    env_client = CreditLensClient(API_BASE_URL)
    print(f"[INFO] Waiting for CreditLens API at {API_BASE_URL} ...", flush=True)
    for attempt in range(30):
        if env_client.health():
            print(f"[INFO] API is ready (attempt {attempt + 1})", flush=True)
            break
        time.sleep(2)
    else:
        print("[ERROR] API did not become ready in time. Exiting.", flush=True)
        sys.exit(1)

    # ── LLM client (OpenAI SDK — required by contest) ─────────────────────
    llm_client = OpenAI(base_url=LLM_BASE_URL, api_key=API_KEY)

    # ── Run all three tasks ────────────────────────────────────────────────
    tasks   = ["easy", "medium", "hard"]
    scores  = {}

    for task_id in tasks:
        score = run_task(
            task_id=task_id,
            client_env=env_client,
            client_llm=llm_client,
            seed=42,
        )
        scores[task_id] = score

    # ── Summary ────────────────────────────────────────────────────────────
    overall = sum(scores.values()) / len(scores)
    print(f"\n[SUMMARY] easy={scores['easy']:.3f} medium={scores['medium']:.3f} "
          f"hard={scores['hard']:.3f} overall={overall:.3f}", flush=True)


if __name__ == "__main__":
    main()