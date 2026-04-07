"""
CreditLens — Inference Script (OpenEnv compliant)
Runs the Ollama LLM agent against all three tasks and produces
structured [START]/[STEP]/[END] stdout logs exactly as required.

Usage:
    python inference.py [--task easy|medium|hard|all] [--seed 42]

Environment variables:
    API_BASE_URL   The CreditLens API endpoint (default: http://localhost:8000)
    MODEL_NAME     Ollama model name (default: auto-detect)
    HF_TOKEN       Hugging Face token (not required for local runs)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional

import httpx
from dotenv import load_dotenv
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from creditlens.inference.agent import OllamaAgent
from creditlens.models import LoanObservation, UnderwritingAction

load_dotenv()

# ─────────────────────────────────────────────
# Config from environment
# ─────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
MODEL_NAME = os.getenv("MODEL_NAME", None)  # None = auto-detect from Ollama
HF_TOKEN = os.getenv("HF_TOKEN", "")

console = Console()

# ─────────────────────────────────────────────
# Structured Logging — Required Format
# ─────────────────────────────────────────────

def log_start(task_id: str, episode_id: str, num_applicants: int):
    """Emit [START] log entry."""
    entry = {
        "type": "START",
        "task_id": task_id,
        "episode_id": episode_id,
        "num_applicants": num_applicants,
        "model": MODEL_NAME or "ollama-auto",
        "api_base_url": API_BASE_URL,
        "timestamp": time.time(),
    }
    print(f"[START] {json.dumps(entry)}", flush=True)


def log_step(
    step: int,
    applicant_id: str,
    action: UnderwritingAction,
    reward: float,
    done: bool,
    info: dict,
):
    """Emit [STEP] log entry."""
    entry = {
        "type": "STEP",
        "step": step,
        "applicant_id": applicant_id,
        "action_type": action.action_type if isinstance(action.action_type, str) else action.action_type.value,
        "action_params": action.params,
        "reasoning": action.reasoning,
        "reward": round(reward, 4),
        "done": done,
        "portfolio_ecl": info.get("portfolio_ecl", 0),
        "macro_shock": info.get("macro_shock_active", False),
        "fraud_caught": info.get("fraud_caught", 0),
        "timestamp": time.time(),
    }
    print(f"[STEP] {json.dumps(entry)}", flush=True)


def log_end(task_id: str, episode_id: str, scores: dict, total_reward: float, duration: float):
    """Emit [END] log entry."""
    entry = {
        "type": "END",
        "task_id": task_id,
        "episode_id": episode_id,
        "final_score": scores.get("score", 0.0),
        "score_details": scores,
        "total_reward": round(total_reward, 4),
        "duration_seconds": round(duration, 2),
        "timestamp": time.time(),
    }
    print(f"[END] {json.dumps(entry)}", flush=True)


# ─────────────────────────────────────────────
# API Client
# ─────────────────────────────────────────────

class CreditLensClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(timeout=30.0)

    def health(self) -> bool:
        try:
            resp = self._client.get(f"{self.base_url}/health")
            return resp.status_code == 200
        except Exception:
            return False

    def reset(self, task_id: str, seed: Optional[int] = None) -> dict:
        resp = self._client.post(
            f"{self.base_url}/reset",
            json={"task_id": task_id, "seed": seed},
        )
        resp.raise_for_status()
        return resp.json()

    def step(self, task_id: str, action: UnderwritingAction) -> dict:
        resp = self._client.post(
            f"{self.base_url}/step",
            json={
                "task_id": task_id,
                "action": action.model_dump(),
            },
        )
        resp.raise_for_status()
        return resp.json()

    def grade(self, task_id: str) -> dict:
        resp = self._client.get(f"{self.base_url}/grade", params={"task_id": task_id})
        resp.raise_for_status()
        return resp.json()


# ─────────────────────────────────────────────
# Episode Runner
# ─────────────────────────────────────────────

def run_episode(
    client: CreditLensClient,
    agent: OllamaAgent,
    task_id: str,
    seed: Optional[int] = None,
) -> dict:
    """Run one complete episode and return grade scores."""

    # --- Reset ---
    reset_data = client.reset(task_id=task_id, seed=seed)
    obs_data = reset_data["observation"]
    episode_id = reset_data["episode_id"]
    num_applicants = reset_data["num_applicants"]

    log_start(task_id, episode_id, num_applicants)

    console.print(Panel(
        f"[bold cyan]Episode {episode_id}[/bold cyan] | Task: [yellow]{task_id}[/yellow] | "
        f"Applicants: {num_applicants} | Model: {agent.model or 'rule-based'}",
        title="CreditLens Episode Start",
    ))

    start_time = time.time()
    step_count = 0
    total_reward = 0.0

    while True:
        obs = LoanObservation(**obs_data)

        # --- Agent decision ---
        action = agent.decide(obs)

        # --- Step ---
        step_data = client.step(task_id=task_id, action=action)
        reward = step_data["reward"]
        done = step_data["done"]
        info = step_data.get("info", {})

        step_count += 1
        total_reward += reward

        log_step(
            step=step_count,
            applicant_id=obs.applicant_id,
            action=action,
            reward=reward,
            done=done,
            info=info,
        )

        # Rich console summary
        action_str = action.action_type if isinstance(action.action_type, str) else action.action_type.value
        color = {"APPROVE": "green", "REJECT": "red", "COUNTER": "yellow", "REQUEST_INFO": "blue"}.get(action_str, "white")
        console.print(
            f"  Step {step_count:02d} | App {obs.applicant_id} | "
            f"[{color}]{action_str}[/{color}] | "
            f"FICO={obs.fico_score} | XGB={obs.xgb_default_prob:.0%} | "
            f"reward=[bold]{reward:+.3f}[/bold]"
        )

        if done:
            break

        next_obs = step_data.get("observation")
        if next_obs is None:
            break
        obs_data = next_obs

    # --- Grade ---
    grade_data = client.grade(task_id=task_id)
    scores = grade_data["scores"]
    duration = time.time() - start_time

    log_end(
        task_id=task_id,
        episode_id=episode_id,
        scores=scores,
        total_reward=total_reward,
        duration=duration,
    )

    console.print(Panel(
        f"Final Score: [bold green]{scores.get('score', 0):.4f}[/bold green] | "
        f"Total Reward: {total_reward:.3f} | Steps: {step_count} | "
        f"Duration: {duration:.1f}s",
        title=f"Episode Complete — {task_id.upper()}",
        style="green",
    ))

    return scores


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CreditLens Inference Script")
    parser.add_argument("--task", default="all", choices=["easy", "medium", "hard", "all"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", type=str, default=MODEL_NAME)
    args = parser.parse_args()

    # Verify API is reachable
    client = CreditLensClient(API_BASE_URL)
    if not client.health():
        console.print("[red]ERROR: CreditLens API is not reachable at " + API_BASE_URL)
        console.print("Start the server first: uvicorn creditlens.inference.service:app --port 8000")
        sys.exit(1)

    # Initialise agent
    agent = OllamaAgent(model=args.model)

    tasks = ["easy", "medium", "hard"] if args.task == "all" else [args.task]

    all_results: Dict[str, dict] = {}

    for task_id in tasks:
        console.rule(f"[bold] Task: {task_id.upper()} ")
        scores = run_episode(client, agent, task_id=task_id, seed=args.seed)
        all_results[task_id] = scores

    # Summary table
    console.rule("[bold]FINAL RESULTS")
    table = Table(title="CreditLens Baseline Scores", show_header=True)
    table.add_column("Task", style="cyan")
    table.add_column("Score", style="bold green")
    table.add_column("Details")

    for task_id, scores in all_results.items():
        final = scores.get("score", 0.0)
        detail_str = " | ".join(f"{k}={v:.3f}" for k, v in scores.items() if k != "score")
        table.add_row(task_id, f"{final:.4f}", detail_str)

    console.print(table)

    # Overall score
    overall = sum(s.get("score", 0) for s in all_results.values()) / max(len(all_results), 1)
    console.print(f"\n[bold]Overall Score: {overall:.4f}[/bold]")

    return all_results


if __name__ == "__main__":
    main()
