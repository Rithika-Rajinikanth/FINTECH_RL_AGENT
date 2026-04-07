"""
CreditLens — FastAPI Inference Service
Implements the full OpenEnv HTTP API: POST /reset, POST /step, GET /state
Also exposes Prometheus metrics at /metrics.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from prometheus_client import Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel
from starlette.responses import PlainTextResponse

from creditlens.env.engine import CreditLensEnv, TASK_CONFIGS
from creditlens.models import LoanObservation, StepResult, UnderwritingAction, EpisodeState
from creditlens.tasks.graders import grade_episode

# ─────────────────────────────────────────────
# Prometheus Metrics
# ─────────────────────────────────────────────

EPISODE_COUNTER = Counter("creditlens_episodes_total", "Total episodes started", ["task_id"])
STEP_COUNTER = Counter("creditlens_steps_total", "Total steps processed", ["task_id"])
REWARD_GAUGE = Gauge("creditlens_episode_reward", "Reward of last completed episode", ["task_id"])
STEP_LATENCY = Histogram("creditlens_step_latency_seconds", "Step processing latency")
APPROVAL_RATE = Gauge("creditlens_approval_rate", "Approval rate in current episode", ["task_id"])
ECL_GAUGE = Gauge("creditlens_portfolio_ecl", "Current portfolio ECL", ["task_id"])

# ─────────────────────────────────────────────
# App Setup
# ─────────────────────────────────────────────

app = FastAPI(
    title="CreditLens OpenEnv",
    description="AI Credit Risk Underwriting Environment — OpenEnv Compliant",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global environment instance (one per server process)
_envs: Dict[str, CreditLensEnv] = {}


def _get_env(task_id: str) -> CreditLensEnv:
    if task_id not in TASK_CONFIGS:
        raise HTTPException(status_code=400, detail=f"Unknown task_id: {task_id}")
    if task_id not in _envs:
        _envs[task_id] = CreditLensEnv(task_id=task_id)
    return _envs[task_id]


# ─────────────────────────────────────────────
# Request / Response Models
# ─────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: str = "easy"
    seed: Optional[int] = None


class ResetResponse(BaseModel):
    observation: LoanObservation
    episode_id: str
    task_id: str
    num_applicants: int
    max_steps: int


class StepRequest(BaseModel):
    task_id: str = "easy"
    action: UnderwritingAction


class StateResponse(BaseModel):
    episode_state: EpisodeState
    task_id: str


class GradeResponse(BaseModel):
    task_id: str
    scores: Dict[str, float]
    final_score: float


# ─────────────────────────────────────────────
# API Endpoints
# ─────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "service": "creditlens"}


@app.get("/tasks")
def list_tasks():
    return {
        task_id: {
            "name": cfg.name,
            "num_applicants": cfg.num_applicants,
            "max_steps": cfg.max_steps,
            "fraud_ring_size": cfg.fraud_ring_size,
            "macro_shock": cfg.macro_shock,
        }
        for task_id, cfg in TASK_CONFIGS.items()
    }


@app.post("/reset", response_model=ResetResponse)
def reset(req: ResetRequest):
    """Start a new episode."""
    env = _get_env(req.task_id)
    config = TASK_CONFIGS[req.task_id]

    observation = env.reset(seed=req.seed)

    EPISODE_COUNTER.labels(task_id=req.task_id).inc()
    logger.info(f"[RESET] task={req.task_id} episode={env.state().episode_id}")

    return ResetResponse(
        observation=observation,
        episode_id=env.state().episode_id,
        task_id=req.task_id,
        num_applicants=config.num_applicants,
        max_steps=config.max_steps,
    )


@app.post("/step", response_model=StepResult)
def step(req: StepRequest):
    """Process one agent action."""
    env = _get_env(req.task_id)

    if env._episode_state is None:
        raise HTTPException(status_code=400, detail="Call /reset before /step")

    start = time.perf_counter()
    result = env.step(req.action)
    latency = time.perf_counter() - start

    STEP_COUNTER.labels(task_id=req.task_id).inc()
    STEP_LATENCY.observe(latency)

    state = env.state()
    ECL_GAUGE.labels(task_id=req.task_id).set(state.portfolio_ecl)

    if result.done:
        REWARD_GAUGE.labels(task_id=req.task_id).set(state.total_reward)

    logger.info(
        f"[STEP] task={req.task_id} step={state.step} "
        f"action={req.action.action_type} reward={result.reward:.3f} done={result.done}"
    )

    return result


@app.get("/state", response_model=StateResponse)
def get_state(task_id: str = "easy"):
    """Return current episode state."""
    env = _get_env(task_id)
    if env._episode_state is None:
        raise HTTPException(status_code=400, detail="No active episode. Call /reset first.")
    return StateResponse(episode_state=env.state(), task_id=task_id)


@app.get("/grade")
def grade(task_id: str = "easy") -> GradeResponse:
    """Grade the current (or last completed) episode."""
    env = _get_env(task_id)
    if env._episode_state is None:
        raise HTTPException(status_code=400, detail="No episode to grade.")

    config = TASK_CONFIGS[task_id]
    scores = grade_episode(env.state(), config)
    final_score = scores.get("score", 0.0)

    logger.info(f"[GRADE] task={task_id} score={final_score:.4f} details={scores}")
    return GradeResponse(task_id=task_id, scores=scores, final_score=final_score)


@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    """Prometheus metrics endpoint."""
    return generate_latest().decode("utf-8")


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("creditlens.inference.service:app", host="0.0.0.0", port=8000, reload=False)
