"""
CreditLens — Core RL Environment Engine
Implements the full OpenEnv step()/reset()/state() interface.
Also wraps as a gymnasium.Env for SB3 PPO training.
"""

from __future__ import annotations

import random
import uuid
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd
from loguru import logger

from creditlens.models import (
    ActionType,
    ApplicantRecord,
    DemographicGroup,
    EpisodeState,
    LoanObservation,
    LoanPurpose,
    RejectReason,
    RequestField,
    RewardBreakdown,
    StepResult,
    TaskConfig,
    UnderwritingAction,
)
from creditlens.data.dataset import sample_applicants
from creditlens.env.reward import RewardEngine


# ─────────────────────────────────────────────
# Task configurations
# ─────────────────────────────────────────────

TASK_CONFIGS: Dict[str, TaskConfig] = {
    "easy": TaskConfig(
        task_id="easy",
        name="Loan Approval Queue",
        num_applicants=10,
        max_steps=20,
        ecl_budget=0.08,
        fraud_ring_size=0,
        macro_shock=False,
        fairness_threshold=0.20,
    ),
    "medium": TaskConfig(
        task_id="medium",
        name="Portfolio Rebalancing Under Rate Shift",
        num_applicants=15,
        max_steps=40,
        ecl_budget=0.05,
        fraud_ring_size=0,
        macro_shock=True,
        shock_step=8,
        shock_bps=75,
        fairness_threshold=0.15,
    ),
    "hard": TaskConfig(
        task_id="hard",
        name="Systemic Shock with Hidden Fraud Ring",
        num_applicants=20,
        max_steps=60,
        ecl_budget=0.04,
        fraud_ring_size=3,
        macro_shock=True,
        shock_step=10,
        shock_bps=100,
        fairness_threshold=0.12,
    ),
}


# ─────────────────────────────────────────────
# Main Environment
# ─────────────────────────────────────────────

class CreditLensEnv:
    """
    Core CreditLens environment.
    Implements: reset() → LoanObservation
                step(action) → StepResult
                state() → EpisodeState
    """

    def __init__(self, task_id: str = "easy", seed: Optional[int] = None):
        if task_id not in TASK_CONFIGS:
            raise ValueError(f"Unknown task_id '{task_id}'. Choose from: {list(TASK_CONFIGS)}")
        self.task_id = task_id
        self.config = TASK_CONFIGS[task_id]
        self.reward_engine = RewardEngine(self.config)
        self._seed = seed
        self._episode_state: Optional[EpisodeState] = None
        self._applicants_df: Optional[pd.DataFrame] = None

    def reset(self, seed: Optional[int] = None) -> LoanObservation:
        """Start a new episode. Returns the first observation."""
        effective_seed = seed if seed is not None else self._seed
        self._applicants_df = sample_applicants(
            n=self.config.num_applicants,
            fraud_count=self.config.fraud_ring_size,
            seed=effective_seed,
        )

        applicant_records = [
            ApplicantRecord(
                applicant_id=row["applicant_id"],
                ground_truth_default=bool(row["will_default"]),
                ground_truth_fraud=bool(row["is_fraud"]),
            )
            for _, row in self._applicants_df.iterrows()
        ]

        self._episode_state = EpisodeState(
            episode_id=str(uuid.uuid4())[:8],
            task_id=self.task_id,
            step=0,
            max_steps=self.config.max_steps,
            applicants=applicant_records,
            portfolio_ecl_budget=self.config.ecl_budget,
            macro_shock_step=self.config.shock_step,
        )

        logger.info(
            f"Episode {self._episode_state.episode_id} started | "
            f"Task={self.task_id} | Applicants={self.config.num_applicants}"
        )
        return self._build_observation()

    def step(self, action: UnderwritingAction) -> StepResult:
        """Process one agent action and return the next observation + reward."""
        if self._episode_state is None:
            raise RuntimeError("Call reset() before step()")

        state = self._episode_state

        # Advance step counter
        state.step += 1

        # Activate macro shock if scheduled
        if self.config.macro_shock and self.config.shock_step == state.step:
            state.macro_shock_active = True
            state.shock_magnitude_bps = self.config.shock_bps
            logger.warning(f"⚡ MACRO SHOCK: +{self.config.shock_bps}bps at step {state.step}")

        # Locate current applicant record
        idx = state.current_applicant_index
        if idx >= len(state.applicants):
            # No more applicants — end episode
            return self._terminate()

        record = state.applicants[idx]
        applicant_row = self._applicants_df.iloc[idx]

        # --- Compute reward for this action ---
        reward_breakdown = self.reward_engine.compute(
            action=action,
            record=record,
            applicant_row=applicant_row,
            state=state,
        )

        # --- Apply action effects ---
        record.action_taken = action.action_type
        record.action_params = action.params
        record.reward_earned = reward_breakdown.total

        if action.action_type == ActionType.REQUEST_INFO:
            record.info_requests += 1
            # REQUEST_INFO does NOT advance to next applicant (costs a step)
        else:
            # Update portfolio ECL
            ecl_delta = self._compute_ecl_delta(action, applicant_row)
            state.portfolio_ecl += ecl_delta

            # Update fairness tracking
            group = applicant_row["demographic_group"]
            state.decisions_by_group[group] = state.decisions_by_group.get(group, 0) + 1
            if action.action_type in (ActionType.APPROVE, ActionType.COUNTER):
                state.approvals_by_group[group] = state.approvals_by_group.get(group, 0) + 1

            # Fraud tracking
            if applicant_row["is_fraud"]:
                if action.action_type == ActionType.REJECT:
                    state.fraud_caught += 1
                else:
                    state.fraud_missed += 1
            else:
                if action.action_type == ActionType.REJECT and applicant_row["is_fraud"] == False:
                    # Check if this looked like a fraud flag (high fraud_ring_score)
                    if applicant_row["fraud_ring_score"] > 0.5:
                        state.false_fraud_flags += 1

            state.current_applicant_index += 1

        state.total_reward += reward_breakdown.total

        # Determine if done
        done = (
            state.current_applicant_index >= len(state.applicants)
            or state.step >= state.max_steps
            or state.portfolio_ecl > state.portfolio_ecl_budget * 2
        )
        state.done = done

        if done:
            next_obs = None
        else:
            next_obs = self._build_observation()

        return StepResult(
            observation=next_obs,
            reward=reward_breakdown.total,
            reward_breakdown=reward_breakdown,
            done=done,
            info={
                "step": state.step,
                "applicant_id": record.applicant_id,
                "action": action.action_type,
                "portfolio_ecl": state.portfolio_ecl,
                "macro_shock_active": state.macro_shock_active,
                "fraud_caught": state.fraud_caught,
                "fraud_missed": state.fraud_missed,
            },
        )

    def state(self) -> EpisodeState:
        if self._episode_state is None:
            raise RuntimeError("Call reset() first")
        return self._episode_state

    def _build_observation(self) -> LoanObservation:
        state = self._episode_state
        idx = state.current_applicant_index
        row = self._applicants_df.iloc[idx]

        # Compute approval rates per group
        ref_decisions = state.decisions_by_group.get("group_a", 0)
        ref_approvals = state.approvals_by_group.get("group_a", 0)
        approval_rate_reference = ref_approvals / max(ref_decisions, 1)

        protected_decisions = (
            state.decisions_by_group.get("group_b", 0)
            + state.decisions_by_group.get("group_c", 0)
        )
        protected_approvals = (
            state.approvals_by_group.get("group_b", 0)
            + state.approvals_by_group.get("group_c", 0)
        )
        approval_rate_protected = protected_approvals / max(protected_decisions, 1)

        return LoanObservation(
            applicant_id=row["applicant_id"],
            episode_id=state.episode_id,
            step_number=state.step,
            fico_score=int(row["fico_score"]),
            income=float(row["income"]),
            loan_amount=float(row["loan_amount"]),
            loan_purpose=LoanPurpose(row["loan_purpose"]),
            employment_years=float(row["employment_years"]),
            dti_ratio=float(row["dti_ratio"]),
            credit_utilization=float(row["credit_utilization"]),
            payment_history_score=float(row["payment_history_score"]),
            num_open_accounts=int(row["num_open_accounts"]),
            num_derogatory_marks=int(row["num_derogatory_marks"]),
            ltv_ratio=float(row["ltv_ratio"]) if pd.notna(row.get("ltv_ratio")) else None,
            xgb_default_prob=float(row["xgb_default_prob"]),
            shap_top_feature=str(row["shap_top_feature"]),
            shap_top_value=float(row["shap_top_value"]),
            fraud_ring_score=float(row["fraud_ring_score"]),
            shared_phone=bool(row["shared_phone"]),
            shared_employer_id=bool(row["shared_employer_id"]),
            graph_cluster_size=int(row["graph_cluster_size"]),
            demographic_group=DemographicGroup(row["demographic_group"]),
            fed_funds_rate=float(row["fed_funds_rate"])
            + (state.shock_magnitude_bps / 100 if state.macro_shock_active else 0),
            treasury_yield_10y=float(row["treasury_yield_10y"]),
            unemployment_rate=float(row["unemployment_rate"]),
            macro_shock_active=state.macro_shock_active,
            shock_magnitude_bps=state.shock_magnitude_bps,
            portfolio_ecl=round(state.portfolio_ecl, 6),
            portfolio_ecl_budget=state.portfolio_ecl_budget,
            approval_rate_protected=round(approval_rate_protected, 4),
            approval_rate_reference=round(approval_rate_reference, 4),
            steps_remaining=state.max_steps - state.step,
        )

    def _compute_ecl_delta(self, action: UnderwritingAction, row: pd.Series) -> float:
        """
        Expected Credit Loss delta for this decision.
        ECL = PD × LGD × EAD (simplified)
        """
        if action.action_type == ActionType.REJECT:
            return 0.0

        pd_estimate = float(row["xgb_default_prob"])
        lgd = 0.45  # standard LGD for unsecured loans
        loan_amount = float(row["loan_amount"])

        if action.action_type == ActionType.APPROVE:
            amount_fraction = action.params.get("amount_fraction", 1.0)
        elif action.action_type == ActionType.COUNTER:
            amount_fraction = action.params.get("revised_amount_fraction", 0.6)
        else:
            amount_fraction = 0.0

        ead = loan_amount * amount_fraction
        ecl = pd_estimate * lgd * ead / 1_000_000  # normalise to [0,1] scale
        return round(ecl, 8)

    def _terminate(self) -> StepResult:
        self._episode_state.done = True
        return StepResult(
            observation=None,
            reward=0.0,
            reward_breakdown=RewardBreakdown.compute(base=0.0),
            done=True,
            info={"reason": "no_more_applicants"},
        )


# ─────────────────────────────────────────────
# Gymnasium Wrapper for SB3 PPO Training
# ─────────────────────────────────────────────

OBSERVATION_DIM = 20  # flattened numeric observation vector


class CreditLensGymEnv(gym.Env):
    """
    gymnasium.Env wrapper around CreditLensEnv for Stable-Baselines3 PPO.
    Observation: 20-dim float vector
    Action: Discrete(4) — APPROVE, REJECT, COUNTER, REQUEST_INFO
    """

    metadata = {"render_modes": []}

    def __init__(self, task_id: str = "easy"):
        super().__init__()
        self._env = CreditLensEnv(task_id=task_id)
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBSERVATION_DIM,), dtype=np.float32
        )
        self._current_obs: Optional[LoanObservation] = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs = self._env.reset(seed=seed)
        self._current_obs = obs
        return self._vectorise(obs), {}

    def step(self, action: int):
        action_type = [ActionType.APPROVE, ActionType.REJECT, ActionType.COUNTER, ActionType.REQUEST_INFO][action]
        params = self._default_params(action_type)
        ua = UnderwritingAction(
            action_type=action_type,
            applicant_id=self._current_obs.applicant_id if self._current_obs else "EP_000",
            params=params,
        )
        result = self._env.step(ua)
        self._current_obs = result.observation
        vec = self._vectorise(result.observation) if result.observation else np.zeros(OBSERVATION_DIM, dtype=np.float32)
        return vec, result.reward, result.done, False, result.info

    def _vectorise(self, obs: LoanObservation) -> np.ndarray:
        if obs is None:
            return np.zeros(OBSERVATION_DIM, dtype=np.float32)
        return np.array([
            obs.fico_score / 850,
            obs.income / 200_000,
            obs.loan_amount / 500_000,
            obs.employment_years / 30,
            obs.dti_ratio,
            obs.credit_utilization,
            obs.payment_history_score,
            obs.num_open_accounts / 20,
            obs.num_derogatory_marks / 5,
            obs.xgb_default_prob,
            obs.shap_top_value,
            obs.fraud_ring_score,
            float(obs.shared_phone),
            float(obs.shared_employer_id),
            obs.graph_cluster_size / 10,
            obs.fed_funds_rate / 10,
            obs.unemployment_rate / 15,
            float(obs.macro_shock_active),
            obs.portfolio_ecl / obs.portfolio_ecl_budget,
            obs.approval_rate_protected - obs.approval_rate_reference,
        ], dtype=np.float32)

    def _default_params(self, action_type: ActionType) -> dict:
        if action_type == ActionType.APPROVE:
            return {"amount_fraction": 1.0}
        elif action_type == ActionType.REJECT:
            return {"reason_code": RejectReason.HIGH_DTI.value}
        elif action_type == ActionType.COUNTER:
            return {"revised_amount_fraction": 0.7, "revised_rate_delta": 1.5}
        else:
            return {"field_name": RequestField.INCOME_PROOF.value}
