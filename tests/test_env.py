"""
CreditLens — Test Suite
Tests for environment correctness, reward engine, and graders.
"""

from __future__ import annotations

import pytest

from creditlens.env.engine import CreditLensEnv, TASK_CONFIGS
from creditlens.models import (
    ActionType,
    RejectReason,
    UnderwritingAction,
)


# ─────────────────────────────────────────────
# Environment Tests
# ─────────────────────────────────────────────

class TestCreditLensEnv:

    def test_reset_returns_observation(self):
        env = CreditLensEnv(task_id="easy", seed=42)
        obs = env.reset(seed=42)
        assert obs is not None
        assert obs.fico_score >= 300
        assert obs.fico_score <= 850
        assert obs.xgb_default_prob >= 0.0
        assert obs.xgb_default_prob <= 1.0

    def test_step_approve(self):
        env = CreditLensEnv(task_id="easy", seed=42)
        obs = env.reset(seed=42)
        action = UnderwritingAction(
            action_type=ActionType.APPROVE,
            applicant_id=obs.applicant_id,
            params={"amount_fraction": 1.0},
        )
        result = env.step(action)
        assert result is not None
        assert isinstance(result.reward, float)
        assert isinstance(result.done, bool)

    def test_step_reject(self):
        env = CreditLensEnv(task_id="easy", seed=42)
        obs = env.reset(seed=42)
        action = UnderwritingAction(
            action_type=ActionType.REJECT,
            applicant_id=obs.applicant_id,
            params={"reason_code": RejectReason.HIGH_DTI.value},
        )
        result = env.step(action)
        assert result is not None

    def test_step_request_info_costs_step(self):
        env = CreditLensEnv(task_id="easy", seed=42)
        obs = env.reset(seed=42)
        action = UnderwritingAction(
            action_type=ActionType.REQUEST_INFO,
            applicant_id=obs.applicant_id,
            params={"field_name": "income_proof"},
        )
        result = env.step(action)
        # REQUEST_INFO should not advance applicant index — same applicant next
        assert result.reward < 0  # step cost applied
        if result.observation:
            assert result.observation.applicant_id == obs.applicant_id

    def test_episode_terminates(self):
        env = CreditLensEnv(task_id="easy", seed=1)
        obs = env.reset(seed=1)
        done = False
        steps = 0
        while not done and steps < 100:
            action = UnderwritingAction(
                action_type=ActionType.APPROVE,
                applicant_id=obs.applicant_id,
                params={"amount_fraction": 1.0},
            )
            result = env.step(action)
            done = result.done
            if result.observation:
                obs = result.observation
            steps += 1
        assert done is True

    def test_state_returns_episode_state(self):
        env = CreditLensEnv(task_id="easy", seed=42)
        env.reset(seed=42)
        state = env.state()
        assert state.task_id == "easy"
        assert state.step >= 0

    def test_all_tasks_can_reset(self):
        for task_id in ["easy", "medium", "hard"]:
            env = CreditLensEnv(task_id=task_id, seed=0)
            obs = env.reset(seed=0)
            assert obs is not None

    def test_reward_breakdown_sums_correctly(self):
        env = CreditLensEnv(task_id="easy", seed=42)
        obs = env.reset(seed=42)
        action = UnderwritingAction(
            action_type=ActionType.APPROVE,
            applicant_id=obs.applicant_id,
            params={"amount_fraction": 1.0},
        )
        result = env.step(action)
        rb = result.reward_breakdown
        # Total should approximately equal sum of components
        computed = (
            rb.base_reward
            - rb.ecl_penalty
            - rb.fairness_penalty
            + rb.fraud_catch_bonus
            - rb.step_cost
            - rb.info_request_cost
            + rb.macro_adaptation_bonus
        )
        assert abs(rb.total - computed) < 1e-6


# ─────────────────────────────────────────────
# Grader Tests
# ─────────────────────────────────────────────

class TestGraders:

    def _run_full_episode(self, task_id: str, action_type: ActionType) -> dict:
        from creditlens.tasks.graders import grade_episode
        env = CreditLensEnv(task_id=task_id, seed=42)
        obs = env.reset(seed=42)
        done = False
        while not done:
            action = UnderwritingAction(
                action_type=action_type,
                applicant_id=obs.applicant_id,
                params=self._default_params(action_type),
            )
            result = env.step(action)
            done = result.done
            if result.observation:
                obs = result.observation
        config = TASK_CONFIGS[task_id]
        return grade_episode(env.state(), config)

    def _default_params(self, action_type: ActionType) -> dict:
        if action_type == ActionType.APPROVE:
            return {"amount_fraction": 1.0}
        elif action_type == ActionType.REJECT:
            return {"reason_code": RejectReason.HIGH_DTI.value}
        elif action_type == ActionType.COUNTER:
            return {"revised_amount_fraction": 0.7, "revised_rate_delta": 1.5}
        return {"field_name": "income_proof"}

    def test_easy_grader_returns_score_in_range(self):
        scores = self._run_full_episode("easy", ActionType.APPROVE)
        assert 0.0 <= scores["score"] <= 1.0

    def test_medium_grader_returns_score_in_range(self):
        scores = self._run_full_episode("medium", ActionType.REJECT)
        assert 0.0 <= scores["score"] <= 1.0

    def test_hard_grader_returns_score_in_range(self):
        scores = self._run_full_episode("hard", ActionType.REJECT)
        assert 0.0 <= scores["score"] <= 1.0

    def test_grader_returns_required_keys(self):
        scores = self._run_full_episode("easy", ActionType.APPROVE)
        assert "score" in scores
        assert "f1" in scores


# ─────────────────────────────────────────────
# Gym Wrapper Tests
# ─────────────────────────────────────────────

class TestGymWrapper:

    def test_gym_reset(self):
        from creditlens.env.engine import CreditLensGymEnv
        env = CreditLensGymEnv(task_id="easy")
        obs, info = env.reset(seed=42)
        assert obs.shape == (20,)

    def test_gym_step(self):
        from creditlens.env.engine import CreditLensGymEnv
        env = CreditLensGymEnv(task_id="easy")
        obs, _ = env.reset(seed=42)
        obs2, reward, done, truncated, info = env.step(0)  # APPROVE
        assert obs2.shape == (20,)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
