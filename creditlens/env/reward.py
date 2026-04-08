"""
CreditLens — Reward Engine
Multi-objective reward function with competing signals.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from creditlens.models import (
    ActionType,
    ApplicantRecord,
    EpisodeState,
    RewardBreakdown,
    TaskConfig,
    UnderwritingAction,
)


class RewardEngine:
    """
    Computes per-step reward with the following components:
    1. Base reward: correctness of the underwriting decision
    2. ECL penalty: portfolio credit loss management
    3. Fairness penalty: demographic parity (ECOA compliance)
    4. Fraud catch bonus: catching fraudulent applicants
    5. Step cost: small fixed cost per step (encourages efficiency)
    6. Info request cost: penalises unnecessary info requests
    7. Macro adaptation bonus: reward for adapting to rate shocks correctly
    """

    # Reward hyperparameters
    CORRECT_APPROVE_REWARD = 0.30
    CORRECT_REJECT_REWARD = 0.25
    WRONG_APPROVE_PENALTY = -0.40  # approving a defaulter
    WRONG_REJECT_PENALTY = -0.20   # rejecting a good borrower
    FRAUD_CATCH_BONUS = 0.50
    FRAUD_MISS_PENALTY = -0.60
    STEP_COST = 0.01
    INFO_REQUEST_COST = 0.05
    COUNTER_CORRECT_REWARD = 0.15  # counter is a partial approval, good for borderline cases
    ECL_PENALTY_SCALE = 2.0
    FAIRNESS_PENALTY_SCALE = 0.30
    MACRO_ADAPT_BONUS = 0.10

    def __init__(self, config: TaskConfig):
        self.config = config

    def compute(
        self,
        action: UnderwritingAction,
        record: ApplicantRecord,
        applicant_row: pd.Series,
        state: EpisodeState,
    ) -> RewardBreakdown:

        action_type = ActionType(action.action_type) if isinstance(action.action_type, str) else action.action_type
        is_fraud = bool(applicant_row["is_fraud"])
        will_default = bool(applicant_row["will_default"])
        truly_creditworthy = not will_default and not is_fraud

        base_reward = 0.0
        ecl_penalty = 0.0
        fairness_penalty = 0.0
        fraud_catch = 0.0
        step_cost = self.STEP_COST
        info_cost = 0.0
        macro_bonus = 0.0

        # ── Base reward logic ──────────────────────────
        if action_type == ActionType.REQUEST_INFO:
            if record.info_requests >= 2:
                # Already asked twice — punish redundancy hard
                info_cost = self.INFO_REQUEST_COST * 3
            elif not truly_creditworthy and applicant_row["fraud_ring_score"] > 0.4:
                # Reasonable to ask for more info on suspicious applicant
                info_cost = self.INFO_REQUEST_COST * 0.5
            else:
                info_cost = self.INFO_REQUEST_COST
            return RewardBreakdown.compute(
                base=0.0, step_cost=step_cost, info_cost=info_cost
            )

        elif action_type == ActionType.APPROVE:
            if is_fraud:
                base_reward = self.FRAUD_MISS_PENALTY
                fraud_catch = 0.0
            elif will_default:
                base_reward = self.WRONG_APPROVE_PENALTY
                # ECL penalty proportional to loan size
                ecl_penalty = self._ecl_penalty(applicant_row, action, state)
            else:
                base_reward = self.CORRECT_APPROVE_REWARD

        elif action_type == ActionType.REJECT:
            if is_fraud:
                base_reward = self.CORRECT_REJECT_REWARD
                fraud_catch = self.FRAUD_CATCH_BONUS
            elif will_default:
                base_reward = self.CORRECT_REJECT_REWARD
            else:
                # Rejecting a good borrower — discrimination risk
                base_reward = self.WRONG_REJECT_PENALTY

        elif action_type == ActionType.COUNTER:
            # Counter-offer: good for borderline cases
            revised_frac = action.params.get("revised_amount_fraction", 0.7)
            if is_fraud:
                base_reward = self.FRAUD_MISS_PENALTY * 0.8
            elif will_default and revised_frac <= 0.5:
                # Good risk management on a borderline defaulter
                base_reward = self.COUNTER_CORRECT_REWARD * 0.5
                ecl_penalty = self._ecl_penalty(applicant_row, action, state) * 0.5
            elif truly_creditworthy:
                # Counter on a good applicant is suboptimal but acceptable
                base_reward = self.COUNTER_CORRECT_REWARD
            else:
                base_reward = self.WRONG_APPROVE_PENALTY * 0.6
                ecl_penalty = self._ecl_penalty(applicant_row, action, state) * 0.6

        # ── Fairness penalty ──────────────────────────
        fairness_penalty = self._fairness_penalty(action_type, applicant_row, state)

        # ── Macro adaptation bonus ──────────────────────
        if state.macro_shock_active and action_type in (ActionType.REJECT, ActionType.COUNTER):
            xgb_prob = float(applicant_row["xgb_default_prob"])
            # After rate shock, being more conservative is rewarded
            if xgb_prob > 0.45:
                macro_bonus = self.MACRO_ADAPT_BONUS

        return RewardBreakdown.compute(
            base=base_reward,
            ecl_penalty=ecl_penalty,
            fairness_penalty=fairness_penalty,
            fraud_catch=fraud_catch,
            step_cost=step_cost,
            info_cost=info_cost,
            macro_bonus=macro_bonus,
        )

    def _ecl_penalty(
        self,
        row: pd.Series,
        action: UnderwritingAction,
        state: EpisodeState,
    ) -> float:
        """Penalty when approving a defaulter — proportional to ECL."""
        pd_est = float(row["xgb_default_prob"])
        loan = float(row["loan_amount"])
        lgd = 0.45

        if action.action_type == ActionType.APPROVE:
            frac = action.params.get("amount_fraction", 1.0)
        else:
            frac = action.params.get("revised_amount_fraction", 0.6)

        ecl = pd_est * lgd * loan * frac / 200_000  # normalise
        # Extra penalty if we're over budget
        if state.portfolio_ecl > state.portfolio_ecl_budget:
            ecl *= 1.5

        return round(float(np.clip(ecl * self.ECL_PENALTY_SCALE, 0, 0.5)), 4)

    def _fairness_penalty(
        self,
        action_type: ActionType,
        row: pd.Series,
        state: EpisodeState,
    ) -> float:
        """
        Demographic parity penalty.
        Penalises significant approval rate gap between reference and protected groups.
        """
        if action_type not in (ActionType.APPROVE, ActionType.REJECT, ActionType.COUNTER):
            return 0.0

        ref_decisions = state.decisions_by_group.get("group_a", 0)
        ref_approvals = state.approvals_by_group.get("group_a", 0)

        protected_decisions = (
            state.decisions_by_group.get("group_b", 0)
            + state.decisions_by_group.get("group_c", 0)
        )
        protected_approvals = (
            state.approvals_by_group.get("group_b", 0)
            + state.approvals_by_group.get("group_c", 0)
        )

        # Not enough data yet for meaningful fairness calc
        if ref_decisions < 3 or protected_decisions < 3:
            return 0.0

        ref_rate = ref_approvals / ref_decisions
        prot_rate = protected_approvals / protected_decisions
        gap = ref_rate - prot_rate  # positive = reference group approved more

        if gap > self.config.fairness_threshold:
            penalty = (gap - self.config.fairness_threshold) * self.FAIRNESS_PENALTY_SCALE
            return round(float(np.clip(penalty, 0, 0.30)), 4)

        return 0.0
