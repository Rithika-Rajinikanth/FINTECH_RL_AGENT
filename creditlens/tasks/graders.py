"""
CreditLens — Task Graders
Each grader scores an episode from 0.0 to 1.0 with partial credit.
Graders are deterministic and reproducible.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np

from creditlens.models import (
    ActionType,
    ApplicantRecord,
    EpisodeState,
    TaskConfig,
)


# ─────────────────────────────────────────────
# Base Grader
# ─────────────────────────────────────────────

class BaseGrader:
    def score(self, state: EpisodeState, config: TaskConfig) -> Dict[str, float]:
        raise NotImplementedError


# ─────────────────────────────────────────────
# Easy Grader — F1 score against ground truth
# ─────────────────────────────────────────────

class EasyGrader(BaseGrader):
    """
    Task 1: Loan Approval Queue
    Score = F1 score of APPROVE/REJECT decisions vs ground truth labels.
    Score 1.0 if F1 ≥ 0.8, linear below.
    """

    def score(self, state: EpisodeState, config: TaskConfig) -> Dict[str, float]:
        tp = fp = fn = tn = 0

        for rec in state.applicants:
            if rec.action_taken is None:
                continue  # untouched — penalised via missing decisions

            action = ActionType(rec.action_taken) if isinstance(rec.action_taken, str) else rec.action_taken
            approved = action in (ActionType.APPROVE, ActionType.COUNTER)
            should_approve = not rec.ground_truth_default and not rec.ground_truth_fraud

            if approved and should_approve:
                tp += 1
            elif approved and not should_approve:
                fp += 1
            elif not approved and should_approve:
                fn += 1
            else:
                tn += 1

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)

        # Penalise for applicants not processed
        processed = sum(1 for r in state.applicants if r.action_taken is not None)
        completion_rate = processed / max(len(state.applicants), 1)

        raw_score = f1 * completion_rate
        # Normalised: 1.0 if F1 ≥ 0.8, linear below
        normalised = min(1.0, raw_score / 0.8)

        return {
            "score": round(normalised, 4),
            "f1": round(f1, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "completion_rate": round(completion_rate, 4),
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "true_negatives": tn,
        }


# ─────────────────────────────────────────────
# Medium Grader — Sharpe-style ECL management
# ─────────────────────────────────────────────

class MediumGrader(BaseGrader):
    """
    Task 2: Portfolio Rebalancing Under Rate Shift
    Score = Sharpe-like: (target_ECL - actual_ECL) / ECL_stddev
    Partial credit for getting direction right.
    """

    def score(self, state: EpisodeState, config: TaskConfig) -> Dict[str, float]:
        target_ecl = config.ecl_budget
        actual_ecl = state.portfolio_ecl

        # Collect per-decision ECL deltas for std dev
        ecl_deltas = []
        for rec in state.applicants:
            if rec.action_taken is not None:
                ecl_deltas.append(rec.reward_earned)

        ecl_stddev = float(np.std(ecl_deltas)) if ecl_deltas else 0.01

        ecl_gap = target_ecl - actual_ecl
        sharpe = ecl_gap / max(ecl_stddev, 0.01)

        # Normalise to [0,1]
        # Perfect score: actual_ecl ≤ target_ecl (sharpe ≥ 0)
        # Partial credit: sharpe in (-2, 0) gets partial score
        if sharpe >= 1.0:
            ecl_score = 1.0
        elif sharpe >= 0:
            ecl_score = 0.5 + sharpe * 0.5
        elif sharpe >= -2.0:
            ecl_score = max(0.0, 0.5 + sharpe * 0.25)
        else:
            ecl_score = 0.0

        # Bonus: did the agent adapt post-shock?
        post_shock_conservative = 0.0
        if state.macro_shock_active:
            post_shock_recs = []
            for i, rec in enumerate(state.applicants):
                if rec.action_taken is not None:
                    action = ActionType(rec.action_taken) if isinstance(rec.action_taken, str) else rec.action_taken
                    if i >= (config.shock_step or 0):
                        post_shock_recs.append(action)

            if post_shock_recs:
                conservative_count = sum(
                    1 for a in post_shock_recs
                    if a in (ActionType.REJECT, ActionType.COUNTER)
                )
                post_shock_conservative = conservative_count / len(post_shock_recs)

        # Fairness component
        fairness_score = self._fairness_score(state, config)

        # Weighted final score
        final = (ecl_score * 0.60 + post_shock_conservative * 0.25 + fairness_score * 0.15)

        return {
            "score": round(min(1.0, final), 4),
            "ecl_score": round(ecl_score, 4),
            "actual_ecl": round(actual_ecl, 6),
            "target_ecl": round(target_ecl, 4),
            "sharpe_ratio": round(sharpe, 4),
            "post_shock_conservative_rate": round(post_shock_conservative, 4),
            "fairness_score": round(fairness_score, 4),
        }

    def _fairness_score(self, state: EpisodeState, config: TaskConfig) -> float:
        ref_d = state.decisions_by_group.get("group_a", 0)
        ref_a = state.approvals_by_group.get("group_a", 0)
        prot_d = state.decisions_by_group.get("group_b", 0) + state.decisions_by_group.get("group_c", 0)
        prot_a = state.approvals_by_group.get("group_b", 0) + state.approvals_by_group.get("group_c", 0)

        if ref_d < 2 or prot_d < 2:
            return 0.5  # insufficient data — neutral

        ref_rate = ref_a / ref_d
        prot_rate = prot_a / prot_d
        gap = abs(ref_rate - prot_rate)

        if gap <= config.fairness_threshold:
            return 1.0
        elif gap <= config.fairness_threshold * 2:
            return 0.5
        else:
            return 0.0


# ─────────────────────────────────────────────
# Hard Grader — Multi-objective simultaneous
# ─────────────────────────────────────────────

class HardGrader(BaseGrader):
    """
    Task 3: Systemic Shock with Hidden Fraud Ring
    Score = weighted combination of:
    - Fraud detection recall (40%)
    - Portfolio stability under shock (30%)
    - Fairness parity maintained (15%)
    - False positive rate on legitimate applicants (15%)
    All must exceed minimum thresholds.
    """

    MIN_FRAUD_RECALL = 0.33  # Must catch at least 1 of 3 fraudsters
    MIN_ECL_RATIO = 0.5

    def score(self, state: EpisodeState, config: TaskConfig) -> Dict[str, float]:

        # 1. Fraud detection score
        fraud_recall = self._fraud_recall(state)
        fraud_precision = self._fraud_precision(state)
        fraud_f1 = 2 * fraud_recall * fraud_precision / max(fraud_recall + fraud_precision, 1e-8)

        # 2. Portfolio stability score
        ecl_ratio = state.portfolio_ecl / max(config.ecl_budget, 1e-8)
        if ecl_ratio <= 1.0:
            portfolio_score = 1.0
        elif ecl_ratio <= 1.5:
            portfolio_score = 1.0 - (ecl_ratio - 1.0) * 2
        else:
            portfolio_score = 0.0

        # 3. Fairness score
        fairness_score = self._fairness_score(state, config)

        # 4. False positive rate (flagging legitimate applicants as fraud)
        fpr_score = self._false_positive_score(state)

        # Apply minimum threshold gates
        gate_penalty = 0.0
        if fraud_recall < self.MIN_FRAUD_RECALL:
            gate_penalty += 0.25  # Major penalty: missed the fraud ring
        if ecl_ratio > 2.05:  # 5% buffer avoids penalising floating-point accumulation
            gate_penalty += 0.20  # Major penalty: blew the ECL budget

        raw_score = (
            fraud_f1 * 0.40
            + portfolio_score * 0.30
            + fairness_score * 0.15
            + fpr_score * 0.15
        )

        final = max(0.0, min(1.0, raw_score - gate_penalty))

        return {
            "score": round(final, 4),
            "fraud_f1": round(fraud_f1, 4),
            "fraud_recall": round(fraud_recall, 4),
            "fraud_precision": round(fraud_precision, 4),
            "portfolio_score": round(portfolio_score, 4),
            "ecl_ratio": round(ecl_ratio, 4),
            "fairness_score": round(fairness_score, 4),
            "false_positive_score": round(fpr_score, 4),
            "fraud_caught": state.fraud_caught,
            "fraud_missed": state.fraud_missed,
            "gate_penalty": round(gate_penalty, 4),
        }

    def _fraud_recall(self, state: EpisodeState) -> float:
        total_fraud = state.fraud_caught + state.fraud_missed
        if total_fraud == 0:
            return 1.0  # no fraudsters in episode
        return state.fraud_caught / total_fraud

    def _fraud_precision(self, state: EpisodeState) -> float:
        caught = state.fraud_caught
        false_flags = state.false_fraud_flags
        if caught + false_flags == 0:
            return 0.0
        return caught / (caught + false_flags)

    def _false_positive_score(self, state: EpisodeState) -> float:
        # Lower false fraud flags = better
        total_legit = sum(
            1 for r in state.applicants if not r.ground_truth_fraud
        )
        if total_legit == 0:
            return 1.0
        fpr = state.false_fraud_flags / total_legit
        return max(0.0, 1.0 - fpr * 3)

    def _fairness_score(self, state: EpisodeState, config: TaskConfig) -> float:
        ref_d = state.decisions_by_group.get("group_a", 0)
        ref_a = state.approvals_by_group.get("group_a", 0)
        prot_d = state.decisions_by_group.get("group_b", 0) + state.decisions_by_group.get("group_c", 0)
        prot_a = state.approvals_by_group.get("group_b", 0) + state.approvals_by_group.get("group_c", 0)

        if ref_d < 2 or prot_d < 2:
            return 0.5

        ref_rate = ref_a / max(ref_d, 1)
        prot_rate = prot_a / max(prot_d, 1)
        gap = abs(ref_rate - prot_rate)

        if gap <= config.fairness_threshold:
            return 1.0
        return max(0.0, 1.0 - (gap / config.fairness_threshold - 1) * 0.5)


# ─────────────────────────────────────────────
# Grader Registry
# ─────────────────────────────────────────────

GRADERS: Dict[str, BaseGrader] = {
    "easy": EasyGrader(),
    "medium": MediumGrader(),
    "hard": HardGrader(),
}


def grade_episode(state: EpisodeState, config: TaskConfig) -> Dict[str, float]:
    """Public interface: grade a completed episode."""
    grader = GRADERS.get(state.task_id)
    if grader is None:
        raise ValueError(f"No grader for task_id='{state.task_id}'")
    return grader.score(state, config)
