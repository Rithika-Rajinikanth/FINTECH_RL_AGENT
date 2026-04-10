"""
CreditLens — Task Graders  (v2 — strict open interval)
Each grader scores an episode strictly within (0.0, 1.0).

Validator rule: score must be > 0.0 AND < 1.0 (strictly open interval).

Changes from v1:
  - All scores clamped to [0.01, 0.99] via _clamp() before returning.
  - Internal sub-scores map to [0.03, 0.97] so weighted sums can never
    hit exact 0.0 or 1.0.
  - _clamp() is the single source of truth for the open-interval contract.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from creditlens.models import (
    ActionType,
    EpisodeState,
    TaskConfig,
)


# ─────────────────────────────────────────────
# Open-interval clamp — the single contract fix
# ─────────────────────────────────────────────

_SCORE_MIN = 0.01   # validator requires strictly > 0.0
_SCORE_MAX = 0.99   # validator requires strictly < 1.0


def _clamp(v: float) -> float:
    """Clamp the final score to the open interval (0, 1) the validator requires."""
    return round(max(_SCORE_MIN, min(_SCORE_MAX, float(v))), 4)


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
    Score = F1 x completion_rate, normalised to (0,1) open interval.

    Perfect performance (F1=1.0, completion=1.0):
      raw = 1.0 / 1.042 = 0.9597 -> clamped = 0.9597   (< 1.0 ✅)
    Zero performance:
      raw = 0.0  -> clamped = 0.01                       (> 0.0 ✅)
    """

    # Slightly above 1.0 so a perfect F1 score maps to ~0.96, never 1.0
    _NORM_DENOM = 1.042

    def score(self, state: EpisodeState, config: TaskConfig) -> Dict[str, float]:
        tp = fp = fn = tn = 0

        for rec in state.applicants:
            if rec.action_taken is None:
                continue

            action = (ActionType(rec.action_taken)
                      if isinstance(rec.action_taken, str)
                      else rec.action_taken)
            approved      = action in (ActionType.APPROVE, ActionType.COUNTER)
            should_approve = not rec.ground_truth_default and not rec.ground_truth_fraud

            if   approved     and     should_approve: tp += 1
            elif approved     and not should_approve: fp += 1
            elif not approved and     should_approve: fn += 1
            else:                                     tn += 1

        precision = tp / max(tp + fp, 1)
        recall    = tp / max(tp + fn, 1)
        f1        = 2 * precision * recall / max(precision + recall, 1e-8)

        processed       = sum(1 for r in state.applicants if r.action_taken is not None)
        completion_rate = processed / max(len(state.applicants), 1)

        # Divide by _NORM_DENOM: perfect score -> ~0.96, never 1.0
        normalised = (f1 * completion_rate) / self._NORM_DENOM

        return {
            "score":           _clamp(normalised),
            "f1":              round(f1, 4),
            "precision":       round(precision, 4),
            "recall":          round(recall, 4),
            "completion_rate": round(completion_rate, 4),
            "true_positives":  tp,
            "false_positives": fp,
            "false_negatives": fn,
            "true_negatives":  tn,
        }


# ─────────────────────────────────────────────
# Medium Grader — Sharpe-style ECL management
# ─────────────────────────────────────────────

class MediumGrader(BaseGrader):
    """
    Task 2: Portfolio Rebalancing Under Rate Shift

    Three sub-scores weighted and summed, all internally bounded to
    [0.03, 0.97] so the weighted sum never reaches exact 0 or 1.
    Final score clamped to [0.01, 0.99].
    """

    def score(self, state: EpisodeState, config: TaskConfig) -> Dict[str, float]:
        target_ecl = config.ecl_budget
        actual_ecl = state.portfolio_ecl

        ecl_deltas = [
            rec.reward_earned
            for rec in state.applicants
            if rec.action_taken is not None
        ]
        ecl_stddev = float(np.std(ecl_deltas)) if ecl_deltas else 0.01
        sharpe = (target_ecl - actual_ecl) / max(ecl_stddev, 0.01)

        # ECL sub-score: [0.03, 0.97]
        if sharpe >= 1.0:
            ecl_score = 0.97
        elif sharpe >= 0:
            ecl_score = 0.48 + sharpe * 0.49        # [0.48, 0.97]
        elif sharpe >= -2.0:
            ecl_score = max(0.03, 0.48 + sharpe * 0.225)
        else:
            ecl_score = 0.03

        # Post-shock conservatism: [0.03, 0.97]
        post_shock_conservative = 0.50  # neutral default (no shock data)
        if state.macro_shock_active:
            post_shock_recs = []
            for i, rec in enumerate(state.applicants):
                if rec.action_taken is not None:
                    a = (ActionType(rec.action_taken)
                         if isinstance(rec.action_taken, str)
                         else rec.action_taken)
                    if i >= (config.shock_step or 0):
                        post_shock_recs.append(a)
            if post_shock_recs:
                conservative_count = sum(
                    1 for a in post_shock_recs
                    if a in (ActionType.REJECT, ActionType.COUNTER)
                )
                raw_psc = conservative_count / len(post_shock_recs)
                post_shock_conservative = 0.03 + raw_psc * 0.94

        fairness_score = self._fairness_score(state, config)  # already in (0.03, 0.97)

        final = (ecl_score * 0.60
                 + post_shock_conservative * 0.25
                 + fairness_score * 0.15)

        return {
            "score":                        _clamp(final),
            "ecl_score":                    round(ecl_score, 4),
            "actual_ecl":                   round(actual_ecl, 6),
            "target_ecl":                   round(target_ecl, 4),
            "sharpe_ratio":                 round(sharpe, 4),
            "post_shock_conservative_rate": round(post_shock_conservative, 4),
            "fairness_score":               round(fairness_score, 4),
        }

    def _fairness_score(self, state: EpisodeState, config: TaskConfig) -> float:
        """Returns a value in [0.03, 0.97] — never exactly 0 or 1."""
        ref_d  = state.decisions_by_group.get("group_a", 0)
        ref_a  = state.approvals_by_group.get("group_a", 0)
        prot_d = (state.decisions_by_group.get("group_b", 0)
                  + state.decisions_by_group.get("group_c", 0))
        prot_a = (state.approvals_by_group.get("group_b", 0)
                  + state.approvals_by_group.get("group_c", 0))

        if ref_d < 2 or prot_d < 2:
            return 0.50  # insufficient data — neutral

        gap       = abs(ref_a / ref_d - prot_a / prot_d)
        threshold = config.fairness_threshold

        if gap <= threshold:
            return 0.97
        elif gap <= threshold * 2:
            t = (gap - threshold) / threshold     # 0 -> 1
            return round(0.97 - t * 0.47, 4)     # [0.50, 0.97]
        else:
            excess = min((gap - threshold * 2) / threshold, 1.0)
            return round(max(0.03, 0.50 - excess * 0.47), 4)


# ─────────────────────────────────────────────
# Hard Grader — Multi-objective simultaneous
# ─────────────────────────────────────────────

class HardGrader(BaseGrader):
    """
    Task 3: Systemic Shock with Hidden Fraud Ring

    Four sub-scores, all mapped to [0.03, 0.97] before weighting.
    Gate penalties may reduce the total.
    Final score clamped to [0.01, 0.99] by _clamp().
    """

    MIN_FRAUD_RECALL = 0.33

    def score(self, state: EpisodeState, config: TaskConfig) -> Dict[str, float]:

        # Raw metrics (unbounded helpers)
        fraud_recall    = self._fraud_recall(state)
        fraud_precision = self._fraud_precision(state)
        ecl_ratio       = state.portfolio_ecl / max(config.ecl_budget, 1e-8)

        # Sub-scores mapped to [0.03, 0.97]
        raw_f1      = (2 * fraud_recall * fraud_precision
                       / max(fraud_recall + fraud_precision, 1e-8))
        fraud_f1    = round(0.03 + raw_f1 * 0.94, 4)

        if ecl_ratio <= 1.0:
            raw_port = 1.0
        elif ecl_ratio <= 1.5:
            raw_port = 1.0 - (ecl_ratio - 1.0) * 2
        else:
            raw_port = 0.0
        portfolio_score = round(0.03 + raw_port * 0.94, 4)

        fairness_score = self._fairness_score(state, config)  # in [0.03, 0.97]

        raw_fpr   = self._false_positive_score(state)         # in [0.01, 0.99]
        fpr_score = round(0.03 + raw_fpr * 0.94, 4)

        # Gate penalties
        gate_penalty = 0.0
        if fraud_recall < self.MIN_FRAUD_RECALL:
            gate_penalty += 0.25
        if ecl_ratio > 2.05:
            gate_penalty += 0.20

        final = (fraud_f1        * 0.40
                 + portfolio_score * 0.30
                 + fairness_score  * 0.15
                 + fpr_score       * 0.15
                 - gate_penalty)

        return {
            "score":                _clamp(final),
            "fraud_f1":             round(fraud_f1, 4),
            "fraud_recall":         round(fraud_recall, 4),
            "fraud_precision":      round(fraud_precision, 4),
            "portfolio_score":      round(portfolio_score, 4),
            "ecl_ratio":            round(ecl_ratio, 4),
            "fairness_score":       round(fairness_score, 4),
            "false_positive_score": round(fpr_score, 4),
            "fraud_caught":         state.fraud_caught,
            "fraud_missed":         state.fraud_missed,
            "gate_penalty":         round(gate_penalty, 4),
        }

    def _fraud_recall(self, state: EpisodeState) -> float:
        total = state.fraud_caught + state.fraud_missed
        if total == 0:
            return 0.99   # no fraudsters → near-perfect but not 1.0
        return state.fraud_caught / total

    def _fraud_precision(self, state: EpisodeState) -> float:
        caught      = state.fraud_caught
        false_flags = state.false_fraud_flags
        if caught + false_flags == 0:
            return 0.01   # no fraud decisions → near-zero precision
        return caught / (caught + false_flags)

    def _false_positive_score(self, state: EpisodeState) -> float:
        total_legit = sum(1 for r in state.applicants if not r.ground_truth_fraud)
        if total_legit == 0:
            return 0.99
        fpr = state.false_fraud_flags / total_legit
        return max(0.01, 1.0 - fpr * 3)

    def _fairness_score(self, state: EpisodeState, config: TaskConfig) -> float:
        """Returns a value in [0.03, 0.97] — never exactly 0 or 1."""
        ref_d  = state.decisions_by_group.get("group_a", 0)
        ref_a  = state.approvals_by_group.get("group_a", 0)
        prot_d = (state.decisions_by_group.get("group_b", 0)
                  + state.decisions_by_group.get("group_c", 0))
        prot_a = (state.approvals_by_group.get("group_b", 0)
                  + state.approvals_by_group.get("group_c", 0))

        if ref_d < 2 or prot_d < 2:
            return 0.50

        gap       = abs(ref_a / max(ref_d, 1) - prot_a / max(prot_d, 1))
        threshold = config.fairness_threshold

        if gap <= threshold:
            return 0.97
        excess = min((gap - threshold) / threshold, 2.0)
        return round(max(0.03, 0.97 - excess * 0.47), 4)


# ─────────────────────────────────────────────
# Grader Registry
# ─────────────────────────────────────────────

GRADERS: Dict[str, BaseGrader] = {
    "easy":   EasyGrader(),
    "medium": MediumGrader(),
    "hard":   HardGrader(),
}


def grade_episode(state: EpisodeState, config: TaskConfig) -> Dict[str, float]:
    """Public interface: grade a completed episode."""
    grader = GRADERS.get(state.task_id)
    if grader is None:
        raise ValueError(f"No grader for task_id='{state.task_id}'")
    return grader.score(state, config)
