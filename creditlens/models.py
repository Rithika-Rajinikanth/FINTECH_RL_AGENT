"""
CreditLens — Pydantic models for Observation, Action, Reward, and Episode State.
All types are fully typed and validated.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator
import uuid


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ActionType(str, Enum):
    APPROVE = "APPROVE"
    REJECT = "REJECT"
    COUNTER = "COUNTER"
    REQUEST_INFO = "REQUEST_INFO"


class LoanPurpose(str, Enum):
    HOME = "home"
    AUTO = "auto"
    EDUCATION = "education"
    BUSINESS = "business"
    PERSONAL = "personal"
    MEDICAL = "medical"


class DemographicGroup(str, Enum):
    GROUP_A = "group_a"   # Reference group
    GROUP_B = "group_b"   # Protected group 1
    GROUP_C = "group_c"   # Protected group 2


class RejectReason(str, Enum):
    HIGH_DTI = "HIGH_DTI"
    LOW_CREDIT_SCORE = "LOW_CREDIT_SCORE"
    INSUFFICIENT_INCOME = "INSUFFICIENT_INCOME"
    FRAUD_SUSPECTED = "FRAUD_SUSPECTED"
    INCOMPLETE_APPLICATION = "INCOMPLETE_APPLICATION"
    RECENT_DEROGATORY = "RECENT_DEROGATORY"


class RequestField(str, Enum):
    INCOME_PROOF = "income_proof"
    EMPLOYMENT_LETTER = "employment_letter"
    BANK_STATEMENTS = "bank_statements"
    TAX_RETURNS = "tax_returns"
    IDENTITY_DOCUMENT = "identity_document"


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

class LoanObservation(BaseModel):
    """Complete observation delivered to the agent at each step."""

    # Identity
    applicant_id: str = Field(..., description="Unique applicant ID")
    episode_id: str = Field(..., description="Current episode ID")
    step_number: int = Field(..., ge=0, description="Current step within episode")

    # Applicant profile
    fico_score: int = Field(..., ge=300, le=850)
    income: float = Field(..., ge=0.0, description="Annual income in USD")
    loan_amount: float = Field(..., ge=0.0)
    loan_purpose: LoanPurpose
    employment_years: float = Field(..., ge=0.0)
    dti_ratio: float = Field(..., ge=0.0, le=2.0, description="Debt-to-income ratio")
    credit_utilization: float = Field(..., ge=0.0, le=1.0)
    payment_history_score: float = Field(..., ge=0.0, le=1.0)
    num_open_accounts: int = Field(..., ge=0)
    num_derogatory_marks: int = Field(..., ge=0)
    ltv_ratio: Optional[float] = Field(None, description="Loan-to-value (home loans only)")

    # ML risk signals
    xgb_default_prob: float = Field(..., ge=0.0, le=1.0, description="XGBoost default probability")
    shap_top_feature: str = Field(..., description="Top SHAP feature driving XGBoost prediction")
    shap_top_value: float = Field(..., description="SHAP value of top feature")

    # Fraud signals
    fraud_ring_score: float = Field(..., ge=0.0, le=1.0, description="NetworkX fraud ring probability")
    shared_phone: bool = Field(False)
    shared_employer_id: bool = Field(False)
    graph_cluster_size: int = Field(1, ge=1)

    # Demographics (for fairness tracking — not for direct decision use)
    demographic_group: DemographicGroup

    # Macroeconomic context
    fed_funds_rate: float = Field(..., description="Current Fed Funds Rate %")
    treasury_yield_10y: float = Field(..., description="10Y Treasury Yield %")
    unemployment_rate: float = Field(..., description="Current unemployment %")
    macro_shock_active: bool = Field(False, description="True if rate shock event is active")
    shock_magnitude_bps: int = Field(0, description="Size of rate shock in basis points")

    # Portfolio state
    portfolio_ecl: float = Field(..., description="Current portfolio Expected Credit Loss")
    portfolio_ecl_budget: float = Field(..., description="ECL budget for episode")
    approval_rate_protected: float = Field(..., ge=0.0, le=1.0,
                                           description="Approval rate for protected groups so far")
    approval_rate_reference: float = Field(..., ge=0.0, le=1.0,
                                           description="Approval rate for reference group so far")
    steps_remaining: int = Field(..., ge=0)

    # Info requests outstanding
    pending_info_requests: List[str] = Field(default_factory=list)

    class Config:
        use_enum_values = True


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

class ApproveParams(BaseModel):
    amount_fraction: float = Field(1.0, ge=0.5, le=1.0,
                                   description="Fraction of requested loan to approve")


class RejectParams(BaseModel):
    reason_code: RejectReason


class CounterParams(BaseModel):
    revised_amount_fraction: float = Field(..., ge=0.25, le=0.9)
    revised_rate_delta: float = Field(..., ge=0.0, le=5.0,
                                      description="Additional rate in percentage points")


class RequestInfoParams(BaseModel):
    field_name: RequestField


class UnderwritingAction(BaseModel):
    """Action returned by the agent."""
    action_type: ActionType
    applicant_id: str
    params: Dict[str, Any] = Field(default_factory=dict)
    reasoning: Optional[str] = Field(None, description="Agent's chain-of-thought (logged)")

    @field_validator("params")
    @classmethod
    def validate_params(cls, v: dict, info: Any) -> dict:
        return v

    class Config:
        use_enum_values = True


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------

class RewardBreakdown(BaseModel):
    """Detailed reward decomposition for transparency."""
    base_reward: float
    ecl_penalty: float = 0.0
    fairness_penalty: float = 0.0
    fraud_catch_bonus: float = 0.0
    step_cost: float = 0.0
    info_request_cost: float = 0.0
    macro_adaptation_bonus: float = 0.0
    total: float

    @classmethod
    def compute(
        cls,
        base: float,
        ecl_penalty: float = 0.0,
        fairness_penalty: float = 0.0,
        fraud_catch: float = 0.0,
        step_cost: float = 0.0,
        info_cost: float = 0.0,
        macro_bonus: float = 0.0,
    ) -> "RewardBreakdown":
        total = base - ecl_penalty - fairness_penalty + fraud_catch - step_cost - info_cost + macro_bonus
        return cls(
            base_reward=base,
            ecl_penalty=ecl_penalty,
            fairness_penalty=fairness_penalty,
            fraud_catch_bonus=fraud_catch,
            step_cost=step_cost,
            info_request_cost=info_cost,
            macro_adaptation_bonus=macro_bonus,
            total=total,
        )


class StepResult(BaseModel):
    """Returned by step() — mirrors OpenEnv spec."""
    observation: Optional[LoanObservation]
    reward: float
    reward_breakdown: RewardBreakdown
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Episode State
# ---------------------------------------------------------------------------

class ApplicantRecord(BaseModel):
    applicant_id: str
    action_taken: Optional[ActionType] = None
    action_params: Dict[str, Any] = Field(default_factory=dict)
    ground_truth_default: bool = False
    ground_truth_fraud: bool = False
    reward_earned: float = 0.0
    info_requests: int = 0


class EpisodeState(BaseModel):
    episode_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    task_id: str
    step: int = 0
    max_steps: int
    done: bool = False
    applicants: List[ApplicantRecord] = Field(default_factory=list)
    current_applicant_index: int = 0
    portfolio_ecl: float = 0.0
    portfolio_ecl_budget: float = 0.05
    total_reward: float = 0.0
    macro_shock_step: Optional[int] = None
    macro_shock_active: bool = False
    shock_magnitude_bps: int = 0
    approvals_by_group: Dict[str, int] = Field(default_factory=lambda: {"group_a": 0, "group_b": 0, "group_c": 0})
    decisions_by_group: Dict[str, int] = Field(default_factory=lambda: {"group_a": 0, "group_b": 0, "group_c": 0})
    fraud_caught: int = 0
    fraud_missed: int = 0
    false_fraud_flags: int = 0


class TaskConfig(BaseModel):
    task_id: str
    name: str
    num_applicants: int
    max_steps: int
    ecl_budget: float
    fraud_ring_size: int = 0
    macro_shock: bool = False
    shock_step: Optional[int] = None
    shock_bps: int = 0
    fairness_threshold: float = 0.15
