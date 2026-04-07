"""
CreditLens — Synthetic Data Generation Pipeline
Generates loans.parquet with realistic correlated applicant data,
XGBoost default probabilities, SHAP explanations, and fraud ring scores.
All free, no external API calls required.
"""

from __future__ import annotations

import random
import warnings
from pathlib import Path
from typing import List, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import shap
from faker import Faker
from imblearn.over_sampling import SMOTE
from loguru import logger
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import joblib

warnings.filterwarnings("ignore")

fake = Faker()
Faker.seed(42)
np.random.seed(42)
random.seed(42)

DATA_DIR = Path(__file__).parent
MODEL_DIR = DATA_DIR.parent.parent / "artifacts"
PARQUET_PATH = DATA_DIR / "loans.parquet"
MODEL_PATH = MODEL_DIR / "xgb_model.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# 1. Applicant Generation
# ─────────────────────────────────────────────

LOAN_PURPOSES = ["home", "auto", "education", "business", "personal", "medical"]
DEMOGRAPHIC_GROUPS = ["group_a", "group_b", "group_c"]
DEMOGRAPHIC_WEIGHTS = [0.60, 0.25, 0.15]  # reference, protected_1, protected_2


def _correlated_fico(income: float, derogatory: int) -> int:
    """FICO score correlated with income and derogatory marks."""
    base = 580 + (income / 200_000) * 200 - derogatory * 35
    noise = np.random.normal(0, 30)
    return int(np.clip(base + noise, 300, 850))


def _compute_dti(income: float, loan_amount: float, existing_debt: float) -> float:
    monthly_income = income / 12
    monthly_debt = (existing_debt * 0.02) + (loan_amount * 0.015)
    return round(min(monthly_debt / (monthly_income + 1e-6), 2.0), 4)


def generate_applicant(
    applicant_id: str,
    fraud: bool = False,
    fraud_ring_id: str | None = None,
    shared_phone: str | None = None,
    shared_employer: str | None = None,
) -> dict:
    """Generate one realistic applicant record."""

    demographic = np.random.choice(DEMOGRAPHIC_GROUPS, p=DEMOGRAPHIC_WEIGHTS)
    income_base = {"group_a": 75_000, "group_b": 58_000, "group_c": 50_000}[demographic]
    income = max(18_000, np.random.lognormal(np.log(income_base), 0.45))

    derogatory = np.random.choice([0, 1, 2, 3], p=[0.60, 0.22, 0.12, 0.06])
    fico = _correlated_fico(income, derogatory)

    employment_years = max(0.0, np.random.exponential(5.0))
    loan_purpose = random.choice(LOAN_PURPOSES)

    loan_amount_map = {
        "home": (50_000, 500_000),
        "auto": (5_000, 60_000),
        "education": (5_000, 80_000),
        "business": (10_000, 200_000),
        "personal": (1_000, 50_000),
        "medical": (1_000, 30_000),
    }
    lo, hi = loan_amount_map[loan_purpose]
    loan_amount = float(np.random.uniform(lo, hi))

    existing_debt = float(np.random.lognormal(np.log(15_000), 0.8))
    credit_utilization = float(np.clip(np.random.beta(2, 5), 0, 1))
    payment_history_score = float(np.clip(1.0 - (derogatory * 0.25) + np.random.normal(0, 0.05), 0, 1))
    num_open_accounts = int(np.random.poisson(4))
    dti_ratio = _compute_dti(income, loan_amount, existing_debt)

    ltv_ratio = None
    if loan_purpose == "home":
        property_value = loan_amount * np.random.uniform(1.1, 2.5)
        ltv_ratio = round(loan_amount / property_value, 4)

    # Fraud overrides — synthetic identity characteristics
    phone = shared_phone if fraud and shared_phone else fake.phone_number()
    employer_id = shared_employer if fraud and shared_employer else str(fake.uuid4())[:8]

    # Default label — logistic model of risk
    default_logit = (
        -3.5
        + (750 - fico) * 0.012
        + dti_ratio * 2.5
        + derogatory * 0.8
        - employment_years * 0.05
        + credit_utilization * 1.2
        - (income / 100_000) * 0.4
    )
    if fraud:
        default_logit += 3.0  # fraudsters almost always default

    default_prob_true = 1 / (1 + np.exp(-default_logit))
    will_default = bool(np.random.random() < default_prob_true)

    return {
        "applicant_id": applicant_id,
        "demographic_group": demographic,
        "fico_score": fico,
        "income": round(income, 2),
        "loan_amount": round(loan_amount, 2),
        "loan_purpose": loan_purpose,
        "employment_years": round(employment_years, 2),
        "dti_ratio": dti_ratio,
        "credit_utilization": round(credit_utilization, 4),
        "payment_history_score": round(payment_history_score, 4),
        "num_open_accounts": num_open_accounts,
        "num_derogatory_marks": derogatory,
        "ltv_ratio": ltv_ratio,
        "existing_debt": round(existing_debt, 2),
        "phone": phone,
        "employer_id": employer_id,
        "fraud_ring_id": fraud_ring_id,
        "is_fraud": fraud,
        "will_default": will_default,
        "default_prob_true": round(default_prob_true, 4),
    }


def generate_dataset(n: int = 5000, fraud_fraction: float = 0.05) -> pd.DataFrame:
    """Generate n applicants with realistic fraud rings."""
    records = []
    n_fraud_rings = max(1, int(n * fraud_fraction / 3))
    fraud_pool: List[Tuple[str, str, str]] = []  # (ring_id, phone, employer)

    for i in range(n_fraud_rings):
        ring_id = f"RING_{i:04d}"
        phone = fake.phone_number()
        employer = str(fake.uuid4())[:8]
        fraud_pool.append((ring_id, phone, employer))

    fraud_applicant_ids = set(
        random.sample(range(n), min(int(n * fraud_fraction), n))
    )

    for i in range(n):
        aid = f"APP_{i:06d}"
        if i in fraud_applicant_ids and fraud_pool:
            ring_id, phone, employer = random.choice(fraud_pool)
            rec = generate_applicant(aid, fraud=True, fraud_ring_id=ring_id,
                                     shared_phone=phone, shared_employer=employer)
        else:
            rec = generate_applicant(aid)
        records.append(rec)

    df = pd.DataFrame(records)
    logger.info(f"Generated {len(df)} applicants, fraud rate={df['is_fraud'].mean():.2%}")
    return df


# ─────────────────────────────────────────────
# 2. Fraud Graph Features
# ─────────────────────────────────────────────

def compute_fraud_graph_features(df: pd.DataFrame) -> pd.DataFrame:
    """Use NetworkX to compute fraud_ring_score from shared attributes."""
    G = nx.Graph()

    for _, row in df.iterrows():
        G.add_node(row["applicant_id"])

    # Connect applicants sharing phone numbers
    phone_groups = df.groupby("phone")["applicant_id"].apply(list)
    for group in phone_groups:
        if len(group) > 1:
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    G.add_edge(group[i], group[j], weight=2, type="shared_phone")

    # Connect applicants sharing employer IDs
    employer_groups = df.groupby("employer_id")["applicant_id"].apply(list)
    for group in employer_groups:
        if len(group) > 1:
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    if G.has_edge(group[i], group[j]):
                        G[group[i]][group[j]]["weight"] += 1
                    else:
                        G.add_edge(group[i], group[j], weight=1, type="shared_employer")

    # Compute per-node fraud score based on cluster density
    fraud_scores = {}
    shared_phones = {}
    shared_employers = {}
    cluster_sizes = {}

    for component in nx.connected_components(G):
        subgraph = G.subgraph(component)
        size = len(component)
        density = nx.density(subgraph) if size > 1 else 0.0
        # Edge weight sum normalised
        weight_sum = sum(d.get("weight", 1) for _, _, d in subgraph.edges(data=True))
        score = min(1.0, (density * 0.5 + weight_sum / max(size, 1) * 0.3 + (size > 2) * 0.2))

        for node in component:
            fraud_scores[node] = round(score, 4)
            cluster_sizes[node] = size

    for _, row in df.iterrows():
        aid = row["applicant_id"]
        neighbors = list(G.neighbors(aid))
        phone_shared = any(
            G[aid][n].get("type") == "shared_phone" for n in neighbors
        )
        employer_shared = any(
            G[aid][n].get("type") == "shared_employer" for n in neighbors
        )
        shared_phones[aid] = phone_shared
        shared_employers[aid] = employer_shared

    df["fraud_ring_score"] = df["applicant_id"].map(fraud_scores).fillna(0.0)
    df["shared_phone"] = df["applicant_id"].map(shared_phones).fillna(False)
    df["shared_employer_id"] = df["applicant_id"].map(shared_employers).fillna(False)
    df["graph_cluster_size"] = df["applicant_id"].map(cluster_sizes).fillna(1).astype(int)

    logger.info(f"Fraud graph built. Nodes={G.number_of_nodes()}, Edges={G.number_of_edges()}")
    return df


# ─────────────────────────────────────────────
# 3. XGBoost Training + SHAP
# ─────────────────────────────────────────────

FEATURE_COLS = [
    "fico_score", "income", "loan_amount", "employment_years",
    "dti_ratio", "credit_utilization", "payment_history_score",
    "num_open_accounts", "num_derogatory_marks", "fraud_ring_score",
]

SHAP_FEATURE_NAMES = [
    "fico_score", "income", "loan_amount", "employment_years",
    "dti_ratio", "credit_utilization", "payment_history_score",
    "num_open_accounts", "num_derogatory_marks", "fraud_ring_score",
]


def train_xgboost(df: pd.DataFrame) -> Tuple[Pipeline, pd.DataFrame]:
    """Train XGBoost default classifier and add predictions to dataframe."""
    X = df[FEATURE_COLS].copy()
    y = df["will_default"].astype(int)

    # SMOTE to handle class imbalance
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    try:
        sm = SMOTE(random_state=42, k_neighbors=min(5, y_train.sum() - 1))
        X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
    except ValueError:
        X_train_res, y_train_res = X_train, y_train

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_res)

    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train_scaled, y_train_res, verbose=False)

    # Predict on full dataset
    X_all_scaled = scaler.transform(X)
    xgb_probs = model.predict_proba(X_all_scaled)[:, 1]
    df["xgb_default_prob"] = np.round(xgb_probs, 4)

    # SHAP explanations
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_all_scaled)

    top_features = []
    top_shap_values = []
    for i in range(len(df)):
        sv = shap_values[i]
        top_idx = int(np.argmax(np.abs(sv)))
        top_features.append(SHAP_FEATURE_NAMES[top_idx])
        top_shap_values.append(round(float(sv[top_idx]), 4))

    df["shap_top_feature"] = top_features
    df["shap_top_value"] = top_shap_values

    # Save artifacts
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    logger.info(f"XGBoost trained. Saved to {MODEL_PATH}")

    return model, df


# ─────────────────────────────────────────────
# 4. Macro Context (Static — no API needed)
# ─────────────────────────────────────────────

def add_macro_context(df: pd.DataFrame) -> pd.DataFrame:
    """Add realistic macro-economic context (static values, no API key needed)."""
    # Use realistic 2024-era values
    df["fed_funds_rate"] = 5.33
    df["treasury_yield_10y"] = 4.25
    df["unemployment_rate"] = 3.9
    df["macro_shock_active"] = False
    df["shock_magnitude_bps"] = 0
    return df


# ─────────────────────────────────────────────
# 5. Ground Truth Labels for Easy Task
# ─────────────────────────────────────────────

def compute_ground_truth_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a deterministic ground truth approval label used by the easy task grader.
    Based on standard underwriting rules (not ML — interpretable, auditable).
    """
    conditions = (
        (df["fico_score"] >= 620)
        & (df["dti_ratio"] <= 0.43)
        & (df["will_default"] == False)
        & (df["is_fraud"] == False)
        & (df["num_derogatory_marks"] <= 1)
    )
    df["ground_truth_approve"] = conditions
    return df


# ─────────────────────────────────────────────
# 6. Main Pipeline
# ─────────────────────────────────────────────

def run_pipeline(n: int = 5000) -> pd.DataFrame:
    logger.info("=== CreditLens Data Generation Pipeline ===")

    logger.info("Step 1/5: Generating synthetic applicants...")
    df = generate_dataset(n=n)

    logger.info("Step 2/5: Computing fraud graph features...")
    df = compute_fraud_graph_features(df)

    logger.info("Step 3/5: Training XGBoost + computing SHAP...")
    _, df = train_xgboost(df)

    logger.info("Step 4/5: Adding macro context...")
    df = add_macro_context(df)

    logger.info("Step 5/5: Computing ground truth labels...")
    df = compute_ground_truth_labels(df)

    PARQUET_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PARQUET_PATH, index=False)
    logger.info(f"Saved {len(df)} records to {PARQUET_PATH}")
    logger.info(f"  Default rate:  {df['will_default'].mean():.2%}")
    logger.info(f"  Fraud rate:    {df['is_fraud'].mean():.2%}")
    logger.info(f"  Approval rate: {df['ground_truth_approve'].mean():.2%}")

    return df


if __name__ == "__main__":
    run_pipeline(n=5000)
