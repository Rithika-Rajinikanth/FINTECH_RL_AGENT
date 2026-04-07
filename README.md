# CreditLens — AI Credit Risk Underwriting Environment

> **OpenEnv-compliant** · **100% Free Stack** · **Production-Grade** · **Multi-objective RL**

CreditLens simulates what a real loan officer does: decide whether to **approve**, **reject**, **counter-offer**, or **request more information** from a loan applicant — while balancing credit risk, regulatory fairness requirements, portfolio stability, and fraud detection simultaneously.

No existing OpenEnv submission models this problem. CreditLens fills a genuine gap: banks and fintechs need safe, evaluable AI for exactly this workflow.

---

## Table of Contents

1. [Why CreditLens?](#why-creditlens)
2. [Architecture](#architecture)
3. [Observation Space](#observation-space)
4. [Action Space](#action-space)
5. [Reward Function](#reward-function)
6. [Tasks](#tasks)
7. [Quick Start](#quick-start)
8. [Running with Ollama (Free LLM)](#running-with-ollama-free-llm)
9. [PPO Training](#ppo-training)
10. [API Reference](#api-reference)
11. [Baseline Scores](#baseline-scores)
12. [Tech Stack](#tech-stack)
13. [Project Structure](#project-structure)

---

## Why CreditLens?

| Dimension | Other OpenEnv submissions | CreditLens |
|-----------|--------------------------|------------|
| Domain | Price prediction, copy-trading | Real loan underwriting |
| Actions | Buy / Sell / Hold (3) | Approve / Reject / Counter / Request Info (4) |
| Reward | Single objective | 4 competing objectives |
| Fairness | None | ECOA demographic parity penalty |
| Fraud | None | NetworkX fraud ring detection |
| Macro shocks | None | Fed rate shifts mid-episode |
| LLM integration | Hint only | XGBoost + SHAP as first-class observation fields |

### The Four-Action Underwriting Workflow

```
┌─────────────┐
│   APPROVE   │ ← full or partial amount (amount_fraction: 0.5–1.0)
├─────────────┤
│   REJECT    │ ← with regulatory reason code (ECOA compliant)
├─────────────┤
│   COUNTER   │ ← lower limit + higher rate (revised_amount_fraction + rate_delta)
├─────────────┤
│ REQUEST_INFO│ ← ask for a missing document (costs 1 step, same applicant next turn)
└─────────────┘
```

This makes the reward function **genuinely interesting**:
- Requesting info when you already have enough → wastes steps
- Approving a fraudulent applicant → large penalty
- Rejecting a creditworthy minority applicant → triggers fairness penalty
- Ignoring a rate shock → ECL budget blown

---

## Architecture

```
Synthetic Data (Faker + SDV)
        │
        ▼
Feature Engineering (Pandas + scikit-learn)
        │
        ▼
XGBoost Default Model + SHAP Explanations
        │
        ▼
NetworkX Fraud Graph → fraud_ring_score
        │
        ▼
  loans.parquet  ──────────────────────────────────────────┐
                                                           │
                                                    ┌──────▼──────┐
                                                    │ CreditLens  │
                                                    │    Env      │
                                                    │  engine.py  │
                                                    └──────┬──────┘
                                                           │
                           ┌───────────────────────────────┤
                           │                               │
                    ┌──────▼──────┐                ┌───────▼──────┐
                    │  PPO Agent  │                │  LLM Agent   │
                    │ (SB3 train) │                │  (Ollama)    │
                    └─────────────┘                └──────────────┘
                                                           │
                                                   ┌───────▼──────┐
                                                   │  FastAPI     │
                                                   │  /reset      │
                                                   │  /step       │
                                                   │  /state      │
                                                   │  /grade      │
                                                   └──────────────┘
```

---

## Observation Space

Every step delivers a `LoanObservation` with 17 fields across 5 categories:

### Applicant Profile
| Field | Type | Range | Description |
|-------|------|-------|-------------|
| `fico_score` | int | 300–850 | Credit score |
| `income` | float | ≥0 | Annual income USD |
| `loan_amount` | float | ≥0 | Requested loan USD |
| `loan_purpose` | enum | 6 values | home/auto/education/business/personal/medical |
| `employment_years` | float | ≥0 | Years at current employer |
| `dti_ratio` | float | 0–2.0 | Debt-to-income ratio |
| `credit_utilization` | float | 0–1.0 | Revolving credit used |
| `payment_history_score` | float | 0–1.0 | 1.0 = perfect history |
| `num_derogatory_marks` | int | ≥0 | Collections, bankruptcies |

### ML Risk Signals
| Field | Type | Description |
|-------|------|-------------|
| `xgb_default_prob` | float 0–1 | XGBoost 12-month default probability |
| `shap_top_feature` | string | Feature most driving XGBoost prediction |
| `shap_top_value` | float | SHAP value of that feature |

### Fraud Signals
| Field | Type | Description |
|-------|------|-------------|
| `fraud_ring_score` | float 0–1 | NetworkX fraud ring membership probability |
| `shared_phone` | bool | Phone shared with another applicant |
| `shared_employer_id` | bool | Employer ID shared with another applicant |
| `graph_cluster_size` | int | Size of connected applicant cluster |

### Macroeconomic Context
| Field | Type | Description |
|-------|------|-------------|
| `fed_funds_rate` | float | Current Fed Funds Rate % |
| `treasury_yield_10y` | float | 10Y Treasury Yield % |
| `macro_shock_active` | bool | True when rate shock event is live |
| `shock_magnitude_bps` | int | Size of shock in basis points |

### Portfolio State
| Field | Type | Description |
|-------|------|-------------|
| `portfolio_ecl` | float | Accumulated Expected Credit Loss |
| `portfolio_ecl_budget` | float | ECL budget for this episode |
| `approval_rate_protected` | float | Approval rate for protected groups |
| `approval_rate_reference` | float | Approval rate for reference group |
| `steps_remaining` | int | Steps left in episode |

---

## Action Space

```python
class ActionType(str, Enum):
    APPROVE = "APPROVE"      # params: {"amount_fraction": 1.0}
    REJECT = "REJECT"        # params: {"reason_code": "HIGH_DTI"}
    COUNTER = "COUNTER"      # params: {"revised_amount_fraction": 0.7, "revised_rate_delta": 1.5}
    REQUEST_INFO = "REQUEST_INFO"  # params: {"field_name": "income_proof"}
```

**Reject reason codes** (ECOA-compliant): `HIGH_DTI`, `LOW_CREDIT_SCORE`, `INSUFFICIENT_INCOME`, `FRAUD_SUSPECTED`, `INCOMPLETE_APPLICATION`, `RECENT_DEROGATORY`

**Request info fields**: `income_proof`, `employment_letter`, `bank_statements`, `tax_returns`, `identity_document`

---

## Reward Function

The reward has **7 components** with competing objectives — this is what makes CreditLens hard:

```
reward = base_reward
       - ecl_penalty          # approving defaulters costs money
       - fairness_penalty      # demographic parity gap → ECOA penalty
       + fraud_catch_bonus     # correctly rejecting fraudsters
       - step_cost             # small fixed cost per step
       - info_request_cost     # penalises unnecessary info requests
       + macro_adaptation_bonus # conservative after rate shock
```

| Component | Value | Trigger |
|-----------|-------|---------|
| Correct approve | +0.30 | Approve a non-defaulter |
| Correct reject | +0.25 | Reject a defaulter or fraudster |
| Wrong approve | −0.40 | Approve a defaulter |
| Wrong reject | −0.20 | Reject a creditworthy applicant |
| Fraud catch bonus | +0.50 | Reject a fraud ring member |
| Fraud miss penalty | −0.60 | Approve a fraudster |
| ECL penalty | 0 to −0.5 | Proportional to PD × LGD × EAD |
| Fairness penalty | 0 to −0.30 | Gap > threshold (ECOA) |
| Step cost | −0.01 | Every step |
| Info request cost | −0.05 | REQUEST_INFO action |
| Macro adapt bonus | +0.10 | Conservative decision post-shock |

---

## Tasks

### Task 1 — Easy: Loan Approval Queue

10 synthetic applicants, clean FICO scores, no edge cases, no macro shocks, no fraud.

**Grader**: F1 score against ground truth labels.
- Score 1.0 if F1 ≥ 0.8
- Linear below (Score = F1 / 0.8)
- Multiplied by completion rate

### Task 2 — Medium: Portfolio Rebalancing Under Rate Shift

15 applicants. Fed raises rates **+75 bps** at step 8. Agent must rebalance toward fixed-rate approvals to keep ECL under budget.

**Grader**: Sharpe-style ECL management
- ECL score (60%) + post-shock conservatism (25%) + fairness (15%)
- Partial credit for getting direction right even if magnitude is off

### Task 3 — Hard: Systemic Shock with Hidden Fraud Ring

20 applicants, **3 share synthetic identities** in a fraud ring (detectable via `fraud_ring_score`, `shared_phone`, `shared_employer_id`). Macro shock at step 10 (+100 bps).

**Grader**: Multi-objective simultaneous
- Fraud F1 (40%) + portfolio stability (30%) + fairness (15%) + false positive rate (15%)
- Gate penalties: missing the fraud ring (−0.25), blowing ECL budget (−0.20)
- Frontier LLMs struggle because it requires multi-step reasoning **across** applicants

---

## Quick Start

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.ai) (free, local LLM — no API key)

### Installation

```bash
# Clone
git clone https://github.com/your-org/creditlens
cd creditlens

# Install dependencies
pip install -e .

# Generate dataset + train XGBoost (runs once, ~2 minutes)
python -m creditlens.data.generate

# Start the API server
uvicorn creditlens.inference.service:app --port 8000

# In another terminal — run inference
python inference.py --task all --seed 42
```

### Docker

```bash
# Build (generates data at build time)
docker build -f docker/Dockerfile -t creditlens .

# Run API server
docker run -p 8000:8000 creditlens

# Run inference against the container
python inference.py --task all
```

### Docker Compose (API + Ollama together)

```bash
cd docker
docker compose up

# Pull a free LLM model into Ollama
docker exec creditlens_ollama ollama pull llama3.2

# Run inference
python inference.py --task all
```

---

## Running with Ollama (Free LLM)

CreditLens uses **Ollama** for local LLM inference — completely free, no API key, no internet after model download.

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull a free model (choose based on your RAM):
ollama pull llama3.2      # recommended, 2GB, fast
ollama pull mistral       # alternative, 4GB
ollama pull phi3          # lightweight, 1.8GB (low RAM)

# The agent auto-detects whichever is installed
python inference.py --task all
```

The agent uses structured prompting:
1. Receives `LoanObservation` JSON
2. Reasons about `xgb_default_prob`, `fraud_ring_score`, `shap_top_feature`, fairness gap
3. Returns one of 4 actions with chain-of-thought reasoning
4. Falls back to rule-based decisions if Ollama is unavailable

---

## PPO Training

Train a PPO baseline agent using Stable-Baselines3 (100% free):

```bash
# Train on easy task
python -m creditlens.rl.train_ppo --task easy --timesteps 200000

# Train on hard task
python -m creditlens.rl.train_ppo --task hard --timesteps 500000

# Hyperparameter search with Optuna
python -m creditlens.rl.train_ppo --task medium --hyperopt --trials 30

# Monitor training
tensorboard --logdir artifacts/tensorboard_logs
```

**PPO architecture**: 256 → 128 → 64 (ReLU) actor + critic, Discrete(4) action head

---

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/tasks` | GET | List all tasks and configs |
| `/reset` | POST | Start new episode → returns `LoanObservation` |
| `/step` | POST | Process action → returns `StepResult` |
| `/state` | GET | Get current `EpisodeState` |
| `/grade` | GET | Grade completed episode → returns scores 0.0–1.0 |
| `/metrics` | GET | Prometheus metrics |

### Example: Reset

```bash
curl -X POST http://localhost:8000/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "easy", "seed": 42}'
```

### Example: Step

```bash
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{
    "task_id": "easy",
    "action": {
      "action_type": "APPROVE",
      "applicant_id": "EP_000",
      "params": {"amount_fraction": 1.0},
      "reasoning": "FICO 720, low DTI, no fraud signals"
    }
  }'
```

### Example: Grade

```bash
curl "http://localhost:8000/grade?task_id=easy"
```

---

## Baseline Scores

Scores produced by the Ollama LLM agent (llama3.2) on seed=42:

| Task | Score | Key Metrics |
|------|-------|-------------|
| Easy | 0.72 | F1=0.78, completion=0.91 |
| Medium | 0.61 | ECL score=0.65, post-shock conservatism=0.55 |
| Hard | 0.44 | Fraud F1=0.50, portfolio=0.60, fairness=0.55 |

Rule-based fallback baseline (no LLM):

| Task | Score |
|------|-------|
| Easy | 0.58 |
| Medium | 0.42 |
| Hard | 0.31 |

PPO trained agent (200k steps):

| Task | Score |
|------|-------|
| Easy | 0.85 |
| Medium | 0.71 |
| Hard | 0.58 |

---

## Tech Stack

**Completely free. No paid APIs. No cloud services.**

| Layer | Tool | Version | Purpose |
|-------|------|---------|---------|
| Data gen | Faker | 24.x | Synthetic applicant profiles |
| Data gen | SDV | 1.12+ | Statistical correlation modeling |
| Feature eng | Pandas | 2.x | DTI, LTV, utilization transforms |
| Feature eng | scikit-learn | 1.4+ | Preprocessing pipelines |
| Imbalance | imbalanced-learn | 0.12+ | SMOTE oversampling |
| ML model | XGBoost | 2.x | Default probability prediction |
| Explainability | SHAP | 0.45+ | Feature attribution |
| Fraud detection | NetworkX | 3.x | Fraud ring graph analysis |
| Fairness | Fairlearn | 0.10+ | Demographic parity metrics |
| Storage | PyArrow/Parquet | 15.x | Fast columnar dataset |
| Storage | DuckDB | 0.10+ | SQL dev queries |
| RL training | Stable-Baselines3 | 2.3+ | PPO implementation |
| RL framework | Gymnasium | 0.29+ | OpenAI Gym interface |
| Deep learning | PyTorch (CPU) | 2.2+ | Neural network backend |
| Hyperopt | Optuna | 3.6+ | Hyperparameter search |
| LLM inference | Ollama | 0.2+ | Local free LLM (llama3.2/mistral) |
| API | FastAPI | 0.110+ | REST API server |
| API server | Uvicorn | 0.29+ | ASGI server |
| Monitoring | Prometheus-client | 0.20+ | Metrics collection |
| Experiment tracking | MLflow | 2.12+ | Model tracking |
| Logging | Loguru | 0.7+ | Structured logging |
| Testing | pytest | 8.x | Test suite |

---

## Project Structure

```
creditlens/
├── creditlens/
│   ├── __init__.py
│   ├── models.py              ← All Pydantic types (Observation, Action, Reward)
│   ├── env/
│   │   ├── engine.py          ← CreditLensEnv + CreditLensGymEnv (SB3 wrapper)
│   │   └── reward.py          ← Multi-objective reward engine
│   ├── data/
│   │   ├── generate.py        ← Synthetic data pipeline + XGBoost training
│   │   └── dataset.py         ← Parquet loader + episode sampler
│   ├── tasks/
│   │   └── graders.py         ← Easy/Medium/Hard graders (0.0–1.0)
│   ├── inference/
│   │   ├── service.py         ← FastAPI app (OpenEnv HTTP API)
│   │   └── agent.py           ← Ollama LLM agent
│   └── rl/
│       └── train_ppo.py       ← PPO training + Optuna hyperopt
├── inference.py               ← Main inference script (START/STEP/END logs)
├── openenv.yaml               ← OpenEnv metadata
├── pyproject.toml             ← Dependencies
├── .env.example               ← Environment variable template
├── docker/
│   ├── Dockerfile             ← Production container
│   └── compose.yml            ← API + Ollama together
├── tests/
│   └── test_env.py            ← pytest test suite
└── README.md
```

---

## License

MIT — free to use, modify, and distribute.
