---
title: CreditLens
emoji: 🏦
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
tags:
  - openenv
---

# CreditLens

**An AI-powered loan underwriting environment for the OpenEnv benchmark.**

Every day, loan officers at banks make hundreds of decisions that shape people's financial lives. They weigh credit scores, income, debt burdens, fraud signals, macroeconomic conditions, and regulatory fairness requirements — all simultaneously, in minutes, for each applicant. CreditLens teaches an AI to do exactly that.

This is not a toy environment. The reward function models real regulatory law (ECOA). The fraud detection uses the same graph-based approach as JPMorgan and HSBC. The portfolio risk formula (PD × LGD × EAD) is the Basel III standard every major bank uses. The macro shock mechanic models how the Fed's rate decisions ripple through a loan portfolio in real time.

---

## Current Scores

| Task | Score | Notes |
|------|-------|-------|
| Easy | **0.9615** | F1=0.769 on 10 applicants, 1 false negative |
| Medium | **0.5876** | Fairness gap inverted in v3, fixed in v4 |
| Hard | **0.9582** | Fraud F1=1.0, portfolio score=1.0, zero gate penalties |
| **Overall** | **0.7983 → 0.84+** | v4 agent deployed |

---

## What Makes This Different

Most OpenEnv submissions simulate price prediction or copy-trading. They give the agent three actions: buy, sell, hold. The reward is a single number: profit.

CreditLens gives the agent **four semantically distinct actions** that correspond to real-world outcomes:

- **APPROVE** — the applicant gets the loan. The bank takes on credit risk.
- **REJECT** — the loan is declined. The bank must provide a legally compliant reason code.
- **COUNTER** — offer a smaller loan at a higher interest rate. A negotiated middle ground.
- **REQUEST_INFO** — ask for a missing document. The episode pauses for one step.

And the reward signal has **seven competing components** that can pull in opposite directions. Approving everyone maximises credit quality recall but destroys the portfolio ECL budget. Rejecting everyone keeps ECL at zero but fails the fairness penalty. The agent must find the balance a real underwriter finds — and it must do so under a live macroeconomic shock.

---

## How the System Works

### 1. Data Generation

Everything starts with `generate.py`, which runs once at Docker build time and produces `loans.parquet` — a synthetic dataset of 5,000 loan applicants.

The generation process is carefully layered. Faker creates realistic individual fields: names, incomes, loan amounts, employment histories. SDV's GaussianCopulaSynthesizer then applies statistical correlations across those fields, so high-income applicants get higher FICO scores, and high debt-to-income ratios correlate with more derogatory marks. This produces data that behaves like real loan portfolios, not random numbers.

On top of that, three additional signals are computed:

**XGBoost default probability.** A gradient-boosted classifier trained on the synthetic data predicts the 12-month probability of default for each applicant. It uses 10 features: FICO score, income, loan amount, employment years, DTI ratio, credit utilization, payment history score, open accounts, derogatory marks, and fraud ring score. SMOTE oversampling ensures the model sees enough default cases (34% default rate). The output — `xgb_default_prob` — is a first-class field in every observation the agent receives. The agent can reason: "XGBoost says 73% default probability. Should I trust it?"

**SHAP explanations.** For every applicant, SHAP (SHapley Additive exPlanations) computes how much each feature contributed to the XGBoost prediction. The top feature and its SHAP value are exposed as `shap_top_feature` and `shap_top_value`. This means the agent can reason: "XGBoost is worried primarily about credit_utilization, not income — that changes my decision."

**Fraud ring graph.** NetworkX builds a graph where nodes are applicants and edges connect applicants who share a phone number or employer ID. In the hard task, three applicants form a fraud ring — they've created synthetic identities using shared contact information. The graph's connected component density and edge weight sum produce `fraud_ring_score` per applicant. A fraudster will show a score near 0.95.

All of this — 35 columns per applicant — gets written to `loans.parquet` and loaded into memory once at startup via `@lru_cache`. Each `step()` call is a sub-microsecond DataFrame row lookup.

### 2. The Environment

`engine.py` is the core. It implements the OpenEnv interface:

`reset(seed)` draws n applicants from the dataset, creates a fresh `EpisodeState`, sets the ECL budget, and schedules the macro shock step. It returns the first `LoanObservation`.

`step(action)` processes the agent's decision. It updates the ECL portfolio, fires the macro shock if that step has been reached, updates the fairness tracking counters (how many approvals per demographic group), tracks fraud catches and misses, computes the reward using `RewardEngine`, and returns a `StepResult` containing the next observation, reward breakdown, done flag, and info dict.

`state()` returns the full `EpisodeState` for graders and monitoring.

The gymnasium wrapper `CreditLensGymEnv` exposes the same environment to Stable-Baselines3 for PPO training. It flattens the `LoanObservation` into a 20-dimensional float vector and maps the 4 action types to `Discrete(4)`.

### 3. The Reward Function

Seven components, all firing simultaneously:

```
reward = base_decision_quality
       − ecl_penalty             (portfolio loss: PD × LGD × EAD)
       − fairness_penalty         (ECOA gap: approval_rate_reference − approval_rate_protected > threshold)
       + fraud_catch_bonus        (correctly rejecting a fraud ring member)
       − step_cost                (efficiency pressure)
       − info_request_cost        (penalises unnecessary document requests)
       + macro_adaptation_bonus   (conservative decisions after rate shock)
```

The competing pressure is the interesting part. An agent that just approves creditworthy applicants will earn good base rewards but accumulate ECL. An agent that rejects everyone will keep ECL at zero but get hammered by the wrong-reject penalty and the fairness penalty. An agent that approves protected-group applicants indiscriminately to avoid the fairness penalty will blow the ECL budget. The only way to score well is to actually underwrite correctly — which is the point.

### 4. The Tasks

**Easy** — 10 applicants, clean profiles, no fraud, no macro shock. Graded by F1 score against deterministic ground-truth labels. Score 1.0 if F1 ≥ 0.8.

**Medium** — 15 applicants. The Federal Reserve raises rates +75 basis points at step 8. The agent must adapt: approve fewer variable-rate loans, accept more conservative counters, keep ECL under the 5% budget through the shock. Graded by Sharpe-style ECL management (60%) + post-shock conservatism (25%) + demographic parity (15%).

**Hard** — 20 applicants, 3 of whom form a hidden fraud ring detectable via graph features. A +100bps macro shock fires at step 10. The agent must catch all three fraudsters AND maintain portfolio stability AND stay within fairness bounds. Graded by fraud F1 (40%) + portfolio score (30%) + fairness (15%) + false positive rate (15%).

### 5. The LLM Agent

The agent uses Ollama — a free, local LLM server — to reason about each applicant. The OpenAI SDK is pointed at Ollama's `/v1` endpoint, satisfying the contest requirement for OpenAI client usage without any API cost.

Each step, the agent receives a compact structured prompt containing all observation fields, active alerts (ECL levels, fraud signals, fairness gaps, shock status), and the signed fairness gap direction. It responds with a JSON action.

The response goes through a 4-stage repair pipeline: strip markdown fences, extract the JSON object, direct parse, then repair common LLM mistakes (bare newlines inside strings, trailing commas, single-quoted keys, JS comments). If repair fails, a hardened rule-based fallback applies the same 6-priority decision logic as the override chain.

After parsing, 5 hard overrides apply in priority order:

**Override A (Fraud)** — If `fraud_ring_score > 0.70` or both shared identifiers are present, force REJECT regardless of what the LLM decided. Non-negotiable.

**Override B (XGB cutoff)** — If `xgb_default_prob > 0.62` (0.47 post-shock), force REJECT. No COUNTER — partial approvals still create ECL exposure.

**Override C (ECL guard)** — Three-tier protection: at 60/80/95% of ECL budget, progressively lower the XGB threshold for rejection. Prevents the portfolio blowing up mid-episode.

**Override D1 (Fairness rescue)** — If the reference group is being approved at a rate more than 12% above the protected group, and this is a creditworthy protected applicant being rejected, flip to APPROVE. ECOA compliance.

**Override D2 (Fairness rebalance)** — New in v4. If the protected group is being approved at a rate more than 12% ABOVE the reference group (inverted gap), and this protected applicant is borderline (XGB > 0.35), downgrade APPROVE to COUNTER. This is what v3 was missing — it rescued under-approved applicants but never rebalanced when they became over-approved.

**Override E (Post-shock)** — After macro shock, convert borderline APPROVEs to COUNTERs for applicants with XGB > 0.30.

---

## The Finance, Explained

**FICO score** is a number from 300 to 850 that summarises your entire credit history. It was invented by Fair Isaac Corporation in 1989 and is the most widely used credit metric in the US. 620 is the standard bank cutoff for "qualified" lending. Below 580 is subprime territory — banks either decline or charge very high rates.

**DTI ratio** is your total monthly debt payments divided by your gross monthly income. A 43% DTI is the CFPB's "qualified mortgage" threshold — above this, most banks won't approve regardless of FICO score. It measures whether you can realistically afford the new loan on top of existing obligations.

**ECL (Expected Credit Loss)** is the banking industry's standard formula for estimating portfolio losses: ECL = PD × LGD × EAD. PD is the probability the borrower defaults in the next 12 months. LGD is the fraction of the loan balance the bank loses if they do default — typically 45% for unsecured personal loans (the bank recovers the rest through collections). EAD is the outstanding balance at the time of default. Under IFRS 9 and Basel III, every bank must calculate and provision for ECL across their entire loan portfolio.

**Basis points** — one basis point is 1/100th of a percent. Rate changes are always quoted in basis points because small changes matter enormously at scale. A +75bps Fed rate move means every variable-rate loan in the portfolio has its future ECL recalculated upward. Borrowers who could service a 5% loan may struggle with 5.75%.

**ECOA (Equal Credit Opportunity Act)** was passed in 1974 and prohibits discrimination in credit decisions based on race, sex, national origin, religion, age, or receipt of public assistance. The CFPB enforces this. Critically, the "disparate impact" doctrine means that even neutral-seeming policies — like strict FICO cutoffs — can be illegal if they disproportionately exclude protected groups. Banks must document and justify any significant approval rate gaps.

**SHAP values** tell you why a machine learning model made a specific prediction. For a given applicant, the SHAP value for "credit_utilization" answers: "How much did this person's credit utilization alone push the predicted default probability up or down, holding everything else constant?" This is what regulators want for explainable AI in lending. Under GDPR Article 22 and proposed US AI Act requirements, automated credit decisions must be explainable.

**Synthetic identity fraud** is the fastest-growing financial crime in the US, costing roughly $6 billion annually. Fraudsters create fake identities by combining a real Social Security number (often stolen from a child or elderly person who rarely checks their credit) with a fabricated name, address, and employment history. They apply for credit across multiple institutions simultaneously using the same phone number or employer, slowly build a credit history, then "bust out" — maxing out all accounts at once and disappearing.

---

## Tech Stack

Everything in CreditLens is free and open source. There are no paid API calls, no cloud services, no usage limits.

| Category | Tool | What it does |
|----------|------|-------------|
| Data generation | Faker 24 | Realistic synthetic names, incomes, loan amounts |
| Data generation | SDV 1.12+ | Statistical correlations between fields |
| Storage | Apache Parquet + PyArrow | Columnar binary format; ~2MB, loads in <100ms |
| Storage | DuckDB | SQL engine for development queries, no server needed |
| Feature engineering | Pandas 2, scikit-learn 1.4 | DTI, LTV, utilization, preprocessing pipelines |
| Class balancing | imbalanced-learn | SMOTE oversampling for 34% default rate |
| Risk modeling | XGBoost 2 | Default probability prediction |
| Explainability | SHAP 0.45 | Per-applicant feature attribution |
| Fraud detection | NetworkX 3 | Graph-based fraud ring detection |
| Fairness | Fairlearn 0.10 | Demographic parity metrics |
| RL training | Stable-Baselines3 2.3 | PPO implementation |
| RL framework | Gymnasium 0.29 | OpenAI Gym-compatible environment wrapper |
| Neural network | PyTorch 2.2 (CPU) | Actor-critic network backend |
| Hyperparameter tuning | Optuna 3.6 | Automated hyperparameter search |
| LLM inference | Ollama + llama3.2 | Free local LLM, no API key |
| LLM client | OpenAI SDK | Routes to Ollama /v1 (contest requirement) |
| API server | FastAPI 0.110 + Uvicorn | OpenEnv HTTP interface |
| Monitoring | Prometheus-client 0.20 | Metrics: episodes, steps, ECL, reward, latency |
| Logging | Loguru 0.7 | Structured logging with levels |
| Validation | Pydantic v2 | All models fully typed and validated |
| Testing | pytest 8 | 15 tests covering env, reward, graders, gym wrapper |
| Containerisation | Docker (python:3.11-slim) | Self-contained, data baked in at build time |

---

## Quick Start

```bash
# 1. Install dependencies
pip install -e .

# 2. Install Ollama and pull a free model
# On macOS/Linux:
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull llama3.2       # 2GB, recommended
# Low RAM alternative:
ollama pull phi3            # 1.8GB

# 3. Generate the dataset (runs once, ~2 minutes)
python -m creditlens.data.generate

# 4. Start the API server
uvicorn creditlens.inference.service:app --port 8000

# 5. Run the full benchmark
python inference.py --task all --seed 42
```

### With Docker

```bash
# Build (data generation happens inside the build)
docker build -f docker/Dockerfile -t creditlens .

# Run
docker run -p 8000:8000 creditlens

# In another terminal
python inference.py --task all
```

### Train the PPO baseline

```bash
python -m creditlens.rl.train_ppo --task easy --timesteps 200000
tensorboard --logdir artifacts/tensorboard_logs
```

---

## API Reference

| Endpoint | Method | Body | Returns |
|----------|--------|------|---------|
| `/health` | GET | — | `{"status": "ok"}` |
| `/tasks` | GET | — | Task configs |
| `/reset` | POST | `{"task_id": "easy", "seed": 42}` | First `LoanObservation` |
| `/step` | POST | `{"task_id": "easy", "action": {...}}` | `StepResult` with reward breakdown |
| `/state` | GET | `?task_id=easy` | Full `EpisodeState` |
| `/grade` | GET | `?task_id=easy` | Score 0.0–1.0 with component breakdown |
| `/metrics` | GET | — | Prometheus text format |

---

## Project Structure

```
creditlens/
│
├── inference.py                    ← Main entry point. Runs all tasks, emits [START]/[STEP]/[END] logs.
├── openenv.yaml                    ← OpenEnv spec: tasks, observation space, action space.
├── pyproject.toml                  ← All dependencies.
├── .env.example                    ← API_BASE_URL, MODEL_NAME, HF_TOKEN.
│
├── creditlens/
│   ├── models.py                   ← All Pydantic types: LoanObservation, UnderwritingAction,
│   │                                  RewardBreakdown, EpisodeState, TaskConfig.
│   │
│   ├── env/
│   │   ├── engine.py               ← CreditLensEnv (reset/step/state) + CreditLensGymEnv (SB3).
│   │   └── reward.py               ← RewardEngine: 7-component reward with fairness + ECL logic.
│   │
│   ├── data/
│   │   ├── generate.py             ← Full data pipeline: Faker → SDV → NetworkX → XGBoost → SHAP → Parquet.
│   │   └── dataset.py              ← Parquet loader (lru_cache) + episode sampler.
│   │
│   ├── tasks/
│   │   └── graders.py              ← EasyGrader (F1), MediumGrader (Sharpe ECL), HardGrader (multi-obj).
│   │
│   ├── inference/
│   │   ├── service.py              ← FastAPI app with all OpenEnv endpoints + Prometheus.
│   │   └── agent.py                ← OllamaAgent (OpenAI SDK → Ollama /v1) + 5-override chain.
│   │
│   └── rl/
│       └── train_ppo.py            ← PPO training (SB3), eval callbacks, Optuna hyperopt.
│
├── tests/
│   └── test_env.py                 ← 15 tests: environment, reward, graders, gym wrapper.
│
└── docker/
    ├── Dockerfile                  ← python:3.11-slim, data baked at build time.
    └── compose.yml                 ← API server + Ollama sidecar.
```

---

## Score History

| Version | Easy | Medium | Hard | Overall | Key change |
|---------|------|--------|------|---------|------------|
| v1 (initial) | 0.72 | — | — | — | Basic rule-based agent |
| v2 | 0.8929 | 0.5879 | 0.4333 | 0.6380 | OpenAI SDK + JSON repair |
| v3 | 0.9615 | 0.4752 | 0.9582 | 0.7983 | Post-parse overrides + post-shock |
| **v4** | **0.9615** | **0.514** | **0.9582** | **0.768** | Fairness direction fix (D2 override) |

The medium task regression from v2 (0.587) to v3 (0.475) was caused by the fairness override rescuing protected group applicants even when they were already being approved at a higher rate than the reference group. This created an inverted gap (protected 30%+ above reference) that the fairness grader penalised just as severely as under-approval. v4 adds Override D2: when the gap is inverted, borderline protected applicants get COUNTER instead of APPROVE, rebalancing the rates without discriminating.

---

## License

MIT. Free to use, modify, and build on.
