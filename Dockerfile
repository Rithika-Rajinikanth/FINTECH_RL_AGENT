# CreditLens — HuggingFace Space Dockerfile
# Serves FastAPI (OpenEnv REST) + Gradio UI on port 7860
#
# KEY FIX: Gradio is mounted at /ui (not /) in app.py
# This ensures POST /reset is handled by FastAPI → returns HTTP 200
# Previous: Gradio at / intercepted POST /reset → 405 Method Not Allowed

FROM python:3.11-slim

# System dependencies
RUN apt-get update && apt-get install -y \
    gcc g++ git curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install PyTorch CPU first (large layer — cache separately)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        torch --index-url https://download.pytorch.org/whl/cpu

# Install all other dependencies
RUN pip install --no-cache-dir \
    faker>=24.0.0 \
    sdv>=1.12.0 \
    duckdb>=0.10.0 \
    pyarrow>=15.0.0 \
    pandas>=2.0.0 \
    scikit-learn>=1.4.0 \
    xgboost>=2.0.0 \
    shap>=0.45.0 \
    networkx>=3.0.0 \
    fairlearn>=0.10.0 \
    imbalanced-learn>=0.12.0 \
    stable-baselines3>=2.3.0 \
    gymnasium>=0.29.0 \
    optuna>=3.6.0 \
    fastapi>=0.110.0 \
    "uvicorn[standard]>=0.29.0" \
    pydantic>=2.6.0 \
    httpx>=0.27.0 \
    openai>=1.0.0 \
    gradio>=4.0.0 \
    prometheus-client>=0.20.0 \
    python-dotenv>=1.0.0 \
    loguru>=0.7.0 \
    rich>=13.0.0 \
    joblib>=1.3.0 \
    pytest>=8.0.0

# Copy source code
COPY . .

# Install the creditlens package itself
RUN pip install --no-cache-dir -e . 2>/dev/null || \
    pip install --no-cache-dir \
        faker>=24.0.0 sdv>=1.12.0 pyarrow>=15.0.0 pandas>=2.0.0 \
        scikit-learn>=1.4.0 xgboost>=2.0.0 shap>=0.45.0 networkx>=3.0.0

# Generate the synthetic dataset and train XGBoost at build time
# This bakes loans.parquet into the image so startup is instant
RUN python -m creditlens.data.generate

# HuggingFace Spaces requires port 7860
EXPOSE 7860

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
# Contest-required environment variables (override at runtime)
ENV API_BASE_URL=http://localhost:7860
ENV MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
ENV LLM_BASE_URL=https://router.huggingface.co/v1
ENV PORT=7860

# Health check — validates both /health (FastAPI) and /ui (Gradio)
HEALTHCHECK --interval=30s --timeout=15s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Entry point: combined FastAPI + Gradio server
# FastAPI handles: GET /health, POST /reset, POST /step, GET /state, GET /grade
# Gradio UI at: /ui
CMD ["python", "app.py"]