# CreditLens — HuggingFace Space Dockerfile
# Serves FastAPI (OpenEnv REST API) + Gradio UI on port 7860

FROM python:3.11-slim

# System deps
RUN apt-get update && apt-get install -y \
    gcc g++ git curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir \
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
        uvicorn>=0.29.0 \
        pydantic>=2.6.0 \
        httpx>=0.27.0 \
        prometheus-client>=0.20.0 \
        python-dotenv>=1.0.0 \
        loguru>=0.7.0 \
        rich>=13.0.0 \
        joblib>=1.3.0 \
        openai>=1.0.0 \
        gradio>=4.0.0 \
        pytest>=8.0.0

# Copy source code
COPY . .

# Install package in editable mode
RUN pip install --no-cache-dir -e . || pip install --no-cache-dir \
        faker>=24.0.0 sdv>=1.12.0 pyarrow>=15.0.0 pandas>=2.0.0 \
        scikit-learn>=1.4.0 xgboost>=2.0.0 shap>=0.45.0 networkx>=3.0.0

# Generate dataset at build time (bakes loans.parquet into the image)
RUN python -m creditlens.data.generate

# HuggingFace Spaces requires port 7860
EXPOSE 7860

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
# Default env vars — override at runtime
ENV API_BASE_URL=http://localhost:7860
ENV MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
ENV PORT=7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Start the combined FastAPI+Gradio app
CMD ["python", "app.py"]