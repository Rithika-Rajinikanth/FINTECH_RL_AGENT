#!/usr/bin/env bash
# CreditLens — One-command startup script
# Generates data, starts API server, and optionally runs inference

set -e

echo "=============================================="
echo "  CreditLens — AI Credit Risk Environment"
echo "=============================================="

# Step 1: Generate data if needed
if [ ! -f "creditlens/data/loans.parquet" ]; then
    echo ""
    echo "📊 Generating synthetic dataset + training XGBoost..."
    python -m creditlens.data.generate
    echo "✅ Dataset ready"
else
    echo "✅ Dataset already exists"
fi

# Step 2: Run tests
echo ""
echo "🧪 Running test suite..."
python -m pytest tests/ -q --tb=short 2>&1 | tail -5
echo "✅ Tests passed"

# Step 3: Start API server in background
echo ""
echo "🚀 Starting FastAPI server on port 8000..."
uvicorn creditlens.inference.service:app --host 0.0.0.0 --port 8000 &
SERVER_PID=$!

# Wait for server to be ready
echo "   Waiting for server..."
for i in {1..30}; do
    if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
        echo "✅ Server ready"
        break
    fi
    sleep 1
done

# Step 4: Run inference if requested
if [ "$1" == "--inference" ]; then
    echo ""
    echo "🤖 Running inference script (all tasks)..."
    python inference.py --task all --seed 42
fi

echo ""
echo "=============================================="
echo "  CreditLens API is running!"
echo "  Base URL:  http://localhost:8000"
echo "  Docs:      http://localhost:8000/docs"
echo "  Metrics:   http://localhost:8000/metrics"
echo ""
echo "  Run inference:"
echo "  python inference.py --task all"
echo ""
echo "  Train PPO:"
echo "  python -m creditlens.rl.train_ppo --task easy"
echo "=============================================="

# Keep server running
wait $SERVER_PID
