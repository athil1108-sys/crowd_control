#!/bin/bash
# ──────────────────────────────────────────────
# CrowdAI — Quick Start Script
# ──────────────────────────────────────────────
set -e

cd "$(dirname "$0")"

echo "🏟️  CrowdAI — Privacy-First Crowd Management System"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Activate venv
if [ -d "venv" ]; then
    echo "✅ Activating virtual environment..."
    source venv/Scripts/activate
else
    echo "⚠️  No venv found. Creating one..."
    python -m venv venv
    source venv/Scripts/activate
    pip install -r requirements.txt
fi

# Train model if not exists
if [ ! -f "models/congestion_model.pkl" ]; then
    echo ""
    echo "🧠 Training ML model (first run only)..."
    python -m src.model
fi

echo ""
echo "🚀 Launching FastAPI dashboard..."
echo "   Open http://localhost:8000 in your browser"
echo ""

uvicorn fastapi_app:app --host 127.0.0.1 --port 8000 --reload
