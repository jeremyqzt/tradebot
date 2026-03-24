#!/bin/bash
# ─────────────────────────────────────────────────────────────
#  IBKR ML Trader – Quick Start Script
# ─────────────────────────────────────────────────────────────
set -e

echo ""
echo "  ╔══════════════════════════════════╗"
echo "  ║     IBKR ML Trader  v1.0         ║"
echo "  ╚══════════════════════════════════╝"
echo ""

# 1. Install dependencies (skip if already installed)
echo "▸ Installing Python dependencies..."
pip install flask pandas numpy yfinance pandas-ta scikit-learn joblib python-dotenv ib_insync 2>&1 | grep -E "(Successfully|already|ERROR)" || true

echo ""
echo "▸ Starting Flask server on http://localhost:5000 ..."
echo "  (Press Ctrl+C to stop)"
echo ""

# 2. Launch
python3 app.py
