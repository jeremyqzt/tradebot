"""
IBKR ML Trading App – Flask Backend  (Python 3.11 compatible)
--------------------------------------
Start: python app.py
Then open: http://localhost:5000
"""
from __future__ import annotations

import json
import logging
import traceback

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

from backtest import Backtest
from data_cache import get_data, cache_info, clear_cache, list_cached_symbols
from ibkr_client import IBKRClient
from ml_model import TradingModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False

# ── Singletons ──────────────────────────────────────────────────────────────
ibkr    = IBKRClient()
model   = TradingModel()
backtest_engine = Backtest()


# ── JSON encoder that handles numpy/pandas types ────────────────────────────
class SafeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return None if np.isnan(obj) else float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.Timestamp):
            return obj.strftime("%Y-%m-%d")
        if isinstance(obj, float) and np.isnan(obj):
            return None
        return super().default(obj)


def safe_json(data, **kwargs):
    return app.response_class(
        response=json.dumps(data, cls=SafeEncoder),
        mimetype="application/json",
        **kwargs,
    )


def df_to_records(df: pd.DataFrame) -> list[dict]:
    """Convert DataFrame to JSON-safe list of dicts."""
    df = df.copy()
    # Reset index and normalise date column
    df = df.reset_index()
    # Rename 'Date' (capital D, from cache index) to 'date' so JS can find it
    # with a simple lowercase key lookup — avoids truthy-array short-circuit bug
    if "Date" in df.columns:
        df = df.rename(columns={"Date": "date"})
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.strftime("%Y-%m-%d")
    # Replace NaN/inf
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.where(pd.notna(df), None)
    return json.loads(df.to_json(orient="records"))


# ============================================================================
#  ROUTES – UI
# ============================================================================

@app.route("/")
def index():
    return render_template("index.html")


# ============================================================================
#  ROUTES – IBKR Connection
# ============================================================================

@app.route("/api/connect", methods=["POST"])
def connect():
    data = request.get_json() or {}
    result = ibkr.connect(
        host=data.get("host", "127.0.0.1"),
        port=int(data.get("port", 4002)),
        client_id=int(data.get("client_id", 1)),
    )
    return safe_json(result)


@app.route("/api/disconnect", methods=["POST"])
def disconnect():
    return safe_json(ibkr.disconnect())


@app.route("/api/status", methods=["GET"])
def status():
    return safe_json({
        "connected": ibkr.connected,
        "message":   "Connected to IB Gateway" if ibkr.connected else "Not connected",
        "model_trained": model.model is not None,
        "trained_symbols": model.list_trained_symbols(),
    })


# ============================================================================
#  ROUTES – Market Data
# ============================================================================

@app.route("/api/fetch-data", methods=["POST"])
def fetch_data():
    try:
        data         = request.get_json() or {}
        symbol       = data.get("symbol", "TD").strip().upper()
        duration     = data.get("duration", "2 Y")
        use_fallback = data.get("use_fallback", True)
        force_reload = data.get("force_reload", False)

        # ── Force reload: clear OHLCV + indicator cache ───────────────
        if force_reload:
            clear_cache(symbol)
            logger.info("Force reload: cleared cache for %s", symbol)

        # ── Smart cache layer ─────────────────────────────────────────
        # get_data returns a df that already has all indicator columns.
        # Indicators are recalculated and saved whenever new OHLCV rows
        # are downloaded; on a pure cache hit they are read straight
        # from the CSV — no recalculation needed.
        df, source = get_data(symbol, duration, ibkr, use_fallback)

        if df is None or df.empty:
            return safe_json({
                "success": False,
                "message": (
                    f"No data found for '{symbol}' (CAD). "
                    "For TSX stocks try the bare ticker, e.g. 'TD' or 'RY'. "
                    + ("" if ibkr.connected
                       else "(IBKR disconnected — enable Yahoo Finance fallback)")
                ),
            })

        # Register the indicator-enriched df with the ML model
        model.set_data(symbol, df)

        ci      = cache_info(symbol)
        records = df_to_records(df.tail(600))

        return safe_json({
            "success":    True,
            "symbol":     symbol,
            "currency":   "CAD",
            "source":     source,
            "rows":       len(df),
            "columns":    list(df.columns),
            "data":       records,
            "cache_info": ci,
        })

    except Exception as e:
        logger.error(traceback.format_exc())
        return safe_json({"success": False, "message": str(e)})


# ============================================================================
#  ROUTES – Cache Management
# ============================================================================

@app.route("/api/cache", methods=["GET"])
def get_cache_list():
    """Return metadata for all locally cached symbols."""
    try:
        return safe_json({"success": True, "cache": list_cached_symbols()})
    except Exception as e:
        return safe_json({"success": False, "message": str(e)})


@app.route("/api/cache/<symbol>", methods=["GET"])
def get_cache_symbol(symbol: str):
    """Return cache metadata for a single symbol."""
    try:
        return safe_json({"success": True, "cache_info": cache_info(symbol.upper())})
    except Exception as e:
        return safe_json({"success": False, "message": str(e)})


@app.route("/api/cache/<symbol>", methods=["DELETE"])
def delete_cache_symbol(symbol: str):
    """Delete the cached CSV for a symbol."""
    try:
        removed = clear_cache(symbol.upper())
        msg = f"Cache cleared for {symbol.upper()}" if removed else f"No cache found for {symbol.upper()}"
        return safe_json({"success": True, "message": msg})
    except Exception as e:
        return safe_json({"success": False, "message": str(e)})


# ============================================================================
#  ROUTES – ML Model
# ============================================================================

@app.route("/api/train", methods=["POST"])
def train():
    try:
        data = request.get_json() or {}
        symbol       = data.get("symbol", "AAPL").strip().upper()
        forward_days = int(data.get("forward_days", 5))
        threshold    = float(data.get("threshold", 0.02))
        n_estimators = int(data.get("n_estimators", 200))
        max_depth    = int(data.get("max_depth", 10))

        result = model.train(
            symbol, forward_days, threshold, n_estimators, max_depth
        )

        if result["success"]:
            # Attach signal data for the chart
            sig_df = model.get_signal_data(symbol)
            if sig_df is not None:
                sig_records = df_to_records(
                    sig_df[["close", "signal", "signal_prob"]].tail(600)
                )
                result["signal_data"] = sig_records

        return safe_json(result)

    except Exception as e:
        logger.error(traceback.format_exc())
        return safe_json({"success": False, "message": str(e)})


@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        data   = request.get_json() or {}
        symbol = data.get("symbol", "AAPL").strip().upper()
        return safe_json(model.predict(symbol))
    except Exception as e:
        return safe_json({"success": False, "message": str(e)})


# ============================================================================
#  ROUTES – Backtest
# ============================================================================

@app.route("/api/backtest", methods=["POST"])
def run_backtest():
    try:
        data = request.get_json() or {}
        symbol           = data.get("symbol", "AAPL").strip().upper()
        initial_capital = float(data.get("initial_capital", 10_000))
        prob_threshold  = float(data.get("prob_threshold", 0.55))
        commission      = float(data.get("commission", 1.0))    # flat $ per trade
        max_short       = float(data.get("max_short", 1_000.0)) # max $ short exposure

        sig_df = model.get_signal_data(symbol)
        if sig_df is None:
            return safe_json({
                "success": False,
                "message": "No signal data. Please train the model first.",
            })

        result = backtest_engine.run(
            sig_df, initial_capital, "signal", prob_threshold, commission, max_short
        )
        result["success"] = True
        return safe_json(result)

    except Exception as e:
        logger.error(traceback.format_exc())
        return safe_json({"success": False, "message": str(e)})


# ============================================================================
#  ROUTES – Paper Trading & Portfolio
# ============================================================================

@app.route("/api/paper-trade", methods=["POST"])
def paper_trade():
    try:
        data        = request.get_json() or {}
        symbol      = data.get("symbol", "TD").strip().upper()
        action      = data.get("action", "BUY").upper()
        quantity    = int(data.get("quantity", 10))
        order_type  = data.get("order_type", "MKT").upper()
        limit_price = data.get("limit_price")

        result = ibkr.place_order(symbol, action, quantity, order_type, limit_price, "CAD")
        return safe_json(result)
    except Exception as e:
        return safe_json({"success": False, "message": str(e)})


@app.route("/api/portfolio", methods=["GET"])
def portfolio():
    try:
        return safe_json({
            "success":   True,
            "portfolio": ibkr.get_portfolio(),
            "account":   ibkr.get_account_summary(),
            "orders":    ibkr.get_open_orders(),
        })
    except Exception as e:
        return safe_json({"success": False, "message": str(e)})


@app.route("/api/save-model", methods=["POST"])
def save_model():
    try:
        data = request.get_json() or {}
        path = data.get("path", "trading_model.joblib")
        ok   = model.save(path)
        return safe_json({"success": ok, "path": path if ok else None})
    except Exception as e:
        return safe_json({"success": False, "message": str(e)})


@app.route("/api/load-model", methods=["POST"])
def load_model():
    try:
        data = request.get_json() or {}
        path = data.get("path", "trading_model.joblib")
        ok   = model.load(path)
        return safe_json({"success": ok, "message": "Model loaded" if ok else "File not found"})
    except Exception as e:
        return safe_json({"success": False, "message": str(e)})


# ============================================================================
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)
