# IBKR ML Trader

A Python web application that fetches market data from Interactive Brokers, calculates technical indicators, runs a Random Forest ML model to generate buy/sell signals, backtests the strategy, and supports paper trading via IB Gateway.

---

## Quick Start

### 1. Install dependencies

```bash
pip install flask pandas numpy yfinance pandas-ta scikit-learn joblib python-dotenv ib_insync
```

Or use the start script:
```bash
chmod +x start.sh
./start.sh
```

### 2. Start the app

```bash
python app.py
```

Then open **http://localhost:5000** in your browser.

---

## IB Gateway Setup (for live/paper data & trading)

1. Download **IB Gateway** from [interactivebrokers.com](https://www.interactivebrokers.com/en/trading/ibgateway-stable.php)
2. Log in with your **paper trading** credentials
3. Enable the API: `Configure → API → Settings → Enable ActiveX and Socket Clients`
4. Set **Socket port to 4002** (paper trading default)
5. Add `127.0.0.1` to trusted IP addresses
6. In the app, go to **Portfolio** tab → enter host `127.0.0.1`, port `4002`, click **Connect**

> **Without IB Gateway**, the app automatically falls back to **Yahoo Finance** for market data.
> Paper trading order placement requires an active IB Gateway connection.

---

## How to Use

### Charts & Data
1. Enter a stock symbol (e.g. `AAPL`, `TD`, `MSFT`)
2. Select a period and currency
3. Click **Fetch Data**
4. View the candlestick chart with overlaid indicators:
   - SMA 20 / 50 / 200
   - EMA 12 / 26
   - Bollinger Bands
5. Scroll down to see RSI, MACD, Stochastic, and ATR subplots
6. Toggle overlays with the buttons in the chart header

### ML Model
1. Go to the **ML Model** tab
2. Configure:
   - **Forward Days**: how many days ahead to predict (default 5)
   - **Return Threshold**: minimum % gain to classify as "BUY" (default 2%)
   - **Estimators**: number of Random Forest trees (default 200)
3. Click **Train Model**
4. View accuracy metrics, feature importances, and signal overlay

### Backtest
1. Go to the **Backtest** tab (train the model first)
2. Set initial capital and probability threshold
3. Click **Run Backtest**
4. View equity curve vs. buy-and-hold, drawdown, and trade log

### Paper Trading
1. Connect to IB Gateway (Portfolio tab)
2. Enter symbol, action (BUY/SELL), quantity
3. Click **Place Order**
4. Monitor positions and account summary

---

## Features

| Feature | Details |
|---|---|
| Data source | IB Gateway (primary) + Yahoo Finance (fallback) |
| Indicators | SMA, EMA, RSI, MACD, Bollinger Bands, ATR, Stochastic, Williams %R, CCI, ADX, OBV, Volume |
| ML Algorithm | Random Forest Classifier (scikit-learn) |
| Validation | Time-series cross-validation (no look-ahead bias) |
| Backtest | Vectorised backtest with transaction costs |
| Metrics | Sharpe, Sortino, Max Drawdown, Win Rate, Profit Factor |
| Trading | Paper order placement via ib_insync |
| UI | Plotly.js interactive charts, dark theme |

---

## File Structure

```
IBRK/
├── app.py            ← Flask backend (routes & API)
├── ibkr_client.py    ← IB Gateway connection & data
├── indicators.py     ← Technical indicator calculation
├── ml_model.py       ← Random Forest model
├── backtest.py       ← Backtesting engine
├── requirements.txt  ← Python dependencies
├── start.sh          ← Quick-start script
└── templates/
    └── index.html    ← Web UI (single-page app)
```

---

## Default Ports

| Connection | Port |
|---|---|
| IB Gateway Paper | 4002 |
| IB Gateway Live  | 4001 |
| TWS Paper        | 7497 |
| TWS Live         | 7496 |
| Flask App        | 5000 |
