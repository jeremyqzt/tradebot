"""
Technical Indicators  (pure pandas / numpy — no external TA library)
----------------------------------------------------------------------
Compatible with Python 3.11 and pandas 2.x.
All indicators are computed from scratch so there are no third-party
library compatibility issues.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Accepts a DataFrame with columns: open, high, low, close, volume.
    Returns the same DataFrame enriched with indicator columns.
    All source columns are lower-cased automatically.
    """
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]

    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    _add_moving_averages(df)   # required for MACD (EMA-12 / EMA-26)
    _add_macd(df)
    _add_rsi(df)
    _add_bollinger(df)

    return df


# ── Moving Averages ──────────────────────────────────────────────────────────

def _add_moving_averages(df: pd.DataFrame) -> None:
    close = df["close"]

    # Simple Moving Averages
    for n in [20, 50, 200]:
        df[f"sma_{n}"] = close.rolling(n, min_periods=n).mean()

    # Exponential Moving Averages
    for n in [9, 12, 26]:
        df[f"ema_{n}"] = close.ewm(span=n, adjust=False).mean()


# ── MACD ─────────────────────────────────────────────────────────────────────

def _add_macd(df: pd.DataFrame) -> None:
    close = df["close"]
    df["macd"]        = df["ema_12"] - df["ema_26"]
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"]   = df["macd"] - df["macd_signal"]


# ── RSI (Wilder / RMA smoothing) ─────────────────────────────────────────────

def _add_rsi(df: pd.DataFrame, length: int = 14) -> None:
    delta  = df["close"].diff()
    gain   = delta.clip(lower=0)
    loss   = (-delta.clip(upper=0))

    # Wilder smoothing (equivalent to EMA with alpha = 1/length)
    avg_gain = gain.ewm(alpha=1 / length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / length, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi_14"] = 100 - (100 / (1 + rs))


# ── Bollinger Bands ───────────────────────────────────────────────────────────

def _add_bollinger(df: pd.DataFrame, length: int = 20, std: float = 2.0) -> None:
    close     = df["close"]
    bb_mid    = close.rolling(length, min_periods=length).mean()
    bb_std    = close.rolling(length, min_periods=length).std(ddof=0)

    df["bb_upper"]  = bb_mid + std * bb_std
    df["bb_middle"] = bb_mid
    df["bb_lower"]  = bb_mid - std * bb_std

    band_width = df["bb_upper"] - df["bb_lower"]
    df["bb_width"] = band_width / bb_mid.replace(0, np.nan)
    df["bb_pct"]   = (close - df["bb_lower"]) / band_width.replace(0, np.nan)


# ── Average True Range ────────────────────────────────────────────────────────

def _add_atr(df: pd.DataFrame, length: int = 14) -> None:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low  - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    # Wilder smoothing
    df["atr_14"] = tr.ewm(alpha=1 / length, adjust=False).mean()


# ── Stochastic Oscillator ─────────────────────────────────────────────────────

def _add_stochastic(df: pd.DataFrame, k: int = 14, d: int = 3) -> None:
    high, low, close = df["high"], df["low"], df["close"]
    low_k  = low.rolling(k,  min_periods=k).min()
    high_k = high.rolling(k, min_periods=k).max()

    stoch_k        = 100 * (close - low_k) / (high_k - low_k).replace(0, np.nan)
    df["stoch_k"]  = stoch_k
    df["stoch_d"]  = stoch_k.rolling(d, min_periods=d).mean()


# ── Williams %R ───────────────────────────────────────────────────────────────

def _add_williams_r(df: pd.DataFrame, length: int = 14) -> None:
    high, low, close = df["high"], df["low"], df["close"]
    high_n = high.rolling(length, min_periods=length).max()
    low_n  = low.rolling(length,  min_periods=length).min()

    df["willr_14"] = -100 * (high_n - close) / (high_n - low_n).replace(0, np.nan)


# ── Commodity Channel Index ───────────────────────────────────────────────────

def _add_cci(df: pd.DataFrame, length: int = 20) -> None:
    tp   = (df["high"] + df["low"] + df["close"]) / 3
    sma  = tp.rolling(length, min_periods=length).mean()
    mad  = tp.rolling(length, min_periods=length).apply(
        lambda x: np.mean(np.abs(x - x.mean())), raw=True
    )
    df["cci_20"] = (tp - sma) / (0.015 * mad.replace(0, np.nan))


# ── Average Directional Index ─────────────────────────────────────────────────

def _add_adx(df: pd.DataFrame, length: int = 14) -> None:
    high, low, close = df["high"], df["low"], df["close"]
    prev_high  = high.shift(1)
    prev_low   = low.shift(1)

    plus_dm  = np.where((high - prev_high) > (prev_low - low),
                         np.maximum(high - prev_high, 0), 0)
    minus_dm = np.where((prev_low - low) > (high - prev_high),
                         np.maximum(prev_low - low, 0), 0)

    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low  - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    plus_dm_s  = pd.Series(plus_dm,  index=df.index).ewm(alpha=1/length, adjust=False).mean()
    minus_dm_s = pd.Series(minus_dm, index=df.index).ewm(alpha=1/length, adjust=False).mean()
    tr_s       = tr.ewm(alpha=1/length, adjust=False).mean()

    plus_di  = 100 * plus_dm_s  / tr_s.replace(0, np.nan)
    minus_di = 100 * minus_dm_s / tr_s.replace(0, np.nan)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    df["adx_14"] = dx.ewm(alpha=1/length, adjust=False).mean()


# ── Volume Indicators ─────────────────────────────────────────────────────────

def _add_volume_indicators(df: pd.DataFrame) -> None:
    vol   = df["volume"]
    close = df["close"]

    df["vol_sma_20"] = vol.rolling(20, min_periods=1).mean()
    df["vol_ratio"]  = vol / df["vol_sma_20"].replace(0, np.nan)

    # On-Balance Volume
    direction = np.sign(close.diff()).fillna(0)
    df["obv"] = (direction * vol).cumsum()


# ── Returns ───────────────────────────────────────────────────────────────────

def _add_returns(df: pd.DataFrame) -> None:
    close = df["close"]
    for n in [1, 5, 10, 20]:
        df[f"return_{n}d"] = close.pct_change(n)


# ── Price Ratios ──────────────────────────────────────────────────────────────

def _add_price_ratios(df: pd.DataFrame) -> None:
    close = df["close"]
    df["close_sma20_ratio"] = close / df["sma_20"].replace(0, np.nan)
    df["close_sma50_ratio"] = close / df["sma_50"].replace(0, np.nan)
    df["sma20_sma50_ratio"] = df["sma_20"] / df["sma_50"].replace(0, np.nan)


# ── Historical Volatility ─────────────────────────────────────────────────────

def _add_volatility(df: pd.DataFrame) -> None:
    df["volatility_20"] = (
        df["return_1d"].rolling(20, min_periods=10).std() * np.sqrt(252)
    )


# ── 5-day Swing High / Low ────────────────────────────────────────────────────

def _add_swing_levels(df: pd.DataFrame) -> None:
    df["high_5d"] = df["high"].rolling(5, min_periods=1).max()
    df["low_5d"]  = df["low"].rolling(5,  min_periods=1).min()
