"""
Local CSV Data Cache  —  CAD-denominated stocks only
------------------------------------------------------
Persists downloaded OHLCV **and pre-computed technical indicators** to
  data/<SYMBOL>.csv
so repeated requests for the same period require no network call and no
indicator recalculation.

Cache logic
-----------
1.  Full hit   — CSV covers the full requested period and is current today
                 (within 1 trading day tolerance for weekends/holidays).
                 → Return cached rows directly, zero downloads.

2.  Stale tail — CSV exists but the last bar is older than yesterday.
                 → Download only  cache_end+1 → today  (OHLCV only),
                   combine all OHLCV, recalculate ALL indicators on the
                   full history, save enriched CSV, return filtered view.

3.  Missing head — Requested start is earlier than the first cached bar.
                   → Download only  requested_start → cache_start-1  (OHLCV),
                     same combine-and-recalculate flow as (2).

4.  Both gaps   — Handle (3) + (2) in one pass.

5.  No cache    — Download everything, compute indicators, save, return.

Force reload    — Delete the CSV before calling get_data() to trigger (5).

Indicator recalculation note
-----------------------------
Indicators like SMA-200 or ATR require a long history to produce accurate
values on recent bars.  Whenever new OHLCV rows are added (cases 2–5),
indicators are always recomputed on the **entire** combined OHLCV history
before saving, guaranteeing correctness at every row.

Currency
--------
All data is CAD-denominated.  The IBKR client is called with currency="CAD"
and yfinance symbols are resolved with the ".TO" (TSX) suffix automatically.
"""
from __future__ import annotations

import logging
import os
from datetime import date, timedelta

import numpy as np
import pandas as pd

from indicators import calculate_indicators

logger = logging.getLogger(__name__)

CACHE_DIR = os.path.join(os.path.dirname(__file__), "data")
CURRENCY  = "CAD"
OHLCV     = ["open", "high", "low", "close", "volume"]

# A sentinel column that tells us indicator columns are already present
_INDICATOR_SENTINEL = "rsi_14"

# Extra calendar days fetched *before* the requested start so that
# slow-to-converge indicators (SMA-200 needs ~290 trading days, EMA/ADX
# need further warm-up) are fully primed for the first visible bar.
# 400 calendar days ≈ 280 trading days — comfortably covers SMA-200.
_WARMUP_DAYS = 400


# ── Public API ───────────────────────────────────────────────────────────────

def get_data(
    symbol: str,
    duration: str,
    ibkr_client,
    use_fallback: bool = True,
) -> tuple[pd.DataFrame | None, str]:
    """
    Return  (df_with_indicators, source_description)  covering the full
    requested period, with all indicators fully primed from the first bar.

    Indicator warm-up strategy
    --------------------------
    Slow indicators like SMA-200 need ~200 trading bars before they produce
    valid values.  We always download _WARMUP_DAYS of extra history *before*
    the requested start, compute indicators on the combined history so every
    indicator is fully converged, then slice back to requested_start before
    returning.  The extended history is saved to cache so future lookups get
    the warm-up rows for free.

    Parameters
    ----------
    symbol       : ticker, e.g. "TD" or "TD.TO"
    duration     : period string  "1 M" | "3 M" | "6 M" | "1 Y" | "2 Y" …
    ibkr_client  : IBKRClient instance (may be disconnected)
    use_fallback : fall back to yfinance when IBKR is unavailable
    """
    requested_start = _duration_to_start_date(duration)
    today           = date.today()

    # Extend the download window backwards to prime all indicators.
    # The caller still gets data from requested_start; the extra rows
    # are stored in cache and used purely for indicator convergence.
    warmup_start = requested_start - timedelta(days=_WARMUP_DAYS)

    cached = _load(symbol)

    # ── Case 5: nothing cached ────────────────────────────────────────
    if cached is None or cached.empty:
        ohlcv, src = _download_ohlcv(
            symbol, warmup_start, today, ibkr_client, use_fallback,
            fallback_duration=duration,
        )
        if ohlcv is None or ohlcv.empty:
            return None, f"no data ({src})"
        df_full = calculate_indicators(ohlcv)
        _save(symbol, df_full)
        return df_full[df_full.index >= pd.Timestamp(requested_start)], \
               f"{src} (full download + {_WARMUP_DAYS}-day warmup, saved to cache)"

    # ── Determine what OHLCV history we already have ──────────────────
    cached_ohlcv  = _extract_ohlcv(cached)
    cache_start   = cached_ohlcv.index.min().date()
    cache_end     = cached_ohlcv.index.max().date()
    yesterday     = today - timedelta(days=1)

    ohlcv_pieces: list[pd.DataFrame] = []
    notes: list[str] = []

    # ── Case 3 / 4: cache doesn't reach back to warmup_start ─────────
    # Download whatever head is missing so indicators can be primed.
    if cache_start > warmup_start:
        earlier, src = _download_ohlcv(
            symbol, warmup_start, cache_start - timedelta(days=1),
            ibkr_client, use_fallback,
        )
        if earlier is not None and not earlier.empty:
            ohlcv_pieces.append(earlier)
            notes.append(f"prepended {len(earlier)} warmup rows ({src})")

    ohlcv_pieces.append(cached_ohlcv)

    # ── Case 2 / 4: cache is stale ────────────────────────────────────
    if cache_end < yesterday:
        newer, src = _download_ohlcv(
            symbol, cache_end + timedelta(days=1), today,
            ibkr_client, use_fallback,
        )
        if newer is not None and not newer.empty:
            ohlcv_pieces.append(newer)
            notes.append(f"appended {len(newer)} rows ({src})")

    # ── Case 1: pure cache hit (warmup already stored) ───────────────
    if len(ohlcv_pieces) == 1 and _has_indicators(cached):
        df_out = cached[cached.index >= pd.Timestamp(requested_start)]
        return df_out, "local cache (fully cached, no download)"

    # ── Combine OHLCV, recompute ALL indicators, save ─────────────────
    combined_ohlcv = pd.concat(ohlcv_pieces)
    combined_ohlcv = combined_ohlcv[~combined_ohlcv.index.duplicated(keep="last")]
    combined_ohlcv = combined_ohlcv.sort_index()

    df_full = calculate_indicators(combined_ohlcv)
    _save(symbol, df_full)

    if notes:
        source = "local cache + " + ", ".join(notes) + " (indicators recalculated)"
    else:
        source = "local cache (indicators recalculated from stored OHLCV)"

    # Return only the user-requested window — indicators are fully primed.
    df_out = df_full[df_full.index >= pd.Timestamp(requested_start)]
    return df_out, source


def cache_info(symbol: str) -> dict:
    """Return metadata about a cached symbol (surfaced in the UI)."""
    cached = _load(symbol)
    if cached is None or cached.empty:
        return {"cached": False}
    path = _cache_path(symbol)
    return {
        "cached":          True,
        "rows":            len(cached),
        "start":           cached.index.min().strftime("%Y-%m-%d"),
        "end":             cached.index.max().strftime("%Y-%m-%d"),
        "has_indicators":  _has_indicators(cached),
        "columns":         len(cached.columns),
        "file":            path,
        "size_kb":         round(os.path.getsize(path) / 1024, 1),
    }


def clear_cache(symbol: str) -> bool:
    """Delete the CSV cache for a symbol (both OHLCV and indicators)."""
    path = _cache_path(symbol)
    if os.path.exists(path):
        os.remove(path)
        logger.info("Cache cleared for %s (data + indicators)", symbol)
        return True
    return False


def list_cached_symbols() -> list[dict]:
    """Return cache metadata for every symbol that has a CSV on disk."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    return [
        cache_info(fname[:-4])
        for fname in sorted(os.listdir(CACHE_DIR))
        if fname.endswith(".csv")
    ]


# ── Internal: file I/O ───────────────────────────────────────────────────────

def _cache_path(symbol: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"{symbol.upper()}.csv")


def _load(symbol: str) -> pd.DataFrame | None:
    """Load the CSV and return a clean DataFrame (OHLCV + any indicator cols)."""
    path = _cache_path(symbol)
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index, utc=False)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df.index.name = "Date"
        df.columns    = [c.lower() for c in df.columns]
        df = df.dropna(how="all").sort_index()
        logger.info(
            "Loaded %d rows / %d cols for %s from cache",
            len(df), len(df.columns), symbol,
        )
        return df
    except Exception as exc:
        logger.warning("Failed to read cache for %s: %s", symbol, exc)
        return None


def _save(symbol: str, df: pd.DataFrame) -> None:
    """Save a full (OHLCV + indicators) DataFrame to CSV."""
    path = _cache_path(symbol)
    try:
        df_save = df.copy()
        df_save.index.name = "Date"
        if hasattr(df_save.index, "tz") and df_save.index.tz is not None:
            df_save.index = df_save.index.tz_localize(None)
        df_save = df_save.sort_index()
        df_save = df_save[~df_save.index.duplicated(keep="last")]
        df_save.to_csv(path)
        logger.info(
            "Saved %d rows / %d cols for %s → %s",
            len(df_save), len(df_save.columns), symbol, path,
        )
    except Exception as exc:
        logger.warning("Failed to save cache for %s: %s", symbol, exc)


# ── Internal: helpers ────────────────────────────────────────────────────────

def _has_indicators(df: pd.DataFrame) -> bool:
    """True if the DataFrame already contains pre-computed indicator columns."""
    return _INDICATOR_SENTINEL in df.columns


def _extract_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Return only the five raw OHLCV columns from an enriched DataFrame."""
    cols = [c for c in OHLCV if c in df.columns]
    return df[cols].copy()


def _duration_to_start_date(duration: str) -> date:
    """
    Convert an IBKR-style duration string to the equivalent start date.
    Supported: Y (years), M (months), W (weeks), D (days).
    """
    today  = date.today()
    parts  = duration.strip().split()
    try:
        amount = int(parts[0])
    except (ValueError, IndexError):
        return today - timedelta(days=730)

    unit = parts[1].upper() if len(parts) > 1 else "Y"

    if unit == "Y":
        try:
            return date(today.year - amount, today.month, today.day)
        except ValueError:
            return date(today.year - amount, today.month, 28)
    if unit == "M":
        month, year = today.month - amount, today.year
        while month <= 0:
            month += 12
            year  -= 1
        try:
            return date(year, month, today.day)
        except ValueError:
            return date(year, month, 28)
    if unit == "W":
        return today - timedelta(weeks=amount)
    if unit == "D":
        return today - timedelta(days=amount)

    return today - timedelta(days=730)


def _days_str(start: date, end: date) -> str:
    """IBKR durationStr covering [start, end] with a small buffer."""
    days = max((end - start).days + 5, 1)
    return f"{days} D"


def _resolve_yf_symbol(symbol: str) -> list[str]:
    """
    Return yfinance symbol candidates for a CAD/TSX ticker.
    Tries  <SYM>.TO  first (main TSX board), then  <SYM>.V  (TSXV),
    then the bare symbol as a last resort.
    """
    sym = symbol.upper()
    if "." in sym:
        return [sym]                         # already qualified
    return [f"{sym}.TO", f"{sym}.V", sym]    # TSX, TSXV, bare


# ── Internal: download ────────────────────────────────────────────────────────

def _download_ohlcv(
    symbol: str,
    start: date,
    end: date,
    ibkr_client,
    use_fallback: bool,
    fallback_duration: str | None = None,
) -> tuple[pd.DataFrame | None, str]:
    """
    Download raw OHLCV bars for [start, end] from IBKR or yfinance (CAD).
    Returns (ohlcv_df, source_label).
    """
    df  = None
    src = "none"

    # ── IBKR ─────────────────────────────────────────────────────────
    if ibkr_client.connected:
        try:
            duration_str = _days_str(start, end)
            end_dt       = end.strftime("%Y%m%d %H:%M:%S")
            df = ibkr_client.get_historical_data(
                symbol, duration_str, "1 day",
                currency=CURRENCY, end_datetime=end_dt,
            )
            if df is not None and not df.empty:
                src = "IBKR"
        except Exception as exc:
            logger.warning("IBKR download failed for %s: %s", symbol, exc)
            df = None

    # ── yfinance fallback ─────────────────────────────────────────────
    if (df is None or df.empty) and use_fallback:
        try:
            import yfinance as yf

            yf_start = start.strftime("%Y-%m-%d")
            yf_end   = (end + timedelta(days=1)).strftime("%Y-%m-%d")

            raw = pd.DataFrame()
            for yf_sym in _resolve_yf_symbol(symbol):
                raw = yf.Ticker(yf_sym).history(start=yf_start, end=yf_end)
                if not raw.empty:
                    logger.info("yfinance resolved %s → %s", symbol, yf_sym)
                    break

            # Period-based fallback for a first full download
            if raw.empty and fallback_duration:
                period_map = {
                    "1 M": "1mo", "3 M": "3mo", "6 M": "6mo",
                    "1 Y": "1y",  "2 Y": "2y",  "3 Y": "3y", "5 Y": "5y",
                }
                period = period_map.get(fallback_duration, "2y")
                for yf_sym in _resolve_yf_symbol(symbol):
                    raw = yf.Ticker(yf_sym).history(period=period)
                    if not raw.empty:
                        break

            if not raw.empty:
                raw.columns = [c.lower() for c in raw.columns]
                df  = raw[[c for c in OHLCV if c in raw.columns]].copy()
                src = "Yahoo Finance"

        except Exception as exc:
            logger.warning("yfinance download failed for %s: %s", symbol, exc)
            df = None

    # ── Clean up index ────────────────────────────────────────────────
    if df is not None and not df.empty:
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]
        if hasattr(df.index, "tz") and df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df.index.name = "Date"
        df = df.sort_index()

    return df, src
