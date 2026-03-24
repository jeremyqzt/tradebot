"""
trade_db.py – SQLite persistence layer for executed trades.
-----------------------------------------------------------
Database file: data/trades.db  (same directory as CSV cache)
"""
from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any

# ── Path ─────────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(_HERE, "data")
DB_PATH = os.path.join(DB_DIR, "trades.db")

os.makedirs(DB_DIR, exist_ok=True)

# ── Schema ────────────────────────────────────────────────────────────────────
_SCHEMA = """
CREATE TABLE IF NOT EXISTS trades (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   TEXT    NOT NULL,          -- ISO-8601, UTC
    symbol      TEXT    NOT NULL,
    action      TEXT    NOT NULL,          -- BUY | SELL
    quantity    INTEGER NOT NULL,
    order_type  TEXT    NOT NULL,          -- MKT | LMT
    limit_price REAL,                      -- NULL for market orders
    currency    TEXT    NOT NULL DEFAULT 'CAD',
    status      TEXT    NOT NULL DEFAULT 'submitted', -- submitted | filled | rejected
    order_id    TEXT,                      -- broker order ID (may be NULL in demo)
    fill_price  REAL,                      -- actual execution price if known
    notes       TEXT                       -- free-form remarks
);

CREATE INDEX IF NOT EXISTS idx_trades_symbol    ON trades(symbol);
CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp);
"""


# ── Connection helper ─────────────────────────────────────────────────────────
@contextmanager
def _conn():
    """Yield a thread-safe, autocommitting connection."""
    con = sqlite3.connect(DB_PATH, check_same_thread=False)
    con.row_factory = sqlite3.Row
    try:
        yield con
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()


# ── Init ──────────────────────────────────────────────────────────────────────
def init_db() -> None:
    """Create tables / indexes if they don't exist yet."""
    with _conn() as con:
        con.executescript(_SCHEMA)


# ── Write ─────────────────────────────────────────────────────────────────────
def record_trade(
    symbol: str,
    action: str,
    quantity: int,
    order_type: str,
    currency: str = "CAD",
    limit_price: float | None = None,
    status: str = "submitted",
    order_id: str | None = None,
    fill_price: float | None = None,
    notes: str | None = None,
) -> int:
    """Insert one trade row and return its new ``id``."""
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with _conn() as con:
        cur = con.execute(
            """
            INSERT INTO trades
              (timestamp, symbol, action, quantity, order_type, limit_price,
               currency, status, order_id, fill_price, notes)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)
            """,
            (ts, symbol.upper(), action.upper(), quantity,
             order_type.upper(), limit_price,
             currency.upper(), status, order_id, fill_price, notes),
        )
        return cur.lastrowid


def update_trade_status(
    trade_id: int,
    status: str,
    fill_price: float | None = None,
) -> bool:
    """Update status (and optionally fill price) of an existing trade."""
    with _conn() as con:
        cur = con.execute(
            "UPDATE trades SET status=?, fill_price=COALESCE(?,fill_price) WHERE id=?",
            (status, fill_price, trade_id),
        )
        return cur.rowcount > 0


# ── Read ──────────────────────────────────────────────────────────────────────
def get_trades(
    symbol: str | None = None,
    action: str | None = None,
    limit: int = 500,
    offset: int = 0,
) -> list[dict]:
    """Return trades as a list of dicts, newest first."""
    clauses: list[str] = []
    params: list[Any] = []
    if symbol:
        clauses.append("symbol = ?")
        params.append(symbol.upper())
    if action:
        clauses.append("action = ?")
        params.append(action.upper())
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    params += [limit, offset]
    with _conn() as con:
        rows = con.execute(
            f"SELECT * FROM trades {where} ORDER BY timestamp DESC LIMIT ? OFFSET ?",
            params,
        ).fetchall()
    return [dict(r) for r in rows]


def get_trade_stats() -> dict:
    """Aggregate statistics across all trades."""
    with _conn() as con:
        total = con.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
        if total == 0:
            return {
                "total": 0,
                "buy_count": 0,
                "sell_count": 0,
                "symbols": [],
                "by_symbol": [],
                "by_day": [],
                "by_action": [],
            }

        buy_count  = con.execute("SELECT COUNT(*) FROM trades WHERE action='BUY'").fetchone()[0]
        sell_count = con.execute("SELECT COUNT(*) FROM trades WHERE action='SELL'").fetchone()[0]

        symbols = [
            r[0] for r in con.execute(
                "SELECT DISTINCT symbol FROM trades ORDER BY symbol"
            ).fetchall()
        ]

        by_symbol = [
            dict(r) for r in con.execute(
                """SELECT symbol,
                          COUNT(*) AS total,
                          SUM(CASE WHEN action='BUY'  THEN 1 ELSE 0 END) AS buys,
                          SUM(CASE WHEN action='SELL' THEN 1 ELSE 0 END) AS sells,
                          SUM(quantity) AS total_qty
                   FROM trades
                   GROUP BY symbol
                   ORDER BY total DESC"""
            ).fetchall()
        ]

        by_day = [
            dict(r) for r in con.execute(
                """SELECT substr(timestamp,1,10) AS day,
                          COUNT(*) AS count
                   FROM trades
                   GROUP BY day
                   ORDER BY day"""
            ).fetchall()
        ]

        by_action = [
            dict(r) for r in con.execute(
                """SELECT action, COUNT(*) AS count, SUM(quantity) AS total_qty
                   FROM trades
                   GROUP BY action"""
            ).fetchall()
        ]

    return {
        "total":      total,
        "buy_count":  buy_count,
        "sell_count": sell_count,
        "symbols":    symbols,
        "by_symbol":  by_symbol,
        "by_day":     by_day,
        "by_action":  by_action,
    }


def delete_trade(trade_id: int) -> bool:
    """Delete a single trade by id. Returns True if a row was removed."""
    with _conn() as con:
        cur = con.execute("DELETE FROM trades WHERE id=?", (trade_id,))
        return cur.rowcount > 0


# ── Module self-init ──────────────────────────────────────────────────────────
init_db()
