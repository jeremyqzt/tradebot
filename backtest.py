"""
Vectorised Backtesting Engine  (Python 3.11 compatible)
-------------------------------
Runs a long / short strategy driven by Random Forest signals.
Compares strategy returns vs. buy-and-hold.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class Backtest:

    # ------------------------------------------------------------------ #
    #  Main entry point
    # ------------------------------------------------------------------ #

    def run(
        self,
        df: pd.DataFrame,
        initial_capital: float = 10_000.0,
        signal_col: str = "signal",
        prob_threshold: float = 0.55,
        commission: float = 1.0,    # flat dollar amount per trade (entry or exit)
        max_short: float = 1_000.0, # maximum dollar value of any short position
    ) -> dict:
        """
        Parameters
        ----------
        df               : DataFrame with 'close', signal_col, optionally 'signal_prob'
        initial_capital  : Starting portfolio value
        signal_col       : Column name holding -1/0/1 signals
        prob_threshold   : Minimum prediction probability to enter a trade
        commission       : Flat dollar commission charged on each entry or exit (0.01–20)
        max_short        : Maximum dollar amount that can be shorted at any one time
        """
        commission = float(np.clip(commission, 0.01, 20.0))
        max_short  = float(max(max_short, 0.0))

        df = df.copy()

        required = {"close", signal_col}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame missing columns: {missing}")

        df = df.dropna(subset=["close", signal_col])

        # ── Apply probability gate ───────────────────────────────────
        if "signal_prob" in df.columns:
            df["position"] = np.where(
                (df[signal_col] == 1)  & (df["signal_prob"] >= prob_threshold),  1,
                np.where(
                (df[signal_col] == -1) & (df["signal_prob"] >= prob_threshold), -1, 0
                )
            )
        else:
            df["position"] = df[signal_col].astype(int)

        # ── Daily price returns ──────────────────────────────────────
        df["price_return"] = df["close"].pct_change().fillna(0)

        # ── Strategy equity curve ────────────────────────────────────
        # Long  (pos=1) : fully invested — return = price_return
        # Short (pos=-1): capped at max_short — return = -price_return * (short_size / equity)
        # Flat  (pos=0) : no market exposure
        positions    = df["position"].tolist()
        price_rets   = df["price_return"].tolist()
        trade_events = df["position"].diff().abs().fillna(0).tolist()

        equity     = initial_capital
        equities   = []
        strat_rets = []
        prev_pos   = 0

        for pos, pret, is_trade in zip(positions, price_rets, trade_events):
            equity_before = equity
            if prev_pos == 1:
                equity = equity * (1 + pret)
            elif prev_pos == -1:
                short_size = min(max_short, max(equity, 0.0))
                equity = equity - short_size * pret   # profit when price falls
            # prev_pos == 0: cash, no change
            if is_trade:
                equity -= commission
            strat_rets.append((equity - equity_before) / max(equity_before, 1e-10))
            equities.append(equity)
            prev_pos = pos

        df["strategy_return"] = strat_rets
        df["equity_strategy"] = equities

        # ── Buy-and-hold equity curve ────────────────────────────────
        df["equity_bah"] = (1 + df["price_return"]).cumprod() * initial_capital

        # ── Build trade log ──────────────────────────────────────────
        trade_log = self._build_trade_log(df)

        # ── Date list ────────────────────────────────────────────────
        dates = self._extract_dates(df)

        return {
            "dates":            dates,
            "price":            df["close"].round(4).tolist(),
            "equity_strategy":  df["equity_strategy"].round(2).tolist(),
            "equity_bah":       df["equity_bah"].round(2).tolist(),
            "position":         df["position"].tolist(),
            "signals":          df[signal_col].tolist(),
            "strategy_metrics": self._metrics(df["strategy_return"], df["equity_strategy"]),
            "bah_metrics":      self._metrics(df["price_return"],    df["equity_bah"]),
            "trade_log":        trade_log,
            "config": {
                "initial_capital": initial_capital,
                "prob_threshold":  prob_threshold,
                "commission":      commission,
                "max_short":       max_short,
            },
        }

    # ------------------------------------------------------------------ #
    #  Metrics
    # ------------------------------------------------------------------ #

    def _metrics(self, returns: pd.Series, equity: pd.Series) -> dict:
        r = returns.dropna()
        trading_days = 252
        n = len(r)

        if n < 2:
            return {}

        total_return = float(equity.iloc[-1] / equity.iloc[0] - 1)
        ann_return   = float((1 + total_return) ** (trading_days / n) - 1)
        ann_vol      = float(r.std() * np.sqrt(trading_days))

        sharpe = float(
            r.mean() * trading_days / (r.std() * np.sqrt(trading_days) + 1e-10)
        )

        # Sortino (downside deviation)
        downside = r[r < 0].std()
        sortino = float(r.mean() * trading_days / (downside * np.sqrt(trading_days) + 1e-10))

        # Max drawdown
        cummax   = equity.cummax()
        drawdown = (equity - cummax) / cummax
        max_dd   = float(drawdown.min())

        # Win rate
        wins   = int((r > 0).sum())
        losses = int((r < 0).sum())
        win_rate = float(wins / (wins + losses + 1e-10))

        # Profit factor
        gross_profit  = float(r[r > 0].sum())
        gross_loss    = float(abs(r[r < 0].sum()))
        profit_factor = float(gross_profit / (gross_loss + 1e-10))

        return {
            "total_return":        round(total_return * 100, 2),
            "annualized_return":   round(ann_return    * 100, 2),
            "annualized_vol":      round(ann_vol       * 100, 2),
            "sharpe_ratio":        round(sharpe,  3),
            "sortino_ratio":       round(sortino, 3),
            "max_drawdown":        round(max_dd   * 100, 2),
            "win_rate":            round(win_rate * 100, 2),
            "profit_factor":       round(profit_factor, 3),
            "total_trades":        wins + losses,
            "winning_trades":      wins,
            "losing_trades":       losses,
            "final_value":         round(float(equity.iloc[-1]), 2),
            "initial_value":       round(float(equity.iloc[0]),  2),
        }

    # ------------------------------------------------------------------ #
    #  Trade log
    # ------------------------------------------------------------------ #

    def _build_trade_log(self, df: pd.DataFrame) -> list[dict]:
        trades      = []
        in_trade    = False
        entry_date  = entry_price = trade_type = None
        prev_pos    = 0
        dates       = self._extract_dates(df)

        for i, (pos, price) in enumerate(zip(df["position"], df["close"])):
            pos   = int(pos)
            price = float(price)

            # Close the current trade whenever position changes
            if in_trade and pos != prev_pos:
                exit_date  = dates[i]
                exit_price = price
                if trade_type == "LONG":
                    ret = (exit_price - entry_price) / entry_price
                else:  # SHORT
                    ret = (entry_price - exit_price) / entry_price
                trades.append({
                    "type":        trade_type,
                    "entry_date":  entry_date,
                    "exit_date":   exit_date,
                    "entry_price": round(entry_price, 4),
                    "exit_price":  round(exit_price,  4),
                    "return_pct":  round(ret * 100,   2),
                    "result":      "WIN" if ret > 0 else "LOSS",
                })
                in_trade = False

            # Open a new trade if position is non-zero
            if not in_trade and pos != 0:
                in_trade    = True
                entry_date  = dates[i]
                entry_price = price
                trade_type  = "LONG" if pos == 1 else "SHORT"

            prev_pos = pos

        # Close any open trade at end of data
        if in_trade:
            exit_price = float(df["close"].iloc[-1])
            if trade_type == "LONG":
                ret = (exit_price - entry_price) / entry_price
            else:
                ret = (entry_price - exit_price) / entry_price
            trades.append({
                "type":        trade_type,
                "entry_date":  entry_date,
                "exit_date":   dates[-1] + " (open)",
                "entry_price": round(entry_price, 4),
                "exit_price":  round(exit_price,  4),
                "return_pct":  round(ret * 100,   2),
                "result":      "OPEN",
            })

        return trades

    # ------------------------------------------------------------------ #
    #  Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _extract_dates(df: pd.DataFrame) -> list[str]:
        idx = df.index
        if hasattr(idx, "strftime"):
            return [d.strftime("%Y-%m-%d") for d in idx]
        return [str(d) for d in idx]
