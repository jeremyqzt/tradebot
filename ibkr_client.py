"""
IBKR Client – wraps ib_insync for the trading app.
Supports IB Gateway (paper port 4002 by default).
Falls back gracefully when ib_insync is unavailable.

Python 3.11 notes
-----------------
ib_insync uses asyncio internally.  On Python 3.10+ the event-loop policy
changed so we call ``util.patchAsyncio()`` (which applies nest_asyncio) as
soon as the library is imported.  This lets ib_insync's nested event-loop
run inside Flask's synchronous request threads without errors.
"""
from __future__ import annotations

import logging
import pandas as pd

logger = logging.getLogger(__name__)

try:
    from ib_insync import IB, Stock, MarketOrder, LimitOrder, util
    # Required for Python 3.10+ (including 3.11): patches asyncio so ib_insync
    # can run its nested event loop inside a synchronous Flask thread.
    util.patchAsyncio()
    IB_AVAILABLE = True
except ImportError:
    IB_AVAILABLE = False
    logger.warning("ib_insync not installed – IBKR connection disabled.")
except Exception as exc:
    # patchAsyncio can raise if nest_asyncio is missing; warn but continue.
    IB_AVAILABLE = False
    logger.warning("ib_insync import/patch failed (%s) – IBKR disabled.", exc)


class IBKRClient:
    def __init__(self):
        self.ib = IB() if IB_AVAILABLE else None
        self.connected = False
        self._host = "127.0.0.1"
        self._port = 4002
        self._client_id = 1

    # ------------------------------------------------------------------ #
    #  Connection
    # ------------------------------------------------------------------ #

    def connect(self, host: str = "127.0.0.1", port: int = 4002, client_id: int = 1):
        if not IB_AVAILABLE:
            return {"success": False, "message": "ib_insync is not installed."}
        try:
            if self.connected:
                self.ib.disconnect()
            self._host = host
            self._port = port
            self._client_id = client_id
            self.ib.connect(host, port, clientId=client_id, timeout=10)
            self.connected = True
            return {
                "success": True,
                "message": f"Connected to IB Gateway at {host}:{port}",
            }
        except Exception as e:
            self.connected = False
            return {"success": False, "message": str(e)}

    def disconnect(self):
        if self.ib and self.connected:
            self.ib.disconnect()
        self.connected = False
        return {"success": True, "message": "Disconnected"}

    # ------------------------------------------------------------------ #
    #  Historical Data
    # ------------------------------------------------------------------ #

    def get_historical_data(
        self,
        symbol: str,
        duration: str = "2 Y",
        bar_size: str = "1 day",
        currency: str = "CAD",
        exchange: str = "SMART",
        end_datetime: str = "",
    ) -> pd.DataFrame | None:
        """
        Fetch historical daily bars from IBKR.

        Parameters
        ----------
        end_datetime : IBKR endDateTime string, e.g. "20240101 00:00:00".
                       Empty string (default) means 'now'.
                       Set this for differential / partial-range downloads.
        """
        if not self.connected or self.ib is None:
            return None
        try:
            contract = Stock(symbol, exchange, currency)
            self.ib.qualifyContracts(contract)
            bars = self.ib.reqHistoricalData(
                contract,
                endDateTime=end_datetime,   # "" = now, or specific date for range
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow="TRADES",
                useRTH=True,
                formatDate=1,
            )
            if not bars:
                return None
            df = util.df(bars)
            df = df.rename(columns={"date": "Date"})
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.set_index("Date")
            df.columns = [c.lower() for c in df.columns]
            # Keep only OHLCV
            df = df[["open", "high", "low", "close", "volume"]].copy()
            return df
        except Exception as e:
            logger.error(f"get_historical_data error: {e}")
            return None

    # ------------------------------------------------------------------ #
    #  Paper Trading
    # ------------------------------------------------------------------ #

    def place_order(
        self,
        symbol: str,
        action: str,
        quantity: int,
        order_type: str = "MKT",
        limit_price: float | None = None,
        currency: str = "CAD",
    ):
        if not self.connected or self.ib is None:
            return {"success": False, "message": "Not connected to IB Gateway"}
        try:
            contract = Stock(symbol, "SMART", currency)
            self.ib.qualifyContracts(contract)

            if order_type == "MKT":
                order = MarketOrder(action.upper(), quantity)
            elif order_type == "LMT" and limit_price:
                order = LimitOrder(action.upper(), quantity, limit_price)
            else:
                return {"success": False, "message": f"Unsupported order type: {order_type}"}

            trade = self.ib.placeOrder(contract, order)
            self.ib.sleep(1)
            return {
                "success": True,
                "order_id": trade.order.orderId,
                "status": trade.orderStatus.status,
                "symbol": symbol,
                "action": action,
                "quantity": quantity,
            }
        except Exception as e:
            return {"success": False, "message": str(e)}

    # ------------------------------------------------------------------ #
    #  Account / Portfolio
    # ------------------------------------------------------------------ #

    def get_portfolio(self):
        if not self.connected or self.ib is None:
            return []
        try:
            return [
                {
                    "symbol": item.contract.symbol,
                    "position": item.position,
                    "avg_cost": round(item.averageCost, 4),
                    "market_price": round(item.marketPrice, 4),
                    "market_value": round(item.marketValue, 2),
                    "unrealized_pnl": round(item.unrealizedPNL, 2),
                    "realized_pnl": round(item.realizedPNL, 2),
                }
                for item in self.ib.portfolio()
            ]
        except Exception as e:
            logger.error(f"get_portfolio error: {e}")
            return []

    def get_account_summary(self):
        if not self.connected or self.ib is None:
            return {}
        try:
            summary = self.ib.accountSummary()
            keep = {
                "NetLiquidation", "TotalCashValue", "BuyingPower",
                "UnrealizedPnL", "RealizedPnL", "GrossPositionValue",
            }
            return {
                item.tag: item.value
                for item in summary
                if item.tag in keep
            }
        except Exception as e:
            logger.error(f"get_account_summary error: {e}")
            return {}

    def get_open_orders(self):
        if not self.connected or self.ib is None:
            return []
        try:
            trades = self.ib.openTrades()
            return [
                {
                    "order_id": t.order.orderId,
                    "symbol": t.contract.symbol,
                    "action": t.order.action,
                    "quantity": t.order.totalQuantity,
                    "order_type": t.order.orderType,
                    "status": t.orderStatus.status,
                }
                for t in trades
            ]
        except Exception as e:
            logger.error(f"get_open_orders error: {e}")
            return []
