"""
IBKR Stock Data Fetcher – ib_insync
------------------------------------
Prerequisites:
  1. Run TWS or IB Gateway and enable the API:
       TWS      → Edit > Global Configuration > API > Settings
       Gateway  → Configure > API > Settings
  2. pip install ib_insync

Default ports:
  IB Gateway live  : 4001
  IB Gateway paper : 4002
  TWS live         : 7496
  TWS paper        : 7497
"""

import sys

from ib_insync import IB, Stock, util


HOST = "127.0.0.1"
PORT = 4002   # change to match your Gateway/TWS setup
CLIENT_ID = 1


# ---------------------------------------------------------------------------
# Connect
# ---------------------------------------------------------------------------

def connect() -> IB:
    ib = IB()
    ib.connect(HOST, PORT, clientId=CLIENT_ID)
    print(f"Connected  →  {ib.isConnected()}")
    return ib


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------

def get_contract(ib: IB, symbol: str, exchange: str = "SMART", currency: str = "CAD") -> Stock:
    contract = Stock(symbol, exchange, currency)
    ib.qualifyContracts(contract)
    return contract


# ---------------------------------------------------------------------------
# Market data (snapshot)
# ---------------------------------------------------------------------------

def get_snapshot(ib: IB, contract: Stock) -> None:
    """Request a delayed snapshot and print key fields."""
    ib.reqMarketDataType(3)  # 3 = delayed (no live subscription needed)
    ticker = ib.reqMktData(contract, snapshot=False)  # stream; snapshot=True needs live sub
    ib.sleep(2)  # wait for delayed tick to arrive

    print(f"\n{'─' * 40}")
    print(f"  Symbol    : {contract.symbol}")
    print(f"  Last      : {ticker.last}")
    print(f"  Bid/Ask   : {ticker.bid} / {ticker.ask}")
    print(f"  Open      : {ticker.open}")
    print(f"  High      : {ticker.high}")
    print(f"  Low       : {ticker.low}")
    print(f"  Close     : {ticker.close}")
    print(f"  Volume    : {ticker.volume}")

    ib.cancelMktData(contract)


# ---------------------------------------------------------------------------
# Historical data
# ---------------------------------------------------------------------------

def get_history(
    ib: IB,
    contract: Stock,
    duration: str = "1 M",
    bar_size: str = "1 day",
    what_to_show: str = "TRADES",
    use_rth: bool = True,
) -> None:
    """
    Fetch OHLCV bars and print them.

    duration     examples: '1 D', '1 W', '1 M', '3 M', '1 Y'
    bar_size     examples: '1 min', '5 mins', '1 hour', '1 day', '1 week'
    what_to_show examples: 'TRADES', 'MIDPOINT', 'BID', 'ASK'
    """
    bars = ib.reqHistoricalData(
        contract,
        endDateTime="",          # empty = now
        durationStr=duration,
        barSizeSetting=bar_size,
        whatToShow=what_to_show,
        useRTH=use_rth,
        formatDate=1,
    )

    if not bars:
        print("No historical data returned.")
        return

    df = util.df(bars)
    print(f"\nHistorical bars  ({contract.symbol} | {bar_size} | last {duration}):")
    print(df[["date", "open", "high", "low", "close", "volume"]].to_string(index=False))


# ---------------------------------------------------------------------------
# Contract details
# ---------------------------------------------------------------------------

def print_contract_details(ib: IB, contract: Stock) -> None:
    details_list = ib.reqContractDetails(contract)
    if not details_list:
        print("No contract details returned.")
        return
    d = details_list[0]
    c = d.contract
    print(f"\n  conid     : {c.conId}")
    print(f"  Exchange  : {c.primaryExchange or c.exchange}")
    print(f"  Currency  : {c.currency}")
    print(f"  Sec type  : {c.secType}")
    print(f"  Long name : {d.longName}")
    print(f"  Industry  : {d.industry}  /  {d.category}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(symbol: str = "AAPL") -> None:
    print(f"=== IBKR Stock Data Fetcher (ib_insync)  |  symbol={symbol} ===\n")

    ib = connect()
    try:
        contract = get_contract(ib, symbol)

        print(f"\n[1] Contract details for '{symbol}':")
        print_contract_details(ib, contract)

        print(f"\n[2] Market snapshot:")
        get_snapshot(ib, contract)

        print(f"\n[3] 1-month daily history:")
        get_history(ib, contract, duration="1 Y", bar_size="1 day")

    finally:
        ib.disconnect()
        print("\nDisconnected.")


if __name__ == "__main__":
    ticker = sys.argv[1] if len(sys.argv) > 1 else "TD"
    main(ticker)
