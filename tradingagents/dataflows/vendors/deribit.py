"""Deribit options vendor, through ccxt.

Deribit lists options on BTC and ETH settled in the coin, and on a handful of
other coins (SOL, XRP, AVAX, ...) settled in USDC. Its market data is public, so
no key is needed. A coin-settled option is quoted in the coin; its prices are
converted to USD at the expiry's forward (``underlying_price``), the price
Deribit itself converts at.

Deribit serves the live chain only, so a run dated before today is told the
chain is withheld rather than handed today's: that would put post-decision
information into a past analysis.
"""

from __future__ import annotations

import functools
import logging
import math
import time
from datetime import UTC, datetime, timedelta

import ccxt
import pandas as pd

from tradingagents.dataflows.date_window import is_historical
from tradingagents.dataflows.errors import NoMarketDataError, VendorUnavailableError
from tradingagents.dataflows.options_metrics import OptionChain, format_chain, format_snapshot
from tradingagents.dataflows.symbols import crypto_base

logger = logging.getLogger(__name__)

VENUE = "Deribit"

# Network timeout, consistent with the other vendors (ccxt takes milliseconds).
REQUEST_TIMEOUT_MS = 30_000

# Coins whose options settle in the coin itself; Deribit settles the rest in USDC.
COIN_SETTLED = frozenset({"BTC", "ETH"})

# Coins Deribit publishes a DVOL implied-volatility index for.
DVOL_CURRENCIES = frozenset({"BTC", "ETH"})

IV_HISTORY_DAYS = 365
CLOSES_DAYS = 60

# How long a fetched chain is reused: an analyst's snapshot and chain calls,
# seconds apart, then read the same quotes rather than two slightly different ones.
CHAIN_TTL_SECONDS = 60

# A dropped request is retried after these pauses (seconds) before the vendor is
# reported unavailable: one blip late in a run would otherwise cost its trade.
RETRY_PAUSES = (1.0, 3.0)
_chains: dict[str, tuple[float, OptionChain]] = {}


@functools.lru_cache(maxsize=1)
def _exchange():
    return ccxt.deribit({"enableRateLimit": True, "timeout": REQUEST_TIMEOUT_MS})


def _call(fn, *args, **kwargs):
    """A ccxt call, with its failures raised as the router's vendor errors.

    A network failure is retried after each of RETRY_PAUSES; an error Deribit
    itself returns is not, since asking again gets the same answer.
    """
    for pause in (*RETRY_PAUSES, None):
        try:
            return fn(*args, **kwargs)
        except ccxt.NetworkError as exc:
            if pause is None:
                raise VendorUnavailableError(f"Deribit request failed: {exc}") from exc
            logger.warning("Deribit request failed (%s); retrying in %gs", exc, pause)
            time.sleep(pause)
        except ccxt.ExchangeError as exc:
            raise VendorUnavailableError(f"Deribit returned an error: {exc}") from exc


def _withheld(as_of_date: str, what: str) -> str:
    return (
        f"Options data is withheld for {as_of_date}. Deribit serves only the live {what}, "
        f"with no historical vintage, so serving it would put post-decision information "
        f"into a {as_of_date} analysis. Proceed without options data."
    )


def _base_or_raise(symbol: str) -> str:
    base = crypto_base(symbol)
    if base is None:
        raise NoMarketDataError(symbol, detail="Deribit lists options on crypto assets only")
    return base


def _float(value) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) else math.nan


def fetch_chain(symbol: str) -> OptionChain:
    """The live Deribit option chain for a crypto symbol such as ``BTC-USD``."""
    base = _base_or_raise(symbol)
    cached = _chains.get(base)
    if cached and time.monotonic() - cached[0] < CHAIN_TTL_SECONDS:
        return cached[1]
    chain = _fetch_chain(symbol, base)
    _chains[base] = (time.monotonic(), chain)
    return chain


def _fetch_chain(symbol: str, base: str) -> OptionChain:
    settle = base if base in COIN_SETTLED else "USDC"
    exchange = _exchange()
    try:
        _call(exchange.load_markets, True)  # reload: new expiries list daily
    except VendorUnavailableError:
        if not exchange.markets:
            raise
        # The list loaded earlier in this process misses at most the day's new
        # expiries; that beats no chain at all.
        logger.warning("Deribit market list could not be refreshed; using the one loaded earlier")
    tickers = _call(exchange.fetch_option_chain, settle)
    as_of = datetime.now(UTC)

    rows, spots = [], []
    for market_symbol, ticker in tickers.items():
        market = exchange.markets.get(market_symbol)
        info = ticker.get("info") or {}
        if not market or market.get("base") != base or market.get("settle") != settle:
            continue
        forward = _float(info.get("underlying_price"))
        iv = _float(info.get("mark_iv")) / 100.0
        if math.isnan(forward) or forward <= 0 or math.isnan(iv) or iv <= 0:
            continue
        # A coin-settled option is quoted in the coin; a USDC one already in USD.
        to_usd = forward if settle == base else 1.0
        rows.append({
            "instrument": info.get("instrument_name") or market.get("id"),
            "expiry": pd.Timestamp(market["expiry"], unit="ms", tz="UTC"),
            "strike": _float(market.get("strike")),
            "type": market.get("optionType"),
            "bid": _float(info.get("bid_price")) * to_usd,
            "ask": _float(info.get("ask_price")) * to_usd,
            "mark": _float(info.get("mark_price")) * to_usd,
            "iv": iv,
            # Deribit counts option open interest and volume in the coin.
            "open_interest": _float(info.get("open_interest")),
            "volume": _float(info.get("volume")),
            "forward": forward,
        })
        spots.append(_float(info.get("estimated_delivery_price")))

    if not rows:
        raise NoMarketDataError(symbol, f"{base} options", detail=f"{VENUE} lists no {base} options")
    quotes = pd.DataFrame(rows)
    quotes[["open_interest", "volume"]] = quotes[["open_interest", "volume"]].fillna(0.0)
    spot = pd.Series(spots).median()
    return OptionChain(underlying=base, venue=VENUE, as_of=as_of, spot=float(spot), quotes=quotes)


def fetch_daily_closes(base: str, days: int = CLOSES_DAYS) -> pd.Series:
    """Completed daily closes of the coin's Deribit perpetual, oldest first."""
    exchange = _exchange()
    perpetual = f"{base}/USD:{base}" if base in COIN_SETTLED else f"{base}/USDC:USDC"
    candles = _call(exchange.fetch_ohlcv, perpetual, "1d", limit=days + 1)
    if not candles:
        return pd.Series(dtype=float)
    frame = pd.DataFrame(candles, columns=["ts", "open", "high", "low", "close", "volume"])
    frame["date"] = pd.to_datetime(frame["ts"], unit="ms", utc=True)
    today = pd.Timestamp.now(tz="UTC").normalize()
    frame = frame[frame["date"] < today]  # today's candle is still open
    return frame.set_index("date")["close"].astype(float)


def fetch_dvol(base: str, days: int = IV_HISTORY_DAYS) -> pd.Series:
    """Daily DVOL closes for the past ``days``, or an empty series where Deribit has none."""
    if base not in DVOL_CURRENCIES:
        return pd.Series(dtype=float)
    end = datetime.now(UTC)
    start = end - timedelta(days=days)
    response = _call(_exchange().public_get_get_volatility_index_data, {
        "currency": base,
        "start_timestamp": int(start.timestamp() * 1000),
        "end_timestamp": int(end.timestamp() * 1000),
        "resolution": "1D",
    })
    data = (response.get("result") or {}).get("data") or []
    if not data:
        return pd.Series(dtype=float)
    frame = pd.DataFrame(data, columns=["ts", "open", "high", "low", "close"])
    index = pd.to_datetime(frame["ts"], unit="ms", utc=True)
    return pd.Series(frame["close"].astype(float).values, index=index).sort_index()


def _optional(fetch, base: str, label: str) -> pd.Series | None:
    """A supporting series, or None when it cannot be fetched: the snapshot still stands."""
    try:
        return fetch(base)
    except Exception as exc:  # noqa: BLE001 - one missing input must not drop the snapshot
        logger.warning("Deribit %s unavailable for %s: %s", label, base, exc)
        return None


def get_options_snapshot(symbol: str, curr_date: str | None = None) -> str:
    """Vol level, term structure, skew and positioning for the symbol's Deribit options."""
    if is_historical(curr_date):
        return _withheld(curr_date, "option chain")
    chain = fetch_chain(symbol)
    closes = _optional(fetch_daily_closes, chain.underlying, "price history")
    dvol = _optional(fetch_dvol, chain.underlying, "DVOL history")
    return format_snapshot(chain, closes=closes, iv_index=dvol, iv_index_name=f"DVOL ({chain.underlying})")


def get_chain(symbol: str, curr_date: str | None = None) -> OptionChain:
    """The live chain itself, for code that prices structures; withheld for a past date."""
    if is_historical(curr_date):
        raise NoMarketDataError(symbol, detail=f"{VENUE} serves only the live chain; withheld for {curr_date}")
    return fetch_chain(symbol)


def get_daily_closes(symbol: str, curr_date: str | None = None) -> pd.Series:
    """The underlying's completed daily closes, for realized volatility; withheld for a past date."""
    base = _base_or_raise(symbol)
    if is_historical(curr_date):
        raise NoMarketDataError(symbol, detail=f"{VENUE} closes are fetched live; withheld for {curr_date}")
    return fetch_daily_closes(base)


def get_option_chain(
    symbol: str, expiry: str = "", moneyness_pct: float = 15.0, curr_date: str | None = None
) -> str:
    """One expiry of the symbol's Deribit chain, strikes near the forward, with greeks."""
    if is_historical(curr_date):
        return _withheld(curr_date, "option chain")
    return format_chain(fetch_chain(symbol), expiry=expiry, moneyness_pct=moneyness_pct)
