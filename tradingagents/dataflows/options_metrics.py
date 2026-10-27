"""Options analytics computed in code, so the agents read numbers rather than make them.

Vendor-neutral: a vendor fetches its chain and normalizes it into an
:class:`OptionChain`, and everything here works on that. Prices are in USD;
volatilities are decimals (0.45 is 45%). Greeks use Black-76 on each expiry's
forward with no discounting, the convention crypto venues such as Deribit quote
their mark IV in, and close enough for the short rates of listed equity options.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

SECONDS_PER_YEAR = 365 * 24 * 3600

# An expiry this close carries a distorted IV and volume/OI, so it is left out
# of the term structure and the 30-day interpolation.
MIN_TERM_DAYS = 1.0

# Crypto trades every day, so a daily return series annualizes by 365.
TRADING_DAYS_PER_YEAR = 365

# The columns a vendor's normalized chain carries.
CHAIN_COLUMNS = (
    "instrument", "expiry", "strike", "type", "bid", "ask", "mark", "iv",
    "open_interest", "volume", "forward",
)


@dataclass
class OptionChain:
    """One underlying's listed options, normalized by a vendor.

    ``quotes`` has one row per contract with :data:`CHAIN_COLUMNS`: ``expiry``
    a UTC timestamp, ``type`` "call" or "put", ``bid``/``ask``/``mark`` in USD
    (NaN when there is no quote), ``iv`` the mark IV as a decimal, and
    ``open_interest``/``volume`` (24h) in units of the underlying (coins,
    shares), not contracts, so venues with different contract sizes compare.
    """

    underlying: str
    venue: str
    as_of: datetime
    spot: float
    quotes: pd.DataFrame


# ---------------------------------------------------------------------------
# Black-76
# ---------------------------------------------------------------------------

def _norm_cdf(x):
    return 0.5 * (1.0 + np.vectorize(math.erf)(np.asarray(x, dtype=float) / math.sqrt(2.0)))


def _norm_pdf(x):
    x = np.asarray(x, dtype=float)
    return np.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def years_to_expiry(expiry, as_of: datetime):
    """Years from ``as_of`` to each expiry, floored at zero."""
    delta = pd.to_datetime(expiry, utc=True) - pd.Timestamp(as_of)
    seconds = delta.dt.total_seconds() if isinstance(delta, pd.Series) else delta.total_seconds()
    return np.maximum(np.asarray(seconds, dtype=float), 0.0) / SECONDS_PER_YEAR


def black76_greeks(forward, strike, years, iv, is_call) -> dict:
    """Delta, gamma, vega (per vol point) and theta (per day) for each contract.

    Arrays broadcast; a contract at or past expiry, or without an IV, gets NaN.
    """
    f, k, t, s = (np.asarray(v, dtype=float) for v in (forward, strike, years, iv))
    call = np.asarray(is_call, dtype=bool)
    valid = (t > 0) & (s > 0) & (f > 0) & (k > 0)
    t_ = np.where(valid, t, 1.0)
    s_ = np.where(valid, s, 1.0)
    f_ = np.where(valid, f, 1.0)
    k_ = np.where(valid, k, 1.0)
    root_t = np.sqrt(t_)
    d1 = (np.log(f_ / k_) + 0.5 * s_ * s_ * t_) / (s_ * root_t)
    pdf = _norm_pdf(d1)
    delta = np.where(call, _norm_cdf(d1), _norm_cdf(d1) - 1.0)
    gamma = pdf / (f_ * s_ * root_t)
    vega = f_ * pdf * root_t / 100.0
    theta = -f_ * pdf * s_ / (2.0 * root_t) / 365.0
    nan = np.nan
    return {
        "delta": np.where(valid, delta, nan),
        "gamma": np.where(valid, gamma, nan),
        "vega": np.where(valid, vega, nan),
        "theta": np.where(valid, theta, nan),
    }


def with_greeks(chain: OptionChain) -> pd.DataFrame:
    """The chain's quotes plus ``years``, ``dte`` and the Black-76 greeks."""
    df = chain.quotes.copy()
    df["years"] = years_to_expiry(df["expiry"], chain.as_of)
    df["dte"] = df["years"] * 365.0
    greeks = black76_greeks(df["forward"], df["strike"], df["years"], df["iv"], df["type"] == "call")
    for name, values in greeks.items():
        df[name] = values
    return df


# ---------------------------------------------------------------------------
# Per-expiry surface metrics
# ---------------------------------------------------------------------------

def _iv_at_strike(group: pd.DataFrame, strike: float) -> float:
    """IV interpolated in strike at ``strike``, calls and puts averaged per strike."""
    by_strike = group.dropna(subset=["iv"]).groupby("strike")["iv"].mean().sort_index()
    if by_strike.empty:
        return math.nan
    return float(np.interp(strike, by_strike.index.values, by_strike.values))


def _iv_at_delta(group: pd.DataFrame, kind: str, target: float) -> float:
    """IV interpolated in delta at ``target`` for one option type, or NaN if out of range."""
    side = group[(group["type"] == kind)].dropna(subset=["iv", "delta"]).sort_values("delta")
    if len(side) < 2 or not side["delta"].min() <= target <= side["delta"].max():
        return math.nan
    return float(np.interp(target, side["delta"].values, side["iv"].values))


def _straddle_move(group: pd.DataFrame, forward: float) -> tuple[float, float]:
    """The ATM straddle's mark as a fraction of the forward, and its strike."""
    strikes = group.groupby("strike")["type"].nunique()
    both = strikes[strikes == 2].index.values
    if len(both) == 0:
        return math.nan, math.nan
    strike = float(both[np.argmin(np.abs(both - forward))])
    at = group[group["strike"] == strike]
    marks = at.set_index("type")["mark"]
    if marks.isna().any():
        return math.nan, strike
    return float(marks.sum() / forward), strike


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0 else math.nan


def expiry_summary(df: pd.DataFrame) -> pd.DataFrame:
    """One row per expiry: forward, ATM IV, 25-delta skew, implied move and positioning.

    ``df`` is :func:`with_greeks` output. ``rr25`` is the 25-delta call IV minus
    the 25-delta put IV (negative: puts are bid over calls); ``bf25`` the wings'
    average IV over ATM.
    """
    rows = []
    for expiry, group in df.groupby("expiry", sort=True):
        forward = float(group["forward"].median())
        atm_iv = _iv_at_strike(group, forward)
        call25 = _iv_at_delta(group, "call", 0.25)
        put25 = _iv_at_delta(group, "put", -0.25)
        move, straddle_strike = _straddle_move(group, forward)
        calls, puts = group[group["type"] == "call"], group[group["type"] == "put"]
        rows.append({
            "expiry": expiry,
            "dte": float(group["dte"].iloc[0]),
            "years": float(group["years"].iloc[0]),
            "forward": forward,
            "atm_iv": atm_iv,
            "call25_iv": call25,
            "put25_iv": put25,
            "rr25": call25 - put25,
            "bf25": (call25 + put25) / 2.0 - atm_iv,
            "implied_move": move,
            "straddle_strike": straddle_strike,
            "pc_oi": _ratio(puts["open_interest"].sum(), calls["open_interest"].sum()),
            "pc_volume": _ratio(puts["volume"].sum(), calls["volume"].sum()),
            "oi_usd": float((group["open_interest"] * forward).sum()),
        })
    return pd.DataFrame(rows)


def constant_maturity_iv(summary: pd.DataFrame, days: float = 30.0) -> tuple[float, bool]:
    """ATM IV at a fixed horizon, interpolated linearly in total variance.

    Returns the IV and whether it had to be extrapolated (no expiry on one side
    of the horizon, so the nearest expiry's IV stands in for it).
    """
    term = summary[(summary["dte"] >= MIN_TERM_DAYS) & summary["atm_iv"].notna()].sort_values("dte")
    if term.empty:
        return math.nan, True
    target = days / 365.0
    years, ivs = term["years"].values, term["atm_iv"].values
    if target <= years[0]:
        return float(ivs[0]), target < years[0]
    if target >= years[-1]:
        return float(ivs[-1]), target > years[-1]
    variance = ivs * ivs * years
    interpolated = np.interp(target, years, variance)
    return float(math.sqrt(interpolated / target)), False


# ---------------------------------------------------------------------------
# Realized volatility and IV history
# ---------------------------------------------------------------------------

def realized_vol(closes: pd.Series, window: int) -> float:
    """Annualized close-to-close volatility over the last ``window`` daily returns."""
    closes = closes.dropna()
    closes = closes[closes > 0]
    returns = np.log(closes).diff().dropna().tail(window)
    if len(returns) < max(2, window // 2):
        return math.nan
    return float(returns.std(ddof=1) * math.sqrt(TRADING_DAYS_PER_YEAR))


def iv_rank(history: pd.Series, current: float) -> tuple[float, float]:
    """Where ``current`` sits in ``history``: rank (0-100 of the range) and percentile."""
    history = history.dropna()
    if history.empty or math.isnan(current):
        return math.nan, math.nan
    low, high = float(history.min()), float(history.max())
    rank = 100.0 * (current - low) / (high - low) if high > low else math.nan
    percentile = 100.0 * float((history < current).mean())
    return rank, percentile


# ---------------------------------------------------------------------------
# Positioning
# ---------------------------------------------------------------------------

def oi_walls(df: pd.DataFrame, kind: str, n: int = 3) -> list[tuple[float, float]]:
    """The ``n`` strikes with the most open interest for one type, across expiries."""
    side = df[df["type"] == kind]
    by_strike = side.groupby("strike")["open_interest"].sum().sort_values(ascending=False)
    return [(float(k), float(v)) for k, v in by_strike.head(n).items() if v > 0]


def max_pain(group: pd.DataFrame) -> float:
    """The settlement price, among listed strikes, that pays option holders least."""
    strikes = np.sort(group["strike"].unique())
    if len(strikes) == 0:
        return math.nan
    calls = group[group["type"] == "call"]
    puts = group[group["type"] == "put"]
    payout = [
        float((calls["open_interest"] * np.maximum(s - calls["strike"], 0.0)).sum()
              + (puts["open_interest"] * np.maximum(puts["strike"] - s, 0.0)).sum())
        for s in strikes
    ]
    return float(strikes[int(np.argmin(payout))])


def most_active(df: pd.DataFrame, n: int = 8) -> pd.DataFrame:
    """The contracts with the most 24h volume, with volume over open interest."""
    active = df[df["volume"] > 0].sort_values("volume", ascending=False).head(n).copy()
    active["vol_oi"] = active["volume"] / active["open_interest"].where(active["open_interest"] > 0)
    return active


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _pct(value: float, digits: int = 1) -> str:
    return "n/a" if value is None or math.isnan(value) else f"{value * 100:.{digits}f}%"


def _pts(value: float) -> str:
    return "n/a" if value is None or math.isnan(value) else f"{value * 100:+.1f}"


def _num(value: float, digits: int = 2) -> str:
    return "n/a" if value is None or math.isnan(value) else f"{value:,.{digits}f}"


def _price(value: float) -> str:
    """A price at a precision that suits its size: BTC strikes and XRP premiums alike."""
    if value is None or math.isnan(value):
        return "n/a"
    digits = 0 if abs(value) >= 1000 else 2 if abs(value) >= 1 else 4
    return f"{value:,.{digits}f}"


def _usd(value: float) -> str:
    return "n/a" if value is None or math.isnan(value) else f"${_price(value)}"


def _implied_range(forward: float, move: float) -> str:
    if math.isnan(move):
        return "n/a"
    return f"{_usd(forward * (1 - move))}-{_usd(forward * (1 + move))}"


def _expiry_label(expiry) -> str:
    return pd.Timestamp(expiry).strftime("%Y-%m-%d")


def format_snapshot(
    chain: OptionChain,
    closes: pd.Series | None = None,
    iv_index: pd.Series | None = None,
    iv_index_name: str = "",
    max_expiries: int = 10,
) -> str:
    """The options-market snapshot an analyst reads: vol level, term structure, skew, positioning.

    ``closes`` (daily underlying closes) gives realized vol; ``iv_index`` (a
    daily IV index such as Deribit's DVOL, in vol points) gives IV rank. Either
    may be None, and its section then says so instead of guessing.
    """
    df = with_greeks(chain)
    summary = expiry_summary(df)
    term = summary[summary["dte"] >= MIN_TERM_DAYS]
    iv30, extrapolated = constant_maturity_iv(summary, 30.0)

    lines = [
        f"## Options snapshot: {chain.underlying} on {chain.venue}",
        f"As of {pd.Timestamp(chain.as_of).strftime('%Y-%m-%d %H:%M UTC')}. "
        f"Spot index {_usd(chain.spot)}. Prices in USD; IV is the venue's mark IV; "
        f"greeks are Black-76 on each expiry's forward. {len(df)} contracts across "
        f"{df['expiry'].nunique()} expiries.",
        "",
        "### Volatility level",
        f"- 30-day ATM IV (constant maturity{', extrapolated' if extrapolated else ''}): {_pct(iv30)}",
    ]

    if closes is not None and not closes.empty:
        rv7, rv30 = realized_vol(closes, 7), realized_vol(closes, 30)
        lines.append(
            f"- Realized vol (close-to-close, annualized): 7-day {_pct(rv7)}, 30-day {_pct(rv30)}"
        )
        if not (math.isnan(iv30) or math.isnan(rv30)) and rv30 > 0:
            lines.append(
                f"- 30-day IV minus 30-day realized: {_pts(iv30 - rv30)} vol pts "
                f"(IV/RV {iv30 / rv30:.2f}); positive means options price more movement than the "
                f"underlying has delivered"
            )
    else:
        lines.append("- Realized vol: unavailable (no underlying price history)")

    if iv_index is not None and not iv_index.empty:
        current = float(iv_index.iloc[-1])
        rank, percentile = iv_rank(iv_index, current)
        start = pd.Timestamp(iv_index.index[0]).strftime("%Y-%m-%d")
        lines.append(
            f"- {iv_index_name or 'IV index'}: {current:.1f}; range since {start} "
            f"{iv_index.min():.1f}-{iv_index.max():.1f}; IV rank {_num(rank, 0)}, "
            f"IV percentile {_num(percentile, 0)}"
        )
    else:
        lines.append("- IV rank: unavailable (no implied-volatility history for this underlying)")

    lines += [
        "",
        "### Term structure and skew",
        "RR25 = 25-delta call IV minus 25-delta put IV (negative: downside protection is bid). "
        "BF25 = average 25-delta wing IV minus ATM IV. Implied move = ATM straddle / forward, "
        "the move to expiry the straddle breaks even on; implied range = the forward plus and "
        "minus that move. Breakevens, not probability bounds.",
        "",
        "| Expiry | DTE | Forward | ATM IV | RR25 (pts) | BF25 (pts) | Implied move | Implied range "
        "| P/C OI | P/C vol | OI notional |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in term.head(max_expiries).itertuples():
        lines.append(
            f"| {_expiry_label(row.expiry)} | {row.dte:.1f} | {_usd(row.forward)} | {_pct(row.atm_iv)} "
            f"| {_pts(row.rr25)} | {_pts(row.bf25)} | {_pct(row.implied_move)} "
            f"| {_implied_range(row.forward, row.implied_move)} "
            f"| {_num(row.pc_oi)} | {_num(row.pc_volume)} | {_usd(row.oi_usd)} |"
        )
    if len(term) > max_expiries:
        lines.append(f"\n({len(term) - max_expiries} later expiries omitted.)")
    skipped = summary[summary["dte"] < MIN_TERM_DAYS]
    if not skipped.empty:
        lines.append(
            f"\nExpiring within a day and left out above: "
            f"{', '.join(_expiry_label(e) for e in skipped['expiry'])}."
        )

    calls_oi = df.loc[df["type"] == "call", "open_interest"].sum()
    puts_oi = df.loc[df["type"] == "put", "open_interest"].sum()
    calls_vol = df.loc[df["type"] == "call", "volume"].sum()
    puts_vol = df.loc[df["type"] == "put", "volume"].sum()
    walls = {kind: ", ".join(f"{_usd(k)} ({_num(v, 1)})" for k, v in oi_walls(df, kind))
             for kind in ("call", "put")}
    lines += [
        "",
        "### Positioning",
        f"Open interest and volume are in {chain.underlying}. Open interest shows how much is open, not who is long or short it.",
        "",
        f"- Put/call open interest, all expiries: {_num(_ratio(puts_oi, calls_oi))} "
        f"(puts {_num(puts_oi, 1)} / calls {_num(calls_oi, 1)} {chain.underlying})",
        f"- Put/call 24h volume: {_num(_ratio(puts_vol, calls_vol))} "
        f"(puts {_num(puts_vol, 1)} / calls {_num(calls_vol, 1)} {chain.underlying})",
        f"- Largest call open interest by strike: {walls['call'] or 'none'}",
        f"- Largest put open interest by strike: {walls['put'] or 'none'}",
    ]
    near = term[term["dte"] <= 45]
    if not near.empty:
        biggest = near.loc[near["oi_usd"].idxmax()]
        pain = max_pain(df[df["expiry"] == biggest["expiry"]])
        lines.append(
            f"- Max pain for the largest expiry within 45 days ({_expiry_label(biggest['expiry'])}, "
            f"{_usd(biggest['oi_usd'])} open): {_usd(pain)}; a weak pull at best, strongest into expiry"
        )

    active = most_active(df)
    if not active.empty:
        lines += [
            "",
            "### Most active contracts (24h volume)",
            "",
            "| Contract | Type | Strike | Expiry | Volume | Open interest | Vol/OI | IV | Delta |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for row in active.itertuples():
            lines.append(
                f"| {row.instrument} | {row.type} | {_usd(row.strike)} | {_expiry_label(row.expiry)} "
                f"| {_num(row.volume, 1)} | {_num(row.open_interest, 1)} | {_num(row.vol_oi)} "
                f"| {_pct(row.iv)} | {_num(row.delta)} |"
            )
    return "\n".join(lines)


def pick_expiry(expiries, requested: str, as_of: datetime, min_days: float = 7.0):
    """The listed expiry nearest a requested date, or the first ``min_days`` out."""
    expiries = sorted(pd.to_datetime(pd.Series(expiries).unique(), utc=True))
    if not expiries:
        return None
    if requested:
        target = pd.Timestamp(requested, tz="UTC")
        return min(expiries, key=lambda e: abs((e.normalize() - target).total_seconds()))
    cutoff = pd.Timestamp(as_of) + pd.Timedelta(days=min_days)
    later = [e for e in expiries if e >= cutoff]
    return later[0] if later else expiries[-1]


def format_chain(chain: OptionChain, expiry: str = "", moneyness_pct: float = 15.0,
                 max_strikes: int = 30) -> str:
    """One expiry's calls and puts side by side, strikes within ``moneyness_pct`` of the forward."""
    df = with_greeks(chain)
    chosen = pick_expiry(df["expiry"], expiry, chain.as_of)
    if chosen is None:
        return f"No listed options for {chain.underlying} on {chain.venue}."
    group = df[df["expiry"] == chosen]
    forward = float(group["forward"].median())
    band = abs(moneyness_pct) / 100.0
    group = group[(group["strike"] >= forward * (1 - band)) & (group["strike"] <= forward * (1 + band))]
    strikes = np.sort(group["strike"].unique())
    if len(strikes) > max_strikes:
        nearest = np.argsort(np.abs(strikes - forward))[:max_strikes]
        strikes = np.sort(strikes[nearest])

    calls = group[group["type"] == "call"].set_index("strike")
    puts = group[group["type"] == "put"].set_index("strike")

    def cells(side: pd.DataFrame, strike: float) -> str:
        if strike not in side.index:
            return "| - | - | - | - | - | - "
        r = side.loc[strike]
        return (f"| {_price(r['bid'])} | {_price(r['ask'])} | {_price(r['mark'])} | {_pct(r['iv'])} "
                f"| {_num(r['delta'])} | {_num(r['open_interest'], 1)} ")

    dte = float(group["dte"].iloc[0]) if not group.empty else math.nan
    listed = ", ".join(_expiry_label(e) for e in sorted(df["expiry"].unique()))
    lines = [
        f"## Option chain: {chain.underlying} on {chain.venue}, expiry {_expiry_label(chosen)} "
        f"({_num(dte, 1)} days)",
        f"Forward {_usd(forward)}; strikes within {abs(moneyness_pct):g}% of it. Prices in USD; "
        f"IV is mark IV; delta is Black-76. Open interest in {chain.underlying}.",
        "",
        "| Strike | C bid | C ask | C mark | C IV | C delta | C OI | P bid | P ask | P mark | P IV | P delta | P OI |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for strike in strikes:
        lines.append(f"| {_usd(strike)} {cells(calls, strike)}{cells(puts, strike)}|")
    if len(strikes) == 0:
        lines.append(f"\nNo strikes within {abs(moneyness_pct):g}% of the forward.")
    lines += ["", f"Listed expiries: {listed}"]
    return "\n".join(lines)
