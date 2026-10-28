"""Option structures: proposed legs checked against the chain and priced in code.

The strategist names the legs; this module decides whether they can be traded and
what they are worth. Each leg is matched to a listed contract and priced on the
side of the book it would fill on (buy at the ask, sell at the bid). The payoff at
expiry then gives the maximum profit and loss, the breakevens and a probability of
profit, so no model does the arithmetic. The venue's taker fees are charged on
entry, and the expected P&L is given at the expiry's ATM IV and at the
underlying's realized volatility: the gap between the two is the volatility edge.

The desk's risk limits are checked here too, so a structure that breaks one never
reaches the risk debate.

Defined risk only: a structure whose loss has no bound (more short calls than long
ones) is refused. The legs share one expiry, so the payoff at expiry is exact.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from tradingagents.dataflows import options_metrics as om

MAX_LEGS = 4

# Too close to expiry to price a structure on: the payoff is decided by the next
# few hours' noise and the quotes are at their thinnest.
MIN_DTE = 2.0

# Risk limits. A leg whose bid-ask spread is wider than this share of its mark,
# or that has no open interest, is too thin to enter and exit.
WIDE_SPREAD = 0.25
# A structure must win more often than not, or pay at least what it risks: a
# credit spread passes on the first, a debit spread or an iron butterfly on the second.
MIN_POP = 0.55
MIN_REWARD_TO_RISK = 1.0

# Taker fee per leg, by venue: a share of the underlying per unit, capped at a
# share of the option's price.
FEES = {"Deribit": {"of_underlying": 0.0003, "cap_of_price": 0.125}}

# What the desk knows about the venue's contracts, so it is not weighed as an unknown.
VENUE_TERMS = {
    "Deribit": (
        "European options, cash-settled at expiry (08:00 UTC) at the average of the Deribit index over "
        "the preceding 30 minutes. BTC and ETH options are quoted and settled in the coin, so their USD "
        "P&L also moves with the coin's price; the other coins settle in USDC. Fees: 0.03% of the "
        "underlying per leg on entry and on exit (capped at 12.5% of the option's price), plus 0.015% "
        "at settlement on a leg that ends in the money, same cap. Collateral: hold at least the maximum "
        "loss in the settlement currency; a short leg can need more initial margin than that under "
        "standard margin."
    ),
}


@dataclass
class Leg:
    """One leg as proposed: ``side`` "buy"/"sell", ``kind`` "call"/"put", ``expiry`` YYYY-MM-DD."""

    side: str
    kind: str
    strike: float
    expiry: str
    quantity: float = 1.0


@dataclass
class PricedLeg:
    """A leg matched to a listed contract, with its quote and greeks (per unit of the underlying)."""

    side: str
    kind: str
    strike: float
    expiry: pd.Timestamp
    quantity: float
    instrument: str
    bid: float
    ask: float
    mark: float
    fill: float
    iv: float
    delta: float
    gamma: float
    vega: float
    theta: float
    open_interest: float

    @property
    def sign(self) -> int:
        return 1 if self.side == "buy" else -1


@dataclass
class StructureEvaluation:
    """What a structure costs and can make or lose at expiry, per unit of the underlying.

    ``net_fill`` and ``net_mid`` are positive for a debit, negative for a credit.
    ``max_loss`` is the loss as a positive number. ``max_profit`` is ``inf`` when
    the upside has no bound.
    """

    underlying: str
    venue: str
    as_of: object
    legs: list[PricedLeg] = field(default_factory=list)
    expiry: pd.Timestamp | None = None
    dte: float = math.nan
    forward: float = math.nan
    atm_iv: float = math.nan
    net_fill: float = math.nan
    net_mid: float = math.nan
    max_profit: float = math.nan
    max_loss: float = math.nan
    breakevens: list[float] = field(default_factory=list)
    pop: float = math.nan
    fees: float = 0.0
    round_trip_cost: float = math.nan
    expected_pnl_iv: float = math.nan
    realized_vol: float = math.nan
    expected_pnl_rv: float = math.nan
    pop_rv: float = math.nan
    greeks: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    breaches: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors and bool(self.legs)

    @property
    def tradeable(self) -> bool:
        """Priced, and within the desk's risk limits."""
        return self.ok and not self.breaches


def _intrinsic(kind: str, strike: float, spot):
    spot = np.asarray(spot, dtype=float)
    return np.maximum(spot - strike, 0.0) if kind == "call" else np.maximum(strike - spot, 0.0)


def _day(timestamp) -> str:
    return pd.Timestamp(timestamp).strftime("%Y-%m-%d")


def _match_leg(df: pd.DataFrame, expiries: list, leg: Leg, ev: StructureEvaluation) -> PricedLeg | None:
    """The listed contract nearest a proposed leg, or None with the reason in ``ev.errors``."""
    label = f"{leg.side} {leg.kind} {leg.strike:g} {leg.expiry}"
    side, kind = str(leg.side).lower(), str(leg.kind).lower()
    if side not in ("buy", "sell") or kind not in ("call", "put"):
        ev.errors.append(f"{label}: side must be buy or sell and type call or put")
        return None
    if not leg.quantity or leg.quantity <= 0:
        ev.errors.append(f"{label}: quantity must be positive")
        return None
    try:
        wanted = pd.Timestamp(leg.expiry, tz="UTC").normalize()
    except (ValueError, TypeError):
        ev.errors.append(f"{label}: expiry is not a date (use YYYY-MM-DD)")
        return None

    expiry = min(expiries, key=lambda e: abs((e.normalize() - wanted).total_seconds()))
    if expiry.normalize() != wanted:
        ev.notes.append(f"{label}: {leg.expiry} is not a listed expiry; used {_day(expiry)}")
    side_rows = df[(df["expiry"] == expiry) & (df["type"] == kind)]
    if side_rows.empty:
        ev.errors.append(f"{label}: no {kind}s listed for {_day(expiry)}")
        return None
    row = side_rows.loc[(side_rows["strike"] - leg.strike).abs().idxmin()]
    if row["strike"] != leg.strike:
        ev.notes.append(f"{label}: strike {leg.strike:g} is not listed; used {row['strike']:g}")

    fill = row["ask"] if side == "buy" else row["bid"]
    if not (fill > 0):
        book = "ask to buy at" if side == "buy" else "bid to sell into"
        ev.errors.append(f"{label}: {row['instrument']} has no {book}")
        return None
    return PricedLeg(
        side=side, kind=kind, strike=float(row["strike"]), expiry=expiry, quantity=float(leg.quantity),
        instrument=str(row["instrument"]), bid=float(row["bid"]), ask=float(row["ask"]),
        mark=float(row["mark"]), fill=float(fill), iv=float(row["iv"]), delta=float(row["delta"]),
        gamma=float(row["gamma"]), vega=float(row["vega"]), theta=float(row["theta"]),
        open_interest=float(row["open_interest"]),
    )


def _payoff_profile(legs: list[PricedLeg], cost: float):
    """P&L at expiry at zero and at every strike, and its slope above the top strike."""
    points = np.array([0.0, *sorted({leg.strike for leg in legs})])
    pnl = sum(leg.sign * leg.quantity * _intrinsic(leg.kind, leg.strike, points) for leg in legs) - cost
    slope = sum(leg.sign * leg.quantity for leg in legs if leg.kind == "call")
    return points, np.asarray(pnl, dtype=float), float(slope)


def _breakevens(points, pnl, slope) -> list[float]:
    found = []
    for i in range(len(points) - 1):
        x0, x1, y0, y1 = points[i], points[i + 1], pnl[i], pnl[i + 1]
        if y0 == 0 and x0 > 0:
            found.append(x0)
        elif y0 * y1 < 0:
            found.append(x0 - y0 * (x1 - x0) / (y1 - y0))
    # Above the top strike the P&L moves at ``slope`` per unit of the underlying.
    if pnl[-1] == 0 and points[-1] > 0:
        found.append(points[-1])
    elif slope != 0 and pnl[-1] * slope < 0:
        found.append(points[-1] - pnl[-1] / slope)
    return sorted({round(float(x), 8) for x in found})


def _b76_value(kind: str, forward: float, strike: float, years: float, vol: float) -> float:
    """Black-76 value of a call or put, undiscounted: its expected payoff at expiry."""
    root = vol * math.sqrt(years)
    d1 = (math.log(forward / strike) + 0.5 * root * root) / root
    d2 = d1 - root

    def n(x):
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    if kind == "call":
        return forward * n(d1) - strike * n(d2)
    return strike * n(-d2) - forward * n(-d1)


def _expected_pnl(legs, cost, forward, vol, years) -> float:
    """Expected P&L at expiry if the underlying ends lognormal at ``vol`` around the forward."""
    if not (forward > 0 and vol > 0 and years > 0):
        return math.nan
    payoff = sum(leg.sign * leg.quantity * _b76_value(leg.kind, forward, leg.strike, years, vol) for leg in legs)
    return float(payoff - cost)


def _fee(venue: str, forward: float, price: float) -> float:
    schedule = FEES.get(venue)
    if not schedule:
        return 0.0
    return min(schedule["of_underlying"] * forward, schedule["cap_of_price"] * price)


def _limit_breaches(ev: StructureEvaluation) -> list[str]:
    """The desk's risk limits the priced structure breaks, if any."""
    found = []
    for p in ev.legs:
        if p.mark > 0 and (p.ask - p.bid) / p.mark > WIDE_SPREAD:
            found.append(f"{p.instrument}: bid-ask spread is {(p.ask - p.bid) / p.mark:.0%} of its mark "
                         f"(limit {WIDE_SPREAD:.0%})")
        if p.open_interest <= 0:
            found.append(f"{p.instrument}: no open interest")
    reward = math.inf if ev.max_profit == math.inf or ev.max_loss <= 0 else ev.max_profit / ev.max_loss
    if not (ev.pop >= MIN_POP or reward >= MIN_REWARD_TO_RISK):
        found.append(f"probability of profit {ev.pop:.0%} is under {MIN_POP:.0%} and reward to risk "
                     f"{reward:.2f} is under {MIN_REWARD_TO_RISK:g}: it neither wins often enough nor "
                     f"pays enough when it does")
    return found


def _probability_of_profit(legs, cost, forward, iv, years) -> float:
    """Chance the structure ends in profit, lognormal at the forward and the given volatility."""
    if not (forward > 0 and iv > 0 and years > 0):
        return math.nan
    z = np.linspace(-8.0, 8.0, 8001)
    spot = forward * np.exp(-0.5 * iv * iv * years + iv * math.sqrt(years) * z)
    pnl = sum(leg.sign * leg.quantity * _intrinsic(leg.kind, leg.strike, spot) for leg in legs) - cost
    weight = np.exp(-0.5 * z * z)
    return float(weight[pnl > 0].sum() / weight.sum())


def evaluate_structure(chain: om.OptionChain, legs: list[Leg], min_dte: float = MIN_DTE,
                       realized_vol: float | None = None) -> StructureEvaluation:
    """Match the legs to listed contracts and price the structure at expiry, after entry fees.

    ``realized_vol`` (annualized), when known, adds the expected P&L and the chance
    of profit if the underlying keeps moving as it has.
    """
    ev = StructureEvaluation(underlying=chain.underlying, venue=chain.venue, as_of=chain.as_of)
    if not legs:
        ev.errors.append("the structure has no legs")
        return ev
    if len(legs) > MAX_LEGS:
        ev.errors.append(f"the structure has {len(legs)} legs; at most {MAX_LEGS} are allowed")
        return ev

    df = om.with_greeks(chain)
    expiries = sorted(df["expiry"].unique())
    priced = [p for p in (_match_leg(df, expiries, leg, ev) for leg in legs) if p is not None]
    if ev.errors:
        return ev
    if len({p.expiry for p in priced}) > 1:
        ev.errors.append("the legs must share one expiry; calendars and diagonals are not supported")
        return ev

    ev.legs = priced
    ev.expiry = priced[0].expiry
    group = df[df["expiry"] == ev.expiry]
    summary = om.expiry_summary(group).iloc[0]
    ev.dte, ev.forward, ev.atm_iv = float(summary["dte"]), float(summary["forward"]), float(summary["atm_iv"])
    if ev.dte < min_dte:
        ev.errors.append(f"{_day(ev.expiry)} is {ev.dte:.1f} days out; use an expiry at least {min_dte:g} days away")
        return ev

    ev.net_fill = sum(p.sign * p.quantity * p.fill for p in priced)
    ev.net_mid = sum(p.sign * p.quantity * p.mark for p in priced)
    ev.fees = sum(p.quantity * _fee(chain.venue, ev.forward, p.fill) for p in priced)
    # Every figure at expiry is after the entry fees.
    cost = ev.net_fill + ev.fees
    # Entering and closing at once at today's quotes: both spreads crossed, fees twice.
    ev.round_trip_cost = sum(p.quantity * (p.ask - p.bid) for p in priced) + 2 * ev.fees
    points, pnl, slope = _payoff_profile(priced, cost)
    if slope < 0:
        ev.errors.append("the loss has no upper bound: more calls are sold than bought. "
                         "Only defined-risk structures are allowed")
        return ev
    ev.max_profit = math.inf if slope > 0 else float(pnl.max())
    ev.max_loss = max(0.0, -float(pnl.min()))
    ev.breakevens = _breakevens(points, pnl, slope)
    years = float(summary["years"])
    ev.pop = _probability_of_profit(priced, cost, ev.forward, ev.atm_iv, years)
    ev.expected_pnl_iv = _expected_pnl(priced, cost, ev.forward, ev.atm_iv, years)
    if realized_vol is not None and realized_vol > 0:
        ev.realized_vol = float(realized_vol)
        ev.expected_pnl_rv = _expected_pnl(priced, cost, ev.forward, ev.realized_vol, years)
        ev.pop_rv = _probability_of_profit(priced, cost, ev.forward, ev.realized_vol, years)
    ev.greeks = {
        name: float(sum(p.sign * p.quantity * getattr(p, name) for p in priced))
        for name in ("delta", "gamma", "vega", "theta")
    }
    ev.breaches = _limit_breaches(ev)
    return ev


def evaluation_record(ev: StructureEvaluation) -> dict:
    """The evaluation as plain data, for the run's state and a later mark-to-market."""
    def number(x):
        return None if x is None or (isinstance(x, float) and not math.isfinite(x)) else round(float(x), 6)

    return {
        "underlying": ev.underlying,
        "venue": ev.venue,
        "as_of": pd.Timestamp(ev.as_of).isoformat() if ev.as_of is not None else None,
        "expiry": _day(ev.expiry) if ev.expiry is not None else None,
        "legs": [{"instrument": p.instrument, "side": p.side, "type": p.kind, "strike": p.strike,
                  "expiry": _day(p.expiry), "quantity": p.quantity, "fill": number(p.fill),
                  "mark": number(p.mark)} for p in ev.legs],
        "net_fill": number(ev.net_fill),
        "net_mid": number(ev.net_mid),
        "max_profit": number(ev.max_profit),
        "max_profit_unbounded": ev.max_profit == math.inf,
        "max_loss": number(ev.max_loss),
        "breakevens": [number(b) for b in ev.breakevens],
        "probability_of_profit": number(ev.pop),
        "entry_fees": number(ev.fees),
        "round_trip_cost": number(ev.round_trip_cost),
        "expected_pnl_at_iv": number(ev.expected_pnl_iv),
        "realized_vol": number(ev.realized_vol),
        "expected_pnl_at_realized_vol": number(ev.expected_pnl_rv),
        "probability_of_profit_at_realized_vol": number(ev.pop_rv),
        "limit_breaches": list(ev.breaches),
        "errors": list(ev.errors),
    }


def is_tradeable(record: dict) -> bool:
    """Whether an evaluation record is a priced structure within the risk limits."""
    return bool(record.get("legs")) and not record.get("errors") and not record.get("limit_breaches")


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _money(value: float) -> str:
    return om._usd(value)


def render_evaluation(ev: StructureEvaluation) -> str:
    """The structure check the rest of the desk reads: the legs as filled and the payoff at expiry.

    Every figure is per structure as listed; how many to trade is the desk's call,
    made against a portfolio the agents do not see.
    """
    stamp = pd.Timestamp(ev.as_of).strftime("%Y-%m-%d %H:%M UTC") if ev.as_of is not None else "n/a"
    lines = [f"#### Structure check (computed from the {ev.venue} chain at {stamp})"]
    if ev.errors:
        lines.append("**Not priced.** The structure could not be checked against the chain:")
        lines += [f"- {e}" for e in ev.errors]
        lines += [f"- {n}" for n in ev.notes]
        return "\n".join(lines)

    unit = ev.underlying
    lines += [
        "",
        f"Expiry {_day(ev.expiry)} ({ev.dte:.1f} days), forward {_money(ev.forward)}, ATM IV {om._pct(ev.atm_iv)}. "
        f"Prices in USD per 1 {unit} of the quantities below; buys fill at the ask, sells at the bid.",
        "",
        "| Side | Contract | Qty | Bid | Ask | Mark | Fill | IV | Delta | Open interest |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for p in ev.legs:
        lines.append(
            f"| {p.side} | {p.instrument} | {p.quantity:g} | {om._price(p.bid)} | {om._price(p.ask)} "
            f"| {om._price(p.mark)} | {om._price(p.fill)} | {om._pct(p.iv)} | {om._num(p.delta)} "
            f"| {om._num(p.open_interest, 1)} |"
        )
    kind = "debit" if ev.net_fill >= 0 else "credit"
    profit = "unlimited" if ev.max_profit == math.inf else _money(ev.max_profit)
    reward = ("n/a" if ev.max_loss <= 0 or ev.max_profit == math.inf
              else f"{ev.max_profit / ev.max_loss:.2f}")
    g = ev.greeks
    fees = (f"; entry fees {_money(ev.fees)} at the {ev.venue} taker rate" if ev.fees > 0
            else "; no venue fees modelled")
    lines += [
        "",
        f"- Net {kind} at fill: {_money(abs(ev.net_fill))} (at mid: {_money(abs(ev.net_mid))}; "
        f"crossing the spread costs {_money(abs(ev.net_fill - ev.net_mid))}{fees}). "
        f"The figures at expiry below are after the entry fees.",
        f"- Max profit at expiry: {profit}",
        f"- Max loss at expiry: {_money(ev.max_loss)}",
        f"- Reward to risk: {reward}",
        f"- Breakevens at expiry: {', '.join(_money(b) for b in ev.breakevens) or 'none'}",
        f"- Probability of profit at expiry: {om._pct(ev.pop, 0)} (risk-neutral, lognormal at the "
        f"expiry's ATM IV; ignores skew, so it understates the tails the skew prices)",
        f"- Expected P&L at expiry at the ATM IV ({om._pct(ev.atm_iv)}): {_money(ev.expected_pnl_iv)}. "
        f"This is what the skew, the bid-ask and the fees cost against options priced flat at the ATM IV.",
    ]
    if math.isfinite(ev.expected_pnl_rv):
        lines.append(
            f"- Expected P&L at expiry at the 30-day realized volatility ({om._pct(ev.realized_vol)}): "
            f"{_money(ev.expected_pnl_rv)}, probability of profit {om._pct(ev.pop_rv, 0)}. This is the "
            f"structure's edge if the underlying keeps moving as it has; the difference from the line "
            f"above is what the volatility view is worth. Lognormal, so it understates crypto's jumps."
        )
    lines += [
        f"- Round trip at today's quotes (enter, then close at once): {_money(ev.round_trip_cost)} in "
        f"spreads and fees",
        f"- Net greeks now: delta {om._num(g['delta'])} {unit}, gamma {g['gamma']:.6f}, "
        f"vega {_money(g['vega'])} per vol point, theta {_money(g['theta'])} per day",
    ]
    if ev.notes:
        lines += ["", "Adjusted to listed contracts:"] + [f"- {n}" for n in ev.notes]
    lines.append("")
    if ev.breaches:
        lines += ["**Breaks the desk's risk limits; it cannot be approved as priced:**"]
        lines += [f"- {b}" for b in ev.breaches]
    else:
        lines.append(f"Risk limits: passed (every leg's bid-ask within {WIDE_SPREAD:.0%} of its mark and "
                     f"with open interest; probability of profit at least {MIN_POP:.0%} or reward to risk "
                     f"at least {MIN_REWARD_TO_RISK:g}).")
    if ev.venue in VENUE_TERMS:
        lines += ["", f"Venue terms ({ev.venue}): {VENUE_TERMS[ev.venue]}"]
    return "\n".join(lines)


def strike_guide(chain: om.OptionChain, min_dte: float = 7.0, max_dte: float = 90.0,
                 deltas: tuple[float, ...] = (0.10, 0.25, 0.40)) -> str:
    """Per expiry in range: the forward, ATM IV, implied move, and the strikes at given deltas."""
    df = om.with_greeks(chain)
    summary = om.expiry_summary(df)
    summary = summary[(summary["dte"] >= min_dte) & (summary["dte"] <= max_dte)]
    if summary.empty:
        return f"No {chain.underlying} expiries between {min_dte:g} and {max_dte:g} days out."

    def at_delta(group, kind, target):
        side = group[group["type"] == kind].dropna(subset=["delta"])
        if side.empty:
            return "n/a"
        row = side.loc[(side["delta"] - target).abs().idxmin()]
        return f"{om._usd(row['strike'])} ({om._usd(row['mark'])})"

    put_cols = [f"{d * 100:.0f}d put" for d in deltas]
    call_cols = [f"{d * 100:.0f}d call" for d in reversed(deltas)]
    lines = [
        f"Strike guide for {chain.underlying}: strike nearest each delta, with its mark in USD. "
        f"Any listed strike can be used; a strike that is not listed is moved to the nearest one.",
        "",
        "| Expiry | DTE | Forward | ATM IV | Implied move | " + " | ".join(put_cols + call_cols) + " |",
        "|" + "---|" * (5 + len(put_cols) + len(call_cols)),
    ]
    for row in summary.itertuples():
        group = df[df["expiry"] == row.expiry]
        cells = [at_delta(group, "put", -d) for d in deltas] + [at_delta(group, "call", d) for d in reversed(deltas)]
        lines.append(
            f"| {_day(row.expiry)} | {row.dte:.1f} | {om._usd(row.forward)} | {om._pct(row.atm_iv)} "
            f"| {om._pct(row.implied_move)} | " + " | ".join(cells) + " |"
        )
    return "\n".join(lines)
