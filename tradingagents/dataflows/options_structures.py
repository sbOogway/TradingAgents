"""Option structures: proposed legs checked against the chain and priced in code.

The strategist names the legs; this module decides whether they can be traded and
what they are worth. Each leg is matched to a listed contract and priced on the
side of the book it would fill on (buy at the ask, sell at the bid). The payoff at
expiry then gives the maximum profit and loss, the breakevens and a probability of
profit, so no model does the arithmetic.

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

# A leg whose bid-ask spread is wider than this share of its mark is flagged.
WIDE_SPREAD = 0.25


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
    greeks: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors and bool(self.legs)


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


def _probability_of_profit(legs, cost, forward, iv, years) -> float:
    """Risk-neutral chance the structure ends in profit: lognormal at the forward and ATM IV."""
    if not (forward > 0 and iv > 0 and years > 0):
        return math.nan
    z = np.linspace(-8.0, 8.0, 8001)
    spot = forward * np.exp(-0.5 * iv * iv * years + iv * math.sqrt(years) * z)
    pnl = sum(leg.sign * leg.quantity * _intrinsic(leg.kind, leg.strike, spot) for leg in legs) - cost
    weight = np.exp(-0.5 * z * z)
    return float(weight[pnl > 0].sum() / weight.sum())


def evaluate_structure(chain: om.OptionChain, legs: list[Leg], min_dte: float = MIN_DTE) -> StructureEvaluation:
    """Match the legs to listed contracts and price the structure at expiry."""
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
    points, pnl, slope = _payoff_profile(priced, ev.net_fill)
    if slope < 0:
        ev.errors.append("the loss has no upper bound: more calls are sold than bought. "
                         "Only defined-risk structures are allowed")
        return ev
    ev.max_profit = math.inf if slope > 0 else float(pnl.max())
    ev.max_loss = max(0.0, -float(pnl.min()))
    ev.breakevens = _breakevens(points, pnl, slope)
    ev.pop = _probability_of_profit(priced, ev.net_fill, ev.forward, ev.atm_iv, float(summary["years"]))
    ev.greeks = {
        name: float(sum(p.sign * p.quantity * getattr(p, name) for p in priced))
        for name in ("delta", "gamma", "vega", "theta")
    }

    for p in priced:
        if p.mark > 0 and (p.ask - p.bid) / p.mark > WIDE_SPREAD:
            ev.warnings.append(f"{p.instrument}: bid-ask spread is {(p.ask - p.bid) / p.mark:.0%} of its mark")
        if p.open_interest <= 0:
            ev.warnings.append(f"{p.instrument}: no open interest")
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
        "errors": list(ev.errors),
    }


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _money(value: float) -> str:
    return om._usd(value)


def render_evaluation(ev: StructureEvaluation) -> str:
    """The structure check the rest of the desk reads: the legs as filled and the payoff at expiry."""
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
    lines += [
        "",
        f"- Net {kind} at fill: {_money(abs(ev.net_fill))} (at mid: {_money(abs(ev.net_mid))}; "
        f"crossing the spread costs {_money(abs(ev.net_fill - ev.net_mid))})",
        f"- Max profit at expiry: {profit}",
        f"- Max loss at expiry: {_money(ev.max_loss)}",
        f"- Reward to risk: {reward}",
        f"- Breakevens at expiry: {', '.join(_money(b) for b in ev.breakevens) or 'none'}",
        f"- Probability of profit at expiry: {om._pct(ev.pop, 0)} (risk-neutral, lognormal at the "
        f"expiry's ATM IV; ignores skew, so it understates the tails the skew prices)",
        f"- Net greeks now: delta {om._num(g['delta'])} {unit}, gamma {g['gamma']:.6f}, "
        f"vega {_money(g['vega'])} per vol point, theta {_money(g['theta'])} per day",
    ]
    if ev.notes:
        lines += ["", "Adjusted to listed contracts:"] + [f"- {n}" for n in ev.notes]
    if ev.warnings:
        lines += ["", "Liquidity warnings:"] + [f"- {w}" for w in ev.warnings]
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
