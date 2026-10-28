"""What an options-mode agent needs to have its legs priced: the chain, the
underlying's realized volatility, and the legs in the pricer's form.

The strategist prices its proposal with these, and the Portfolio Manager an
adjustment of it.
"""

from __future__ import annotations

import math

import pandas as pd

from tradingagents.dataflows import options_metrics as om
from tradingagents.dataflows.options_structures import Leg, StructureEvaluation, evaluate_structure
from tradingagents.dataflows.router import route_to_vendor

# The window the structure's expected P&L at realized volatility uses.
REALIZED_VOL_WINDOW = 30


def load_chain(symbol: str, trade_date: str) -> om.OptionChain | str:
    """The live chain, or the router's sentinel string when there is none."""
    return route_to_vendor("get_option_chain_data", symbol, trade_date)


def load_realized_vol(symbol: str, trade_date: str) -> float | None:
    """The underlying's realized volatility, or None when its closes are unavailable."""
    closes = route_to_vendor("get_underlying_closes", symbol, trade_date)
    if not isinstance(closes, pd.Series):
        return None  # the router's sentinel: no closes, so no realized-vol line
    vol = om.realized_vol(closes, REALIZED_VOL_WINDOW)
    return vol if math.isfinite(vol) else None


def legs_of(specs) -> list[Leg]:
    """Schema leg specs (``OptionLegSpec``) as the pricer's legs."""
    return [Leg(leg.side, leg.option_type, leg.strike, leg.expiry, leg.quantity) for leg in specs]


def price(chain: om.OptionChain, specs, realized_vol: float | None) -> StructureEvaluation:
    return evaluate_structure(chain, legs_of(specs), realized_vol=realized_vol)
