"""The risk debate's three roles in options mode.

Aggressive, conservative and neutral are stances on how much risk to take, which
fits a position to size. An options structure is judged on other questions: is
its edge real, what breaks it, and would other legs carry the same view better.
So in options mode the three debate seats argue those instead. The seats, their
order and the routing stay the same; only the brief and the speaker label change.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from tradingagents.agents.context import (
    NO_SIZING,
    get_instrument_context_from_state,
    get_language_instruction,
    opponent_argument_or_opening,
    report_or_absent,
)


@dataclass(frozen=True)
class OptionsRole:
    name: str
    brief: str


OPTIONS_ROLES = {
    "aggressive": OptionsRole(
        "Trade Advocate",
        "You make the case for the structure as priced. Show how it expresses the research plan's "
        "direction and volatility view, and where its edge is: the expected P&L at realized "
        "volatility against the expected P&L at the ATM IV, what theta earns or costs over the "
        "horizon, and how vega lines up with the volatility view. Answer the Tail-Risk Sceptic's "
        "and the Structurer's points with the check's figures, and concede the ones that are right.",
    ),
    "conservative": OptionsRole(
        "Tail-Risk Sceptic",
        "You look for what breaks the structure. A jump through the strikes, which the lognormal "
        "figures understate and the skew prices; gamma as expiry nears; a rise in implied "
        "volatility against its vega; events before expiry from the news; and the round-trip cost "
        "of getting out early against the edge the check shows. For BTC and ETH options, remember "
        "the P&L is settled in the coin. Quantify from the check where it can be, and say what "
        "would make the risk acceptable rather than only that it exists.",
    ),
    "neutral": OptionsRole(
        "Structurer",
        "You judge whether other legs would carry the same plan better: other listed strikes, a "
        "narrower or wider spread, another expiry nearer the horizon or clear of an event, or "
        "another structure from the plan's row of the structure matrix. Name concrete listed "
        "contracts from the chain below, and say what each change gives up and gains. If the "
        "proposed structure is the best of them, say so. The Portfolio Manager can adjust the "
        "structure, and any adjustment is priced in code before it is approved.",
    ),
}

_OTHERS = {"aggressive": ("conservative", "neutral"),
           "conservative": ("aggressive", "neutral"),
           "neutral": ("aggressive", "conservative")}


def options_role_prompt(seat: str, state: Mapping[str, Any]) -> str:
    """The prompt for the debate seat ``seat`` ("aggressive", "conservative", "neutral")."""
    role = OPTIONS_ROLES[seat]
    risk = state["risk_debate_state"]
    others = "\n".join(
        f"Last argument from the {OPTIONS_ROLES[o].name}: "
        + opponent_argument_or_opening(risk.get(f"current_{o}_response", ""), OPTIONS_ROLES[o].name)
        for o in _OTHERS[seat]
    )
    view = (state.get("options_chain_view") or "").strip()
    chain = f"\nChain the structure was chosen from:\n{view}\n" if seat == "neutral" and view else ""
    return f"""You are the {role.name} in the risk debate on an options structure. {role.brief}

Argue from the structure check's computed figures, not figures of your own: it was priced in code against the chain. {NO_SIZING}

The strategist's structure and its check:
{state["trader_investment_plan"]}

{get_instrument_context_from_state(state)}
Research plan: {state.get("investment_plan", "")}
Options market report: {report_or_absent(state.get("options_report", ""), "options")}
Market research report: {report_or_absent(state.get("market_report", ""), "market")}
News report: {report_or_absent(state.get("news_report", ""), "news")}
{chain}
Debate so far: {risk.get("history", "")}
{others}

Respond to the other two directly where they have spoken. Output conversationally, without special formatting.""" + get_language_instruction()
