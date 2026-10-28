"""Options Strategist: the Trader of an options-mode run.

It turns the research plan's direction and volatility view into an options
structure. The model names the legs; code matches them to the listed chain and
prices them (fill, fees, maximum profit and loss, breakevens, probability of
profit, expected P&L at implied and realized volatility), and a structure the
chain cannot price, or that breaks the desk's risk limits, goes back to the model
once, with the reasons, to be fixed.

It proposes one structure, not a size: how many to trade is the desk's call.
"""

from __future__ import annotations

import re
from datetime import timedelta

import pandas as pd
from langchain_core.messages import AIMessage

from tradingagents.agents.context import (
    NO_SIZING,
    get_instrument_context_from_state,
    get_language_instruction,
    get_portfolio_context_from_state,
)
from tradingagents.agents.options_pricing import load_chain, load_realized_vol, price
from tradingagents.agents.schemas import (
    OptionsStrategy,
    OptionsTradeProposal,
    render_options_proposal,
)
from tradingagents.agents.structured import NO_EXTERNAL_TOOLS, bind_structured, invoke_structured
from tradingagents.dataflows import options_metrics as om
from tradingagents.dataflows.options_structures import (
    MAX_LEGS,
    MIN_POP,
    MIN_REWARD_TO_RISK,
    WIDE_SPREAD,
    evaluation_record,
    render_evaluation,
    strike_guide,
)

MIN_DTE, MAX_DTE = 7, 90

# The plan's horizon when it gives none.
DEFAULT_HORIZON_DAYS = 30

RISK_LIMITS = (
    "The desk's risk limits, checked in code: every leg's bid-ask spread within "
    f"{WIDE_SPREAD:.0%} of its mark and with open interest; a probability of profit of at "
    f"least {MIN_POP:.0%} or a reward to risk of at least {MIN_REWARD_TO_RISK:g}."
)

STRUCTURE_MATRIX = """How the two views map to structures (defined risk only, all legs on one expiry):

| Direction | Long volatility (options cheap) | Neutral volatility | Short volatility (options rich) |
|---|---|---|---|
| Bullish (Buy / Overweight) | Long call, or bull call spread | Bull call spread | Bull put spread, or cash-secured put |
| Bearish (Sell / Underweight) | Long put, or bear put spread | Bear put spread | Bear call spread |
| Neutral (Hold) | Long straddle or long strangle | No trade | Iron condor or iron butterfly |

A Hold with no volatility edge is a No trade, not a structure for its own sake."""


def _horizon_days(plan: str) -> int:
    """The research plan's horizon, read off its rendered ``**Horizon**`` line."""
    match = re.search(r"\*\*Horizon\*\*:\s*(\d+)", plan or "")
    return int(match.group(1)) if match else DEFAULT_HORIZON_DAYS


def chain_view(chain, horizon_days: int) -> str:
    """The strike guide, and the full ladder of the expiry nearest the horizon."""
    target = pd.Timestamp(chain.as_of) + timedelta(days=min(max(horizon_days, MIN_DTE), MAX_DTE))
    ladder = om.format_chain(chain, expiry=target.strftime("%Y-%m-%d"), moneyness_pct=25.0, max_strikes=30)
    return (f"{strike_guide(chain, MIN_DTE, MAX_DTE)}\n\n"
            f"Chain ladder for the expiry nearest the plan's {horizon_days}-day horizon:\n{ladder}")


def _no_structure(reason: str) -> dict:
    plan = f"**Strategy**: {OptionsStrategy.NO_TRADE.value}\n\n**Rationale**: {reason}"
    return {"messages": [AIMessage(content=plan)], "trader_investment_plan": plan,
            "options_trade": {"strategy": OptionsStrategy.NO_TRADE.value, "reason": reason}}


def create_options_strategist(llm):
    structured_llm = bind_structured(llm, OptionsTradeProposal, "Options Strategist")

    def options_strategist_node(state):
        symbol = state["company_of_interest"]
        chain = load_chain(symbol, state["trade_date"])
        if isinstance(chain, str):
            # The router's sentinel: no chain, so nothing can be proposed or priced.
            return _no_structure(f"No option chain is available for this run, so no structure "
                                 f"can be proposed or priced. {chain}")

        realized_vol = load_realized_vol(symbol, state["trade_date"])
        view = chain_view(chain, _horizon_days(state["investment_plan"]))
        options_report = (state.get("options_report") or "").strip() or "(No options report in this run.)"
        market_report = (state.get("market_report") or "").strip() or "(No market report in this run.)"
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an options strategist. Turn the research plan's direction and "
                    "volatility view into one options structure on the instrument, or No trade. "
                    f"Use one to {MAX_LEGS} legs, all on the same listed expiry between {MIN_DTE} and "
                    f"{MAX_DTE} days out, matching the plan's horizon. Defined risk only: never sell "
                    "more calls than you buy. Take strikes and expiries from the strike guide; "
                    "prefer expiries and strikes with open interest, and use the chain ladder to "
                    "weigh strike width: a wider spread collects or risks more, a narrower one "
                    "pays the bid-ask on less. Do not compute prices, payoffs or probabilities: "
                    "the structure is priced in code against the chain after you answer, and that "
                    "check is what the desk reads.\n\n"
                    f"{RISK_LIMITS} {NO_SIZING}\n\n"
                    + STRUCTURE_MATRIX + "\n\n"
                    + NO_EXTERNAL_TOOLS
                    + get_language_instruction()
                ),
            },
            {
                "role": "user",
                "content": (
                    f"{get_instrument_context_from_state(state)}\n\n"
                    f"{get_portfolio_context_from_state(state)}\n\n"
                    f"Research plan:\n{state['investment_plan']}\n\n"
                    f"Options market report:\n{options_report}\n\n"
                    f"Technical market report:\n{market_report}\n\n"
                    f"{view}\n\n"
                    "Propose the structure."
                ),
            },
        ]

        proposal = invoke_structured(structured_llm, messages, "Options Strategist")
        if proposal is None:
            text = llm.invoke(messages).content
            plan = (f"{text}\n\n**Not priced**: the proposal did not come back in a form the "
                    "chain can be checked against, so it has no computed risk figures.")
            return {"messages": [AIMessage(content=plan)], "trader_investment_plan": plan,
                    "options_trade": {"strategy": "unparsed"}, "options_chain_view": view}

        evaluation = None
        if proposal.strategy != OptionsStrategy.NO_TRADE:
            evaluation = price(chain, proposal.legs, realized_vol)
            if not evaluation.tradeable:
                # One chance to fix it, with the reasons the chain or the limits gave.
                problem = ("could not be priced against the chain" if evaluation.errors
                           else "breaks the desk's risk limits")
                retry = [
                    *messages,
                    {"role": "assistant", "content": render_options_proposal(proposal)},
                    {"role": "user", "content": (
                        f"That structure {problem}:\n"
                        + "\n".join(f"- {e}" for e in evaluation.errors or evaluation.breaches)
                        + "\n\nFix it, or propose No trade, and give the whole proposal again."
                    )},
                ]
                second = invoke_structured(structured_llm, retry, "Options Strategist")
                if second is not None:
                    proposal = second
                    evaluation = (price(chain, proposal.legs, realized_vol)
                                  if proposal.strategy != OptionsStrategy.NO_TRADE else None)

        plan = render_options_proposal(proposal)
        trade = {"strategy": proposal.strategy.value}
        if evaluation is not None:
            plan += "\n\n" + render_evaluation(evaluation)
            trade.update(evaluation_record(evaluation))
        return {"messages": [AIMessage(content=plan)], "trader_investment_plan": plan,
                "options_trade": trade, "options_chain_view": view}

    return options_strategist_node
