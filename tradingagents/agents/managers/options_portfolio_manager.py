"""Portfolio Manager of an options-mode run: approves or rejects the priced structure.

It reads the research plan, the strategist's structure with its priced check, and
the risk debate, and returns a typed ``OptionsPortfolioDecision``. Its rating is
the run's ``final_rating``, so the memory log keeps scoring the directional call;
the verdict on the structure is added to ``options_trade``.
"""

from __future__ import annotations

from tradingagents.agents.context import (
    get_instrument_context_from_state,
    get_language_instruction,
    get_portfolio_context_from_state,
)
from tradingagents.agents.rating import parse_rating
from tradingagents.agents.schemas import OptionsPortfolioDecision, render_options_pm_decision
from tradingagents.agents.structured import NO_EXTERNAL_TOOLS, bind_structured, invoke_structured


def create_options_portfolio_manager(llm):
    structured_llm = bind_structured(llm, OptionsPortfolioDecision, "Portfolio Manager")

    def options_portfolio_manager_node(state) -> dict:
        risk_debate_state = state["risk_debate_state"]
        past_context = state.get("past_context", "")
        lessons_line = (
            f"- Lessons from prior decisions and outcomes:\n{past_context}\n" if past_context else ""
        )
        trade = dict(state.get("options_trade") or {})
        priced = bool(trade.get("legs")) and not trade.get("errors")

        prompt = f"""As the Portfolio Manager, decide on the options structure the strategist proposed, after the risk analysts' debate.

{get_instrument_context_from_state(state)}

{get_portfolio_context_from_state(state)}

---

**Rating Scale** for the underlying (use exactly one): Buy / Overweight / Hold / Underweight / Sell. A non-directional structure such as an iron condor or a straddle goes with Hold.

**Structure Decision**: Approve the structure as priced, or Reject it. {"The structure below was priced against the chain: judge it on those computed figures (fill, maximum loss and profit, breakevens, probability of profit, liquidity), not on figures of your own." if priced else "No priced structure reached you (a No trade, or one the chain could not price): Reject, and say in the thesis what would make a structure worth taking."}

Approve only when the structure fits the research plan's direction, volatility view and horizon, its maximum loss is worth its payoff, and its contracts are liquid enough to enter and exit. Set the risk budget as the maximum loss to accept, in percent of capital.

**Context:**
- Research plan: **{state["investment_plan"]}**
- Strategist's structure and its priced check: **{state["trader_investment_plan"]}**
{lessons_line}
**Risk Analysts Debate History:**
{risk_debate_state["history"]}

---

## Output

Write these sections, in this order, starting with the rating on its own line:

- **Rating**: exactly one of Buy / Overweight / Hold / Underweight / Sell
- **Structure Decision**: Approve or Reject
- **Risk Budget**: maximum loss as a percentage of capital, for an approval
- **Executive Summary**: the decision and how to act on it
- **Investment Thesis**: the evidence that decided it
- **Exit Plan**: profit target, loss limit, and when to close

{NO_EXTERNAL_TOOLS}{get_language_instruction()}"""

        decision = invoke_structured(structured_llm, prompt, "Portfolio Manager")
        if decision is not None:
            final_trade_decision = render_options_pm_decision(decision)
            final_rating = decision.rating.value
            trade.update({"decision": decision.structure_decision.value,
                          "approved_risk_budget_pct": decision.risk_budget_pct})
        else:
            final_trade_decision = llm.invoke(prompt).content
            final_rating = parse_rating(final_trade_decision)
            trade.update({"decision": "unparsed"})

        return {
            "risk_debate_state": {**risk_debate_state, "latest_speaker": "Judge"},
            "final_trade_decision": final_trade_decision,
            "final_rating": final_rating,
            "options_trade": trade,
        }

    return options_portfolio_manager_node
