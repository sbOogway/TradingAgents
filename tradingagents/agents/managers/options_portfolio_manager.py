"""Portfolio Manager of an options-mode run: approves or rejects the priced structure.

It reads the research plan, the strategist's structure with its priced check, and
the risk debate, and returns a typed ``OptionsPortfolioDecision``. Its rating is
the run's ``final_rating``, so the memory log keeps scoring the directional call;
the verdict on the structure is added to ``options_trade``.

Two rules are enforced in code, whatever the model answers: a structure that was
not priced, or breaks the desk's risk limits, is rejected; and an approved risk
budget never exceeds the desk's limit.
"""

from __future__ import annotations

import math

from tradingagents.agents.context import (
    get_instrument_context_from_state,
    get_language_instruction,
    get_portfolio_context_from_state,
    options_budget_line,
    options_risk_budget_pct,
)
from tradingagents.agents.rating import parse_rating
from tradingagents.agents.schemas import (
    OptionsPortfolioDecision,
    StructureDecision,
    render_options_pm_decision,
)
from tradingagents.agents.structured import NO_EXTERNAL_TOOLS, bind_structured, invoke_structured
from tradingagents.dataflows.options_structures import (
    capital_per_structure,
    is_tradeable,
    size_line,
)

JUDGING_BRIEF = (
    "Judge it on the check's computed figures, not figures of your own. Reward to risk alone does "
    "not decide a structure: a credit spread that wins often always risks more than it makes. The "
    "test is whether its expected P&L holds up: at the ATM IV it shows what the skew, the bid-ask and "
    "the fees cost; at realized volatility it shows the edge if the plan's volatility view is right. "
    "Weigh that edge against the jump risk the lognormal leaves out and the round-trip cost of an "
    "early exit. The risk limits are already checked in code, and the venue terms in the check are "
    "known, not open questions. Live order-book depth and the caller's account apply to every trade: "
    "make them conditions in the exit plan, not reasons to reject."
)


def create_options_portfolio_manager(llm):
    structured_llm = bind_structured(llm, OptionsPortfolioDecision, "Portfolio Manager")

    def options_portfolio_manager_node(state) -> dict:
        risk_debate_state = state["risk_debate_state"]
        past_context = state.get("past_context", "")
        lessons_line = (
            f"- Lessons from prior decisions and outcomes:\n{past_context}\n" if past_context else ""
        )
        trade = dict(state.get("options_trade") or {})
        priced = is_tradeable(trade)
        limit = options_risk_budget_pct()

        prompt = f"""As the Portfolio Manager, decide on the options structure the strategist proposed, after the risk analysts' debate.

{get_instrument_context_from_state(state)}

{get_portfolio_context_from_state(state)}

---

**Rating Scale** for the underlying (use exactly one): Buy / Overweight / Hold / Underweight / Sell. A non-directional structure such as an iron condor or a straddle goes with Hold.

**Structure Decision**: Approve the structure as priced, or Reject it. {"The structure below was priced against the chain and is within the desk's risk limits. " + JUDGING_BRIEF if priced else "No structure within the desk's risk limits reached you (a No trade, one the chain could not price, or one that breaks a limit): Reject, and say in the thesis what would make a structure worth taking."}

Approve when the structure fits the research plan's direction, volatility view and horizon and its expected P&L justifies its maximum loss. {options_budget_line()} Set the risk budget as the maximum loss to accept, in percent of capital, at most {limit:g}.

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
            decision = _enforce_limits(decision, priced, trade.get("risk_budget_pct"), limit)
            final_trade_decision = render_options_pm_decision(decision)
            final_rating = decision.rating.value
            trade.update({"decision": decision.structure_decision.value,
                          "approved_risk_budget_pct": decision.risk_budget_pct})
            if decision.structure_decision == StructureDecision.APPROVE:
                max_loss = trade.get("max_loss") or math.nan
                size = size_line(max_loss, decision.risk_budget_pct, trade.get("underlying", "unit"))
                if size:
                    trade["capital_per_structure"] = round(capital_per_structure(max_loss, decision.risk_budget_pct), 2)
                    final_trade_decision += f"\n\n**{size.replace(': ', '**: ', 1)}"
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


def _enforce_limits(decision: OptionsPortfolioDecision, priced: bool, proposed: float | None,
                    limit: float) -> OptionsPortfolioDecision:
    """The decision with the desk's two hard rules applied: no approval of a structure
    outside the limits, and no budget above the limit."""
    if decision.structure_decision == StructureDecision.APPROVE and not priced:
        return decision.model_copy(update={
            "structure_decision": StructureDecision.REJECT, "risk_budget_pct": None,
            "executive_summary": ("Rejected: no structure within the desk's risk limits was priced, so "
                                  "none can be approved. " + decision.executive_summary),
        })
    if decision.structure_decision == StructureDecision.APPROVE:
        budget = decision.risk_budget_pct if decision.risk_budget_pct is not None else proposed
        budget = limit if budget is None or budget <= 0 else min(budget, limit)
        return decision.model_copy(update={"risk_budget_pct": budget})
    return decision
