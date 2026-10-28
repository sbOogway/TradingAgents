"""Portfolio Manager of an options-mode run: approves, adjusts or rejects the priced structure.

It reads the research plan, the strategist's structure with its priced check, and
the risk debate, and returns a typed ``OptionsPortfolioDecision``. Its rating is
the run's ``final_rating``, so the memory log keeps scoring the directional call;
the verdict on the structure is added to ``options_trade``.

An adjustment is priced against the chain like the strategist's proposal, and
goes back to the Portfolio Manager with its check, to approve or reject; nothing
is approved unpriced. A structure that was not priced, or breaks the desk's risk
limits, is rejected whatever the model answers. The decision carries no size:
the desk sizes trades against a portfolio the agents do not see.
"""

from __future__ import annotations

from tradingagents.agents.context import (
    NO_SIZING,
    get_instrument_context_from_state,
    get_language_instruction,
    get_portfolio_context_from_state,
)
from tradingagents.agents.options_pricing import load_chain, load_realized_vol, price
from tradingagents.agents.rating import parse_rating
from tradingagents.agents.schemas import (
    OptionsPortfolioDecision,
    StructureDecision,
    render_options_pm_decision,
)
from tradingagents.agents.structured import NO_EXTERNAL_TOOLS, bind_structured, invoke_structured
from tradingagents.dataflows.options_structures import (
    evaluation_record,
    is_tradeable,
    render_evaluation,
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

ADJUST_BRIEF = (
    "Adjust when other listed strikes, a different width or another expiry would express the same "
    "plan better, for instance as the Structurer argued: give the adjusted strategy and all its legs, "
    "from the chain below. The adjustment is priced against the chain and shown to you before "
    "anything is approved."
)


def _reject(decision: OptionsPortfolioDecision, reason: str) -> OptionsPortfolioDecision:
    return decision.model_copy(update={
        "structure_decision": StructureDecision.REJECT,
        "executive_summary": f"Rejected: {reason} {decision.executive_summary}",
    })


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
        view = (state.get("options_chain_view") or "").strip()

        if priced:
            choice = ("Approve the structure as priced, Adjust it, or Reject it. The structure below "
                      "was priced against the chain and is within the desk's risk limits. " + JUDGING_BRIEF)
        elif view:
            choice = ("No structure within the desk's risk limits reached you (a No trade, one the chain "
                      "could not price, or one that breaks a limit), so it cannot be approved as it is. "
                      "Reject, or Adjust to a structure from the same plan that fits the limits.")
        else:
            choice = ("No option chain was available, so no structure can be priced: Reject, and say in "
                      "the thesis what would make a structure worth taking.")
        chain_block = f"\n**Chain the structure was chosen from:**\n{view}\n" if view else ""

        prompt = f"""As the Portfolio Manager, decide on the options structure the strategist proposed, after the risk analysts' debate.

{get_instrument_context_from_state(state)}

{get_portfolio_context_from_state(state)}

---

**Rating Scale** for the underlying (use exactly one): Buy / Overweight / Hold / Underweight / Sell. A non-directional structure such as an iron condor or a straddle goes with Hold.

**Structure Decision**: {choice}

Approve when the structure fits the research plan's direction, volatility view and horizon and its expected P&L justifies its maximum loss. {ADJUST_BRIEF if view else ""} {NO_SIZING}

**Context:**
- Research plan: **{state["investment_plan"]}**
- Strategist's structure and its priced check: **{state["trader_investment_plan"]}**
{lessons_line}{chain_block}
**Risk Analysts Debate History:**
{risk_debate_state["history"]}

---

## Output

Write these sections, in this order, starting with the rating on its own line:

- **Rating**: exactly one of Buy / Overweight / Hold / Underweight / Sell
- **Structure Decision**: Approve, Adjust (with the adjusted strategy and legs) or Reject
- **Executive Summary**: the decision and how to act on it
- **Investment Thesis**: the evidence that decided it
- **Exit Plan**: profit target, loss limit, and when to close

{NO_EXTERNAL_TOOLS}{get_language_instruction()}"""

        decision = invoke_structured(structured_llm, prompt, "Portfolio Manager")
        if decision is None:
            final_trade_decision = llm.invoke(prompt).content
            trade.update({"decision": "unparsed"})
            return {
                "risk_debate_state": {**risk_debate_state, "latest_speaker": "Judge"},
                "final_trade_decision": final_trade_decision,
                "final_rating": parse_rating(final_trade_decision),
                "options_trade": trade,
            }

        check = ""
        if decision.structure_decision == StructureDecision.ADJUST:
            decision, trade, check = _adjust(structured_llm, prompt, decision, state, trade)
        elif decision.structure_decision == StructureDecision.APPROVE and not priced:
            decision = _reject(decision, "no structure within the desk's risk limits was priced, so none "
                                         "can be approved.")

        trade["decision"] = decision.structure_decision.value
        final_trade_decision = render_options_pm_decision(decision)
        if check:
            final_trade_decision += "\n\n" + check
        return {
            "risk_debate_state": {**risk_debate_state, "latest_speaker": "Judge"},
            "final_trade_decision": final_trade_decision,
            "final_rating": decision.rating.value,
            "options_trade": trade,
        }

    return options_portfolio_manager_node


def _adjust(structured_llm, prompt: str, decision: OptionsPortfolioDecision, state, trade: dict):
    """Price the adjustment and put it back to the Portfolio Manager to approve or reject.

    Returns the final decision, the trade record, and the adjustment's structure
    check for the decision text ("" when it could not be priced at all).
    """
    if not decision.adjusted_legs:
        return _reject(decision, "the adjustment named no legs, so there is nothing to price."), trade, ""
    symbol, trade_date = state["company_of_interest"], state["trade_date"]
    chain = load_chain(symbol, trade_date)
    if isinstance(chain, str):
        return _reject(decision, "the adjustment could not be priced: no option chain is available."), trade, ""

    evaluation = price(chain, decision.adjusted_legs, load_realized_vol(symbol, trade_date))
    check = render_evaluation(evaluation)
    strategy = decision.adjusted_strategy.value if decision.adjusted_strategy else trade.get("strategy")
    record = {"strategy": strategy, **evaluation_record(evaluation)}
    if not evaluation.tradeable:
        why = "could not be priced" if evaluation.errors else "breaks the desk's risk limits"
        rejected = _reject(decision, f"the adjusted structure {why}, and the original cannot be approved "
                                     "as it stands.")
        return rejected, {**trade, "adjustment": record}, check

    confirm = (f"{prompt}\n\n---\n\nYou adjusted the structure to: {strategy}. It was priced against "
               f"the chain:\n\n{check}\n\nApprove this adjusted structure as priced, or Reject it. A second "
               f"adjustment is not possible.")
    final = invoke_structured(structured_llm, confirm, "Portfolio Manager")
    adjusted = {"adjusted_strategy": decision.adjusted_strategy, "adjusted_legs": decision.adjusted_legs}
    if final is None or final.structure_decision != StructureDecision.APPROVE:
        final = _reject(final or decision, "the adjusted structure was not confirmed once priced.")
        return final.model_copy(update=adjusted), {**trade, "adjustment": record}, check

    original = {"strategy": trade.get("strategy"), "legs": trade.get("legs", [])}
    return final.model_copy(update=adjusted), {**record, "adjusted_from": original}, check
