"""Research Manager: turns the bull/bear debate into a structured investment plan for the trader.

In options mode the plan also carries a volatility view and a horizon, which the
options strategist needs to pick a structure.
"""

from __future__ import annotations

from tradingagents.agents.context import (
    get_instrument_context_from_state,
    get_language_instruction,
    options_report_line,
)
from tradingagents.agents.schemas import (
    OptionsResearchPlan,
    ResearchPlan,
    render_options_research_plan,
    render_research_plan,
)
from tradingagents.agents.structured import (
    NO_EXTERNAL_TOOLS,
    bind_structured,
    invoke_structured_or_freetext,
)

VOLATILITY_BRIEF = """

---

**Volatility View** (options mode, use exactly one): Long volatility / Neutral volatility / Short volatility. Judge whether the options market prices more or less movement than the evidence supports: implied against realized volatility, IV rank, the term structure, the skew and any expected event. The direction and the volatility view are separate calls; make both. Give the horizon in days over which the view should play out.

{options_line}"""


def create_research_manager(llm, options_mode: bool = False):
    schema, render = ((OptionsResearchPlan, render_options_research_plan) if options_mode
                      else (ResearchPlan, render_research_plan))
    structured_llm = bind_structured(llm, schema, "Research Manager")

    def research_manager_node(state) -> dict:
        instrument_context = get_instrument_context_from_state(state)
        history = state["investment_debate_state"].get("history", "")

        investment_debate_state = state["investment_debate_state"]
        volatility_brief = (
            VOLATILITY_BRIEF.format(options_line=options_report_line(state)) if options_mode else ""
        )
        volatility_outputs = (
            "- **Volatility View**: exactly one of Long volatility / Neutral volatility / Short volatility\n"
            "- **Volatility Rationale**: the evidence for it\n"
            "- **Horizon**: days for the view to play out\n"
            if options_mode else ""
        )

        prompt = f"""As the Research Manager and debate facilitator, your role is to critically evaluate this round of debate and deliver a clear, actionable investment plan for the trader.

{instrument_context}

---

**Rating Scale** (use exactly one):
- **Buy**: Strong conviction in the bull thesis; recommend taking or growing the position
- **Overweight**: Constructive view; recommend gradually increasing exposure
- **Hold**: Balanced view; recommend maintaining the current position
- **Underweight**: Cautious view; recommend trimming exposure
- **Sell**: Strong conviction in the bear thesis; recommend exiting or avoiding the position

The debate always contains conflicting arguments; deciding which side is stronger is the job, so conflict alone is not a reason to Hold. Commit to the side with the stronger case, sized by how decisively it wins. Choose Hold only when the evidence is still balanced after that weighing, or too thin to support a call; do not manufacture a direction to appear decisive. Weigh the bull and bear cases on their merits, independent of which side spoke first or last.

---

**Debate History:**
{history}
{volatility_brief}
## Output

Write these sections, in this order, starting with the recommendation on its own line:

- **Recommendation**: exactly one of Buy / Overweight / Hold / Underweight / Sell
- **Rationale**: which arguments decided it
- **Strategic Actions**: concrete steps for the trader, sized against a standard allocation
{volatility_outputs}
{NO_EXTERNAL_TOOLS}""" + get_language_instruction()

        investment_plan = invoke_structured_or_freetext(
            structured_llm,
            llm,
            prompt,
            render,
            "Research Manager",
        )

        new_investment_debate_state = {
            "history": investment_debate_state.get("history", ""),
            "bear_history": investment_debate_state.get("bear_history", ""),
            "bull_history": investment_debate_state.get("bull_history", ""),
            "current_response": investment_plan,
            "count": investment_debate_state["count"],
        }

        return {
            "investment_debate_state": new_investment_debate_state,
            "investment_plan": investment_plan,
        }

    return research_manager_node
