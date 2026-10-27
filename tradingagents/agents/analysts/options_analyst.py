from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tradingagents.agents.analysts.turn import take_turn
from tradingagents.agents.context import get_instrument_context_from_state, get_language_instruction
from tradingagents.agents.tools import get_option_chain, get_options_snapshot

# The tools this analyst is offered; its tool node is built from the same tuple.
TOOLS = (
    get_options_snapshot,
    get_option_chain,
)


def create_options_analyst(llm):

    def options_analyst_node(state):
        current_date = state["trade_date"]
        instrument_context = get_instrument_context_from_state(state)

        system_message = (
            """You are a derivatives analyst reading the listed options market for the instrument under analysis. Your report tells the rest of the desk what the options market is pricing and how it is positioned, so that a directional view can be turned into the right options structure later.

Call get_options_snapshot first for the current date. If an expiry stands out (a kink in the term structure, an outsized implied move, heavy open interest or unusual activity), call get_option_chain for that expiry to look at its strikes; at most two such calls.

Every number in the snapshot and chain is computed in code from the venue's data. Quote figures only from tool output and do not compute new ones (no probabilities, no payoffs, no Greeks the tools did not give). If a tool says data is withheld or unavailable, say so plainly and write the report from what remains; do not estimate the missing values.

Cover, with the figures that support each point:
- Volatility level: 30-day implied volatility against realized volatility, and IV rank / percentile where given. Is option premium rich or cheap relative to how the underlying has actually moved, and relative to its own history?
- Term structure: upward sloping (contango) or inverted, and any expiry whose IV or implied move stands out, which usually marks an expected event.
- Skew: what the 25-delta risk reversal and butterfly say about demand for downside protection versus upside calls, and how that changes across expiries.
- Expected moves: the move the straddle prices to each near expiry, in percent and in price terms around the forward.
- Positioning: put/call ratios, the strikes with the most open interest, max pain for the large near expiry, and the most active contracts. Open interest does not show who is long or short; say what the data does and does not show.
- Implications for structures: whether the volatility picture favours buying premium (long options, debit spreads) or selling it (credit spreads, covered calls, cash-secured puts), and which expiries look best suited, conditional on a bullish, bearish or neutral view. Do not pick the direction yourself: another agent decides the trade."""
            + """ Make sure to append a Markdown table at the end of the report to organize key points in the report, organized and easy to read."""
            + get_language_instruction()
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " Use the provided tools to progress towards answering the question."
                    " If you are unable to fully answer, that's OK; another assistant with different tools"
                    " will help where you left off. Execute what you can to make progress."
                    " Report what your tools support; another agent decides the trade."
                    " You have access to the following tools: {tool_names}."
                    " Today's date is {current_date}; treat it as 'now' for all analysis and tool-call date ranges. {instrument_context}\n"
                    "{system_message}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in TOOLS]))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(instrument_context=instrument_context)

        result, report = take_turn(prompt, llm, TOOLS, state["messages"])

        return {
            "messages": [result],
            "options_report": report,
        }

    return options_analyst_node
