"""Options mode: structures priced in code, the strategist's propose-check-fix loop,
the Portfolio Manager's verdict, and the graph and CLI wiring.

The chain is synthetic and every model is faked, so these run offline.
"""

import math
from unittest import mock

import pytest
from langchain_core.messages import AIMessage

from tests.test_options_data import AS_OF, FORWARD, _chain
from tradingagents.agents.schemas import (
    OptionsPortfolioDecision,
    OptionsResearchPlan,
    OptionsTradeProposal,
    render_options_research_plan,
)
from tradingagents.dataflows import options_structures as st

EXPIRY = "2026-11-26"  # _chain's 30-day expiry, (AS_OF + 30 days)


def _ev(*legs, days=(10, 30)):
    return st.evaluate_structure(_chain(expiry_days=days), [st.Leg(*leg) for leg in legs])


# ---------------------------------------------------------------------------
# Pricing structures
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_a_debit_spread_risks_its_debit_and_makes_the_width_less_it():
    ev = _ev(("buy", "call", 80_000, EXPIRY), ("sell", "call", 90_000, EXPIRY))
    assert ev.ok, ev.errors
    long_leg, short_leg = ev.legs
    debit = long_leg.ask - short_leg.bid
    assert ev.net_fill == pytest.approx(debit)
    assert ev.max_loss == pytest.approx(debit)
    assert ev.max_profit == pytest.approx(10_000 - debit)
    assert ev.breakevens == [pytest.approx(80_000 + debit)]
    assert 0 < ev.pop < 0.5
    assert ev.net_mid < ev.net_fill  # crossing the spread costs more than mid


@pytest.mark.unit
def test_a_credit_spread_risks_the_width_less_its_credit():
    ev = _ev(("sell", "put", 75_000, EXPIRY), ("buy", "put", 70_000, EXPIRY))
    credit = -ev.net_fill
    assert credit > 0
    assert ev.max_profit == pytest.approx(credit)
    assert ev.max_loss == pytest.approx(5_000 - credit)
    assert ev.breakevens == [pytest.approx(75_000 - credit)]
    assert ev.pop > 0.5


@pytest.mark.unit
def test_an_iron_condor_has_two_breakevens_and_its_wider_wing_sets_the_loss():
    ev = _ev(("sell", "put", 74_000, EXPIRY), ("buy", "put", 70_000, EXPIRY),
             ("sell", "call", 86_000, EXPIRY), ("buy", "call", 88_000, EXPIRY))
    credit = -ev.net_fill
    assert ev.max_loss == pytest.approx(4_000 - credit)
    assert ev.breakevens == [pytest.approx(74_000 - credit), pytest.approx(86_000 + credit)]
    assert abs(ev.greeks["delta"]) < 0.2 and ev.greeks["vega"] < 0 and ev.greeks["theta"] > 0


@pytest.mark.unit
def test_a_long_straddle_has_unlimited_upside_and_breakevens_either_side():
    ev = _ev(("buy", "call", 80_000, EXPIRY), ("buy", "put", 80_000, EXPIRY))
    assert ev.max_profit == math.inf
    assert ev.max_loss == pytest.approx(ev.net_fill)
    low, high = ev.breakevens
    assert low == pytest.approx(80_000 - ev.net_fill) and high == pytest.approx(80_000 + ev.net_fill)
    assert ev.greeks["vega"] > 0


@pytest.mark.unit
def test_a_cash_secured_put_is_defined_risk():
    ev = _ev(("sell", "put", 75_000, EXPIRY))
    assert ev.ok
    assert ev.max_loss == pytest.approx(75_000 + ev.net_fill)  # strike less the credit


@pytest.mark.unit
def test_a_naked_short_call_is_refused():
    ev = _ev(("sell", "call", 90_000, EXPIRY))
    assert not ev.ok and "no upper bound" in ev.errors[0]
    assert "Not priced" in st.render_evaluation(ev)


@pytest.mark.unit
def test_legs_on_two_expiries_are_refused():
    ev = _ev(("buy", "call", 80_000, "2026-11-06"), ("sell", "call", 90_000, EXPIRY))
    assert any("share one expiry" in e for e in ev.errors)


@pytest.mark.unit
def test_an_unlisted_strike_or_expiry_moves_to_the_nearest_listed_one():
    ev = _ev(("buy", "call", 80_400, "2026-11-25"))
    assert ev.ok
    assert ev.legs[0].strike == 80_000 and ev.legs[0].expiry.strftime("%Y-%m-%d") == EXPIRY
    assert len(ev.notes) == 2


@pytest.mark.unit
def test_a_leg_without_a_bid_to_sell_into_is_refused():
    chain = _chain(expiry_days=(30,))
    q = chain.quotes
    q.loc[(q["strike"] == 90_000) & (q["type"] == "call"), "bid"] = math.nan
    ev = st.evaluate_structure(chain, [st.Leg("buy", "call", 80_000, EXPIRY), st.Leg("sell", "call", 90_000, EXPIRY)])
    assert any("no bid to sell into" in e for e in ev.errors)


@pytest.mark.unit
def test_an_expiry_too_close_is_refused():
    ev = _ev(("buy", "call", 80_000, "2026-10-28"), days=(1, 30))
    assert any("days out" in e for e in ev.errors)


@pytest.mark.unit
def test_the_record_is_plain_data():
    import json

    ev = _ev(("buy", "call", 80_000, EXPIRY), ("buy", "put", 80_000, EXPIRY))
    record = st.evaluation_record(ev)
    json.dumps(record)
    assert record["max_profit"] is None and record["max_profit_unbounded"]
    assert record["legs"][0]["instrument"].startswith("BTC-30-80000")


@pytest.mark.unit
def test_strike_guide_lists_expiries_in_range_with_delta_strikes():
    guide = st.strike_guide(_chain(expiry_days=(3, 10, 30, 120)))
    assert "2026-11-06" in guide and EXPIRY in guide
    assert "2026-10-30" not in guide and "2027-02-24" not in guide
    assert "25d put" in guide and "10d call" in guide


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------

def _state(**extra):
    state = {
        "company_of_interest": "BTC-USD", "trade_date": AS_OF.strftime("%Y-%m-%d"), "asset_type": "crypto",
        "instrument_context": "", "portfolio_context": "", "past_context": "",
        "market_report": "m", "sentiment_report": "", "news_report": "", "fundamentals_report": "",
        "options_report": "IV rich to realized.", "investment_plan": "**Recommendation**: Overweight",
        "trader_investment_plan": "", "options_trade": {},
        "investment_debate_state": {"bull_history": "", "bear_history": "", "history": "debate",
                                    "current_response": "", "count": 0},
        "risk_debate_state": {"history": "risk debate", "latest_speaker": "", "count": 0,
                              "aggressive_history": "", "conservative_history": "", "neutral_history": "",
                              "current_aggressive_response": "", "current_conservative_response": "",
                              "current_neutral_response": ""},
    }
    state.update(extra)
    return state


class _StructuredLLM:
    """A model whose structured answers come from a queue; it records each prompt."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.prompts = []

    def with_structured_output(self, schema):
        return self

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return self.answers.pop(0)


def _proposal(*legs, strategy="Bull call spread"):
    return OptionsTradeProposal(
        strategy=strategy, rationale="fits the plan", exit_plan="close at 50% of max profit",
        risk_budget_pct=1.0,
        legs=[{"side": s, "option_type": t, "strike": k, "expiry": e} for s, t, k, e in legs],
    )


def _strategist(llm, chain):
    from tradingagents.agents.trader import options_strategist

    patch = mock.patch.object(options_strategist, "route_to_vendor", return_value=chain)
    return options_strategist.create_options_strategist(llm), patch


@pytest.mark.unit
def test_strategist_prices_its_structure_and_records_it():
    llm = _StructuredLLM(_proposal(("buy", "call", 80_000, EXPIRY), ("sell", "call", 90_000, EXPIRY)))
    node, patch = _strategist(llm, _chain(expiry_days=(10, 30)))
    with patch:
        out = node(_state())
    assert "Structure check" in out["trader_investment_plan"] and "Max loss at expiry" in out["trader_investment_plan"]
    trade = out["options_trade"]
    assert trade["strategy"] == "Bull call spread" and len(trade["legs"]) == 2 and not trade["errors"]
    assert "Strike guide" in llm.prompts[0][1]["content"]


@pytest.mark.unit
def test_strategist_gets_one_chance_to_fix_a_structure_the_chain_refuses():
    bad = _proposal(("sell", "call", 90_000, EXPIRY), strategy="Bear call spread")
    good = _proposal(("sell", "call", 90_000, EXPIRY), ("buy", "call", 95_000, EXPIRY), strategy="Bear call spread")
    llm = _StructuredLLM(bad, good)
    node, patch = _strategist(llm, _chain(expiry_days=(10, 30)))
    with patch:
        out = node(_state())
    assert "no upper bound" in llm.prompts[1][-1]["content"]
    assert len(out["options_trade"]["legs"]) == 2 and not out["options_trade"]["errors"]


@pytest.mark.unit
def test_strategist_without_a_chain_proposes_no_trade():
    node, patch = _strategist(_StructuredLLM(), "NO_DATA_AVAILABLE: withheld")
    with patch:
        out = node(_state())
    assert out["options_trade"]["strategy"] == "No trade"
    assert "NO_DATA_AVAILABLE" in out["trader_investment_plan"]


@pytest.mark.unit
def test_strategist_no_trade_is_not_priced():
    node, patch = _strategist(_StructuredLLM(_proposal(strategy="No trade")), _chain())
    with patch:
        out = node(_state())
    assert out["options_trade"] == {"strategy": "No trade", "risk_budget_pct": 1.0}
    assert "Structure check" not in out["trader_investment_plan"]


@pytest.mark.unit
def test_portfolio_manager_rules_on_the_structure_and_keeps_the_rating_line_first():
    from tradingagents.agents.managers.options_portfolio_manager import (
        create_options_portfolio_manager,
    )

    decision = OptionsPortfolioDecision(
        rating="Overweight", structure_decision="Approve", risk_budget_pct="0.5%",
        executive_summary="take it", investment_thesis="fits", exit_plan="half profit")
    llm = _StructuredLLM(decision)
    trade = {"strategy": "Bull call spread", "legs": [{"instrument": "x"}], "errors": []}
    out = create_options_portfolio_manager(llm)(_state(options_trade=trade, trader_investment_plan="plan"))
    assert out["final_rating"] == "Overweight"
    assert out["final_trade_decision"].startswith("**Rating**: Overweight")
    assert out["options_trade"]["decision"] == "Approve"
    assert out["options_trade"]["approved_risk_budget_pct"] == 0.5
    assert "priced against the chain" in llm.prompts[0]


@pytest.mark.unit
def test_research_manager_in_options_mode_asks_for_a_volatility_view():
    from tradingagents.agents.managers.research_manager import create_research_manager

    plan = OptionsResearchPlan(recommendation="Overweight", rationale="r", strategic_actions="s",
                               volatility_view="Short volatility", volatility_rationale="IV rich",
                               horizon_days=30)
    llm = _StructuredLLM(plan)
    out = create_research_manager(llm, options_mode=True)(_state())
    assert "**Volatility View**: Short volatility" in out["investment_plan"]
    assert "Options market report: IV rich to realized." in llm.prompts[0]
    assert "**Horizon**: 30 days" in render_options_research_plan(plan)


@pytest.mark.unit
@pytest.mark.parametrize("name", ["aggressive", "conservative", "neutral"])
def test_risk_analysts_get_the_options_brief_only_in_options_mode(name):
    import importlib

    from tradingagents.dataflows.config import run_config

    module = importlib.import_module(f"tradingagents.agents.risk_mgmt.{name}_debator")
    factory = getattr(module, f"create_{name}_debator")

    class _LLM:
        def __init__(self):
            self.prompts = []

        def invoke(self, prompt):
            self.prompts.append(prompt)
            return AIMessage("argument")

    llm = _LLM()
    with run_config({"decision_mode": "options"}):
        factory(llm)(_state())
    factory(llm)(_state(options_report=""))
    assert "structure check priced in code" in llm.prompts[0]
    assert "Options Market Report: IV rich to realized." in llm.prompts[0]
    assert "structure check" not in llm.prompts[1] and "Options Market Report" not in llm.prompts[1]


# ---------------------------------------------------------------------------
# Graph and CLI
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_options_mode_needs_the_options_analyst():
    from tradingagents.graph.setup import check_decision_mode

    check_decision_mode("options", ["market", "options"])
    check_decision_mode("direction", ["market"])
    with pytest.raises(ValueError, match="options analyst"):
        check_decision_mode("options", ["market"])
    with pytest.raises(ValueError, match="unknown decision_mode"):
        check_decision_mode("straddles", ["options"])


@pytest.mark.unit
def test_options_mode_builds_the_options_nodes():
    from tradingagents.graph.conditional_logic import ConditionalLogic
    from tradingagents.graph.setup import GraphSetup

    class _LLM:
        def with_structured_output(self, schema):
            return self

        def bind_tools(self, tools):
            return self

    made = {}
    with mock.patch("tradingagents.graph.setup.create_options_strategist",
                    side_effect=lambda llm: made.setdefault("strategist", lambda s: {})), \
         mock.patch("tradingagents.graph.setup.create_options_portfolio_manager",
                    side_effect=lambda llm: made.setdefault("pm", lambda s: {})):
        GraphSetup(_LLM(), _LLM(), ConditionalLogic(), 3).setup_graph(["options"], decision_mode="options")
    assert set(made) == {"strategist", "pm"}


@pytest.mark.unit
def test_cli_refuses_an_unknown_mode_before_any_question(monkeypatch):
    import typer

    from cli import run

    monkeypatch.setattr(run, "get_user_selections", lambda flags: pytest.fail("asked the questions"))
    with pytest.raises(typer.Exit):
        run.run_analysis(flags={"mode": "gamma"})


@pytest.mark.unit
def test_the_strike_guide_and_check_use_the_forward_not_spot():
    ev = _ev(("buy", "call", 80_000, EXPIRY))
    assert ev.forward == FORWARD
