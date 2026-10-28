"""Options mode: structures priced in code, the strategist's propose-check-fix loop,
the Portfolio Manager's verdict, and the graph and CLI wiring.

The chain is synthetic and every model is faked, so these run offline.
"""

import dataclasses
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


def _ev(*legs, days=(10, 30), **kwargs):
    return st.evaluate_structure(_chain(expiry_days=days, open_interest=100.0), [st.Leg(*leg) for leg in legs],
                                 **kwargs)


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
def test_venue_fees_are_charged_on_entry_and_inside_every_figure():
    chain = dataclasses.replace(_chain(expiry_days=(10, 30), open_interest=100.0), venue="Deribit")
    legs = [st.Leg("sell", "put", 75_000, EXPIRY), st.Leg("buy", "put", 70_000, EXPIRY)]
    ev = st.evaluate_structure(chain, legs)
    short, long_ = ev.legs
    fee = sum(min(0.0003 * FORWARD, 0.125 * p.fill) for p in ev.legs)
    assert ev.fees == pytest.approx(fee) and fee > 0
    credit = short.fill - long_.fill
    assert ev.max_profit == pytest.approx(credit - fee)
    assert ev.max_loss == pytest.approx(5_000 - credit + fee)
    assert ev.round_trip_cost == pytest.approx((short.ask - short.bid) + (long_.ask - long_.bid) + 2 * fee)
    assert "Venue terms (Deribit)" in st.render_evaluation(ev)


@pytest.mark.unit
def test_expected_pnl_at_iv_is_the_cost_of_crossing_and_realized_vol_gives_the_edge():
    # The test chain is flat at 40% with quotes 2% either side of mark, and no fees.
    ev = _ev(("buy", "call", 80_000, EXPIRY), realized_vol=0.60)
    leg = ev.legs[0]
    assert ev.expected_pnl_iv == pytest.approx(leg.mark - leg.ask, rel=1e-3)
    assert ev.expected_pnl_rv > 0 > ev.expected_pnl_iv  # options bought below the vol realized
    assert ev.pop_rv > ev.pop
    text = st.render_evaluation(ev)
    assert "at the 30-day realized volatility (60.0%)" in text
    unknown = _ev(("buy", "call", 80_000, EXPIRY))
    assert math.isnan(unknown.expected_pnl_rv) and "realized volatility" not in st.render_evaluation(unknown)


@pytest.mark.unit
def test_a_leg_without_open_interest_or_with_a_wide_spread_breaks_the_limits():
    chain = _chain(expiry_days=(10, 30), open_interest=100.0)
    q = chain.quotes
    q.loc[(q["strike"] == 90_000) & (q["type"] == "call"), "open_interest"] = 0.0
    wide = (q["strike"] == 80_000) & (q["type"] == "call")
    q.loc[wide, "ask"] = q.loc[wide, "mark"] * 1.4
    ev = st.evaluate_structure(chain, [st.Leg("buy", "call", 80_000, EXPIRY), st.Leg("sell", "call", 90_000, EXPIRY)])
    assert ev.ok and not ev.tradeable
    assert any("no open interest" in b for b in ev.breaches)
    assert any("bid-ask spread is" in b for b in ev.breaches)
    assert "Breaks the desk's risk limits" in st.render_evaluation(ev)
    assert not st.is_tradeable(st.evaluation_record(ev))


@pytest.mark.unit
def test_a_structure_that_neither_wins_often_nor_pays_enough_breaks_the_limits():
    # A credit spread passes on its probability of profit, an out-of-the-money debit
    # spread on its reward to risk.
    credit = _ev(("sell", "put", 75_000, EXPIRY), ("buy", "put", 70_000, EXPIRY))
    assert credit.pop >= st.MIN_POP and credit.tradeable
    assert "Risk limits: passed" in st.render_evaluation(credit)
    debit = _ev(("buy", "call", 95_000, EXPIRY), ("sell", "call", 100_000, EXPIRY))
    assert debit.pop < st.MIN_POP and debit.max_profit / debit.max_loss >= st.MIN_REWARD_TO_RISK
    assert debit.tradeable

    # Bought dear, the same spread rarely wins and pays less than it costs.
    chain = _chain(expiry_days=(10, 30), open_interest=100.0)
    q = chain.quotes
    row = (q["strike"] == 95_000) & (q["type"] == "call")
    q.loc[row, ["bid", "mark", "ask"]] = [2_900.0, 3_000.0, 3_100.0]
    dear = st.evaluate_structure(chain, [st.Leg("buy", "call", 95_000, EXPIRY), st.Leg("sell", "call", 100_000, EXPIRY)])
    assert any("neither wins often enough nor pays enough" in b for b in dear.breaches)


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
        legs=[{"side": s, "option_type": t, "strike": k, "expiry": e} for s, t, k, e in legs],
    )


def _vendor(chain, closes=None):
    """The options pricer's router, answering from a test chain."""
    from tradingagents.agents import options_pricing

    def vendor(method, *args):
        if method == "get_underlying_closes":
            return closes if closes is not None else "NO_DATA_AVAILABLE: no closes"
        return chain

    return mock.patch.object(options_pricing, "route_to_vendor", side_effect=vendor)


def _strategist(llm, chain, closes=None):
    from tradingagents.agents.trader import options_strategist

    return options_strategist.create_options_strategist(llm), _vendor(chain, closes)


def _liquid(days=(10, 30)):
    return _chain(expiry_days=days, open_interest=100.0)


def _closes():
    import pandas as pd

    return pd.Series([FORWARD * (1.01 if i % 2 else 1.0) for i in range(40)])


@pytest.mark.unit
def test_strategist_prices_its_structure_and_records_it():
    llm = _StructuredLLM(_proposal(("buy", "call", 80_000, EXPIRY), ("sell", "call", 90_000, EXPIRY)))
    node, patch = _strategist(llm, _liquid(), closes=_closes())
    with patch:
        out = node(_state(investment_plan="**Recommendation**: Overweight\n\n**Horizon**: 30 days"))
    plan = out["trader_investment_plan"]
    assert "Structure check" in plan and "Max loss at expiry" in plan and "Risk limits: passed" in plan
    assert "at the 30-day realized volatility" in plan
    assert "Risk Budget" not in plan and "Size" not in plan  # sizing is the desk's call
    trade = out["options_trade"]
    assert trade["strategy"] == "Bull call spread" and len(trade["legs"]) == 2 and not trade["errors"]
    assert trade["realized_vol"] > 0 and trade["limit_breaches"] == [] and "risk_budget_pct" not in trade
    prompt = llm.prompts[0][1]["content"]
    assert "Strike guide" in prompt and f"expiry {EXPIRY}" in prompt  # the ladder is the horizon's expiry
    assert "Do not size the position" in llm.prompts[0][0]["content"]
    assert out["options_chain_view"] in prompt


@pytest.mark.unit
def test_strategist_gets_one_chance_to_fix_a_structure_that_breaks_a_limit():
    chain = _liquid()
    q = chain.quotes
    q.loc[(q["strike"] == 95_000) & (q["type"] == "call"), "open_interest"] = 0.0
    thin = _proposal(("sell", "call", 90_000, EXPIRY), ("buy", "call", 95_000, EXPIRY), strategy="Bear call spread")
    fixed = _proposal(("sell", "call", 90_000, EXPIRY), ("buy", "call", 94_000, EXPIRY), strategy="Bear call spread")
    llm = _StructuredLLM(thin, fixed)
    node, patch = _strategist(llm, chain)
    with patch:
        out = node(_state())
    assert "breaks the desk's risk limits" in llm.prompts[1][-1]["content"]
    assert "no open interest" in llm.prompts[1][-1]["content"]
    assert [leg["strike"] for leg in out["options_trade"]["legs"]] == [90_000, 94_000]
    assert st.is_tradeable(out["options_trade"])


@pytest.mark.unit
def test_strategist_gets_one_chance_to_fix_a_structure_the_chain_refuses():
    bad = _proposal(("sell", "call", 90_000, EXPIRY), strategy="Bear call spread")
    good = _proposal(("sell", "call", 90_000, EXPIRY), ("buy", "call", 95_000, EXPIRY), strategy="Bear call spread")
    llm = _StructuredLLM(bad, good)
    node, patch = _strategist(llm, _liquid())
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
    node, patch = _strategist(_StructuredLLM(_proposal(strategy="No trade")), _liquid())
    with patch:
        out = node(_state())
    assert out["options_trade"] == {"strategy": "No trade"}
    assert "Structure check" not in out["trader_investment_plan"]


def _pm(*decisions, chain=None, **state):
    from tradingagents.agents.managers.options_portfolio_manager import (
        create_options_portfolio_manager,
    )

    llm = _StructuredLLM(*decisions)
    with _vendor(chain if chain is not None else _liquid()):
        out = create_options_portfolio_manager(llm)(_state(trader_investment_plan="plan", **state))
    return out, llm


def _decision(**fields):
    base = {"rating": "Overweight", "structure_decision": "Approve",
            "executive_summary": "take it", "investment_thesis": "fits", "exit_plan": "half profit"}
    return OptionsPortfolioDecision(**{**base, **fields})


PRICED = {"strategy": "Bull call spread", "legs": [{"instrument": "x", "strike": 80_000.0}], "errors": [],
          "limit_breaches": []}


@pytest.mark.unit
def test_portfolio_manager_rules_on_the_structure_and_keeps_the_rating_line_first():
    out, llm = _pm(_decision(), options_trade=PRICED)
    assert out["final_rating"] == "Overweight"
    assert out["final_trade_decision"].startswith("**Rating**: Overweight")
    assert out["options_trade"]["decision"] == "Approve"
    assert "Risk Budget" not in out["final_trade_decision"]
    assert "priced against the chain" in llm.prompts[0]
    assert "Reward to risk alone does not decide" in llm.prompts[0] and "Do not size" in llm.prompts[0]


@pytest.mark.unit
def test_portfolio_manager_cannot_approve_a_structure_outside_the_limits():
    trade = {**PRICED, "limit_breaches": ["x: no open interest"]}
    out, llm = _pm(_decision(), options_trade=trade)
    assert out["options_trade"]["decision"] == "Reject"
    assert "**Structure Decision**: Reject" in out["final_trade_decision"]
    assert "cannot be approved as it is" not in llm.prompts[0]  # no chain view, so no adjustment offered
    assert "No option chain was available" in llm.prompts[0]


def _adjust_to(*legs, strategy="Bull call spread"):
    return _decision(structure_decision="Adjust", adjusted_strategy=strategy,
                     adjusted_legs=[{"side": s, "option_type": t, "strike": k, "expiry": e} for s, t, k, e in legs])


@pytest.mark.unit
def test_an_adjustment_is_priced_and_shown_back_before_it_is_approved():
    adjust = _adjust_to(("buy", "call", 80_000, EXPIRY), ("sell", "call", 85_000, EXPIRY))
    out, llm = _pm(adjust, _decision(), options_trade=PRICED, options_chain_view="ladder")
    assert len(llm.prompts) == 2
    assert "Structure check" in llm.prompts[1] and "Approve this adjusted structure" in llm.prompts[1]
    trade = out["options_trade"]
    assert trade["decision"] == "Approve" and [leg["strike"] for leg in trade["legs"]] == [80_000, 85_000]
    assert trade["adjusted_from"]["legs"] == PRICED["legs"]
    text = out["final_trade_decision"]
    assert "**Adjusted Structure**: Bull call spread" in text and "Structure check" in text


@pytest.mark.unit
def test_an_adjustment_outside_the_limits_is_rejected_without_a_second_question():
    chain = _liquid()
    q = chain.quotes
    q.loc[(q["strike"] == 85_000) & (q["type"] == "call"), "open_interest"] = 0.0
    adjust = _adjust_to(("buy", "call", 80_000, EXPIRY), ("sell", "call", 85_000, EXPIRY))
    out, llm = _pm(adjust, chain=chain, options_trade=PRICED, options_chain_view="ladder")
    assert len(llm.prompts) == 1
    trade = out["options_trade"]
    assert trade["decision"] == "Reject" and trade["legs"] == PRICED["legs"]
    assert trade["adjustment"]["limit_breaches"]
    assert "breaks the desk's risk limits" in out["final_trade_decision"]


@pytest.mark.unit
def test_an_adjustment_not_confirmed_once_priced_is_rejected():
    adjust = _adjust_to(("buy", "call", 80_000, EXPIRY), ("sell", "call", 85_000, EXPIRY))
    out, _ = _pm(adjust, _decision(structure_decision="Adjust"), options_trade=PRICED, options_chain_view="ladder")
    assert out["options_trade"]["decision"] == "Reject"
    assert "not confirmed once priced" in out["final_trade_decision"]


@pytest.mark.unit
def test_a_structure_outside_the_limits_can_be_adjusted_into_them():
    trade = {**PRICED, "limit_breaches": ["x: no open interest"]}
    adjust = _adjust_to(("buy", "call", 80_000, EXPIRY), ("sell", "call", 85_000, EXPIRY))
    out, llm = _pm(adjust, _decision(), options_trade=trade, options_chain_view="ladder")
    assert "Reject, or Adjust" in llm.prompts[0] and "ladder" in llm.prompts[0]
    assert out["options_trade"]["decision"] == "Approve" and st.is_tradeable(out["options_trade"])


@pytest.mark.unit
def test_research_manager_in_options_mode_asks_for_a_volatility_view():
    from tradingagents.agents.managers.research_manager import create_research_manager

    plan = OptionsResearchPlan(recommendation="Overweight", rationale="r", strategic_actions="s",
                               volatility_view="Short volatility", volatility_rationale="IV rich",
                               horizon_days=30)
    llm = _StructuredLLM(plan)
    out = create_research_manager(llm, options_mode=True)(_state())
    assert "Do not size the position" in llm.prompts[0] and "standard allocation" not in llm.prompts[0]
    assert "**Volatility View**: Short volatility" in out["investment_plan"]
    assert "Options market report: IV rich to realized." in llm.prompts[0]
    assert "**Horizon**: 30 days" in render_options_research_plan(plan)


@pytest.mark.unit
@pytest.mark.parametrize(("name", "role"), [("aggressive", "Trade Advocate"),
                                            ("conservative", "Tail-Risk Sceptic"),
                                            ("neutral", "Structurer")])
def test_risk_seats_take_options_roles_only_in_options_mode(name, role):
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
        out = factory(llm)(_state(options_chain_view="LADDER"))
    plain = factory(llm)(_state(options_report="", options_chain_view="LADDER"))
    options_prompt, direction_prompt = llm.prompts
    assert f"You are the {role}" in options_prompt and "Do not size the position" in options_prompt
    assert "Options market report: IV rich to realized." in options_prompt
    assert ("LADDER" in options_prompt) == (name == "neutral")  # only the Structurer gets the chain
    assert out["risk_debate_state"]["history"].endswith(f"{role}: argument")
    assert role not in direction_prompt and "Options Market Report" not in direction_prompt
    assert plain["risk_debate_state"]["latest_speaker"] == out["risk_debate_state"]["latest_speaker"]

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
def test_only_a_structure_within_the_limits_goes_to_the_risk_debate():
    from tradingagents.graph.setup import _debate_or_decide

    priced = {"legs": [{"instrument": "x"}], "errors": [], "limit_breaches": []}
    assert _debate_or_decide({"options_trade": priced}) == "Aggressive Analyst"
    assert _debate_or_decide({"options_trade": {**priced, "limit_breaches": ["thin"]}}) == "Portfolio Manager"
    assert _debate_or_decide({"options_trade": {"strategy": "No trade"}}) == "Portfolio Manager"


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
