"""Options data: the metrics computed in code, the Deribit vendor's normalization
and point-in-time guard, routing, and the options analyst's wiring.

Vendor access is faked, so these run without a network connection; the one live
check is marked integration.
"""

import copy
import math
from datetime import UTC, datetime, timedelta
from unittest import mock

import ccxt
import numpy as np
import pandas as pd
import pytest
from langchain_core.messages import AIMessage

import tradingagents.dataflows.config as config_module
import tradingagents.default_config as default_config
from cli.models import AnalystType, AssetType
from cli.prompts import filter_analysts_for_asset_type, parse_analysts
from tradingagents.dataflows import options_metrics as om, router
from tradingagents.dataflows.errors import NoMarketDataError, VendorUnavailableError
from tradingagents.dataflows.vendors import deribit
from tradingagents.graph.analyst_execution import build_analyst_execution_plan

AS_OF = datetime(2026, 10, 27, 8, 0, tzinfo=UTC)
FORWARD = 80_000.0


def _norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _black76(forward, strike, years, iv, call):
    d1 = (math.log(forward / strike) + 0.5 * iv * iv * years) / (iv * math.sqrt(years))
    d2 = d1 - iv * math.sqrt(years)
    if call:
        return forward * _norm_cdf(d1) - strike * _norm_cdf(d2)
    return strike * _norm_cdf(-d2) - forward * _norm_cdf(-d1)


def _chain(expiry_days=(10, 40), smile=lambda k, f: 0.40, open_interest=None, volume=None) -> om.OptionChain:
    """A chain priced by Black-76 off a known smile, so the expected metrics are known.

    ``open_interest`` is a number for every contract, or a dict by (days, strike, kind).
    """
    if isinstance(open_interest, (int, float)):
        open_interest = {(d, float(k), kind): float(open_interest) for d in expiry_days
                         for k in np.arange(50_000, 110_001, 1_000.0) for kind in ("call", "put")}
    rows = []
    for days in expiry_days:
        expiry = pd.Timestamp(AS_OF + timedelta(days=days))
        years = days / 365.0
        for strike in np.arange(50_000, 110_001, 1_000.0):
            iv = smile(strike, FORWARD)
            for kind in ("call", "put"):
                mark = _black76(FORWARD, strike, years, iv, kind == "call")
                rows.append({
                    "instrument": f"BTC-{days}-{int(strike)}-{kind[0].upper()}",
                    "expiry": expiry, "strike": strike, "type": kind,
                    "bid": mark * 0.98, "ask": mark * 1.02, "mark": mark, "iv": iv,
                    "open_interest": (open_interest or {}).get((days, strike, kind), 0.0),
                    "volume": (volume or {}).get((days, strike, kind), 0.0),
                    "forward": FORWARD,
                })
    return om.OptionChain("BTC", "TestVenue", AS_OF, FORWARD, pd.DataFrame(rows))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_years_to_expiry_counts_to_the_hour():
    expiry = pd.Series([pd.Timestamp("2026-10-29 08:00", tz="UTC"), pd.Timestamp("2026-10-01", tz="UTC")])
    years = om.years_to_expiry(expiry, AS_OF)
    assert years[0] == pytest.approx(2 / 365)
    assert years[1] == 0.0  # past expiries floor at zero


@pytest.mark.unit
def test_black76_greeks_obey_put_call_parity_and_signs():
    g = om.black76_greeks([FORWARD] * 2, [FORWARD] * 2, [0.1] * 2, [0.5] * 2, [True, False])
    assert g["delta"][0] - g["delta"][1] == pytest.approx(1.0)
    assert 0.5 < g["delta"][0] < 0.6  # ATM-forward call delta is N(sigma*sqrt(T)/2)
    assert g["gamma"][0] == pytest.approx(g["gamma"][1]) and g["gamma"][0] > 0
    assert g["theta"][0] < 0 and g["vega"][0] > 0


@pytest.mark.unit
def test_black76_greeks_are_nan_at_expiry_or_without_iv():
    g = om.black76_greeks([FORWARD] * 2, [FORWARD] * 2, [0.0, 0.1], [0.5, math.nan], [True, True])
    assert np.isnan(g["delta"]).all()


@pytest.mark.unit
def test_flat_smile_gives_atm_iv_no_skew_and_the_straddle_move():
    summary = om.expiry_summary(om.with_greeks(_chain()))
    row = summary.iloc[0]
    assert row["atm_iv"] == pytest.approx(0.40)
    assert row["rr25"] == pytest.approx(0.0, abs=1e-9)
    assert row["bf25"] == pytest.approx(0.0, abs=1e-9)
    # An ATM-forward straddle is worth F * (2N(sigma*sqrt(T)/2) - 1) * 2.
    expected = 2 * (2 * _norm_cdf(0.40 * math.sqrt(10 / 365) / 2) - 1)
    assert row["implied_move"] == pytest.approx(expected, rel=1e-6)


@pytest.mark.unit
def test_put_skew_reads_as_a_negative_risk_reversal():
    chain = _chain(smile=lambda k, f: 0.40 - 0.30 * math.log(k / f))
    row = om.expiry_summary(om.with_greeks(chain)).iloc[0]
    assert row["put25_iv"] > row["atm_iv"] > row["call25_iv"]
    assert row["rr25"] < 0


@pytest.mark.unit
def test_constant_maturity_iv_interpolates_total_variance():
    summary = pd.DataFrame({"dte": [15.0, 45.0], "years": [15 / 365, 45 / 365], "atm_iv": [0.50, 0.30]})
    iv, extrapolated = om.constant_maturity_iv(summary, 30)
    v15, v45 = 0.25 * 15 / 365, 0.09 * 45 / 365
    assert iv == pytest.approx(math.sqrt((v15 + (v45 - v15) / 2) / (30 / 365)))
    assert not extrapolated


@pytest.mark.unit
def test_constant_maturity_iv_flags_extrapolation():
    summary = pd.DataFrame({"dte": [5.0, 10.0], "years": [5 / 365, 10 / 365], "atm_iv": [0.5, 0.6]})
    assert om.constant_maturity_iv(summary, 30) == (0.6, True)


@pytest.mark.unit
def test_realized_vol_annualizes_daily_log_returns_by_365():
    returns = np.array([0.01, -0.02, 0.015, -0.005, 0.0, 0.02, -0.01, 0.005])
    closes = pd.Series(100 * np.exp(np.concatenate([[0.0], np.cumsum(returns)])))
    assert om.realized_vol(closes, 7) == pytest.approx(np.std(returns[-7:], ddof=1) * math.sqrt(365))
    assert math.isnan(om.realized_vol(closes.head(3), 30))  # too little history to say


@pytest.mark.unit
def test_iv_rank_and_percentile():
    rank, percentile = om.iv_rank(pd.Series([20.0, 30.0, 40.0]), 35.0)
    assert rank == pytest.approx(75.0)
    assert percentile == pytest.approx(200 / 3)


@pytest.mark.unit
def test_max_pain_is_the_strike_paying_holders_least():
    group = pd.DataFrame({
        "strike": [90.0, 100.0, 100.0, 110.0],
        "type": ["call", "call", "put", "put"],
        "open_interest": [5.0, 1.0, 1.0, 5.0],
    })
    assert om.max_pain(group) == 100.0


@pytest.mark.unit
def test_oi_walls_rank_strikes_across_expiries():
    oi = {(10, 90_000.0, "call"): 5.0, (40, 90_000.0, "call"): 5.0, (10, 95_000.0, "call"): 7.0}
    df = om.with_greeks(_chain(open_interest=oi))
    assert om.oi_walls(df, "call", n=2) == [(90_000.0, 10.0), (95_000.0, 7.0)]
    assert om.oi_walls(df, "put") == []


@pytest.mark.unit
def test_snapshot_says_what_it_could_not_compute():
    out = om.format_snapshot(_chain())
    assert "30-day ATM IV" in out and "40.0%" in out
    move = 2 * (2 * _norm_cdf(0.40 * math.sqrt(10 / 365) / 2) - 1)
    assert f"${FORWARD * (1 - move):,.0f}-${FORWARD * (1 + move):,.0f}" in out
    assert "Realized vol: unavailable" in out
    assert "IV rank: unavailable" in out
    assert "| Expiry | DTE |" in out


@pytest.mark.unit
def test_snapshot_compares_iv_with_realized_and_its_history():
    closes = pd.Series(100 * np.exp(np.cumsum(np.tile([0.01, -0.01], 20))))
    out = om.format_snapshot(_chain(), closes=closes, iv_index=pd.Series([30.0, 50.0, 40.0]),
                             iv_index_name="DVOL (BTC)")
    assert "30-day IV minus 30-day realized" in out
    assert "DVOL (BTC): 40.0" in out and "IV rank 50" in out


@pytest.mark.unit
def test_snapshot_leaves_a_same_day_expiry_out_of_the_term_structure():
    out = om.format_snapshot(_chain(expiry_days=(0.5, 10, 40)))
    assert "Expiring within a day and left out above" in out


@pytest.mark.unit
def test_chain_defaults_to_the_first_expiry_a_week_out_and_bands_strikes():
    out = om.format_chain(_chain(expiry_days=(3, 10, 40)), moneyness_pct=5)
    assert "expiry 2026-11-06" in out  # 10 days out; 3 days is too near
    assert "| $76,000 |" in out and "| $84,000 |" in out
    assert "| $75,000 |" not in out and "| $85,000 |" not in out


@pytest.mark.unit
def test_chain_takes_the_listed_expiry_nearest_the_request():
    out = om.format_chain(_chain(expiry_days=(3, 10, 40)), expiry="2026-12-01")
    assert "expiry 2026-12-06" in out


# ---------------------------------------------------------------------------
# Deribit vendor
# ---------------------------------------------------------------------------

def _market(base, settle, strike, kind, days=10):
    return {
        "id": f"{base}-{strike}-{kind[0].upper()}", "base": base, "settle": settle,
        "strike": strike, "optionType": kind,
        "expiry": int((AS_OF + timedelta(days=days)).timestamp() * 1000), "option": True,
    }


@pytest.fixture(autouse=True)
def _no_cached_chains():
    deribit._chains.clear()
    yield
    deribit._chains.clear()


class _FakeDeribit:
    def __init__(self, markets, tickers, error=None):
        self.markets = markets
        self._tickers = tickers
        self._error = error

    def load_markets(self, reload=False):
        return self.markets

    def fetch_option_chain(self, code):
        if self._error:
            raise self._error
        return self._tickers[code]


def _ticker(mark, iv, underlying, oi=1.0, volume=0.0):
    return {"info": {"mark_price": mark, "bid_price": mark * 0.9, "ask_price": mark * 1.1,
                     "mark_iv": iv, "underlying_price": underlying, "estimated_delivery_price": underlying,
                     "open_interest": oi, "volume": volume}}


@pytest.mark.unit
def test_historical_date_is_withheld_without_a_request():
    with mock.patch.object(deribit, "_exchange", side_effect=AssertionError("reached Deribit")):
        assert "withheld for 2020-01-02" in deribit.get_options_snapshot("BTC-USD", "2020-01-02")
        assert "withheld for 2020-01-02" in deribit.get_option_chain("BTC-USD", "", 10, "2020-01-02")


@pytest.mark.unit
def test_a_non_crypto_symbol_has_no_deribit_options():
    with mock.patch.object(deribit, "_exchange", side_effect=AssertionError("reached Deribit")), \
         pytest.raises(NoMarketDataError):
        deribit.fetch_chain("AAPL")


@pytest.mark.unit
def test_coin_settled_prices_convert_to_usd_and_other_coins_are_dropped():
    markets = {
        "BTC/USD:BTC-1-C": _market("BTC", "BTC", 80_000.0, "call"),
        "BTC/USD:BTC-2-P": _market("BTC", "BTC", 80_000.0, "put"),
        "ETH/USD:ETH-1-C": _market("ETH", "ETH", 3_000.0, "call"),
        "BTC/USD:BTC-3-C": _market("BTC", "BTC", 90_000.0, "call"),
    }
    tickers = {"BTC": {
        "BTC/USD:BTC-1-C": _ticker(0.02, 45.0, 80_500.0, oi=12.0),
        "BTC/USD:BTC-2-P": _ticker(0.015, 46.0, 80_500.0),
        "ETH/USD:ETH-1-C": _ticker(0.05, 60.0, 3_010.0),
        "BTC/USD:BTC-3-C": _ticker(0.001, 0.0, 80_500.0),  # no mark IV: dropped
    }}
    with mock.patch.object(deribit, "_exchange", return_value=_FakeDeribit(markets, tickers)):
        chain = deribit.fetch_chain("BTC-USD")
    q = chain.quotes.set_index("type")
    assert len(chain.quotes) == 2
    assert q.loc["call", "mark"] == pytest.approx(0.02 * 80_500.0)
    assert q.loc["call", "iv"] == pytest.approx(0.45)
    assert q.loc["call", "open_interest"] == 12.0
    assert chain.spot == 80_500.0 and chain.underlying == "BTC"


@pytest.mark.unit
def test_a_chain_is_reused_for_a_minute_then_fetched_again():
    markets = {"BTC/USD:BTC-1-C": _market("BTC", "BTC", 80_000.0, "call")}
    fake = _FakeDeribit(markets, {"BTC": {"BTC/USD:BTC-1-C": _ticker(0.02, 45.0, 80_500.0)}})
    with mock.patch.object(deribit, "_exchange", return_value=fake), \
         mock.patch.object(fake, "fetch_option_chain", wraps=fake.fetch_option_chain) as fetched, \
         mock.patch.object(deribit.time, "monotonic", side_effect=[0.0, 30.0, 61.0, 61.0]):
        first = deribit.fetch_chain("BTC-USD")
        assert deribit.fetch_chain("BTC-USD") is first
        assert deribit.fetch_chain("BTC-USD") is not first
    assert fetched.call_count == 2


@pytest.mark.unit
def test_usdc_settled_prices_are_already_usd():
    markets = {"SOL/USDC:USDC-1-C": _market("SOL", "USDC", 100.0, "call")}
    tickers = {"USDC": {"SOL/USDC:USDC-1-C": _ticker(2.5, 55.0, 101.0)}}
    with mock.patch.object(deribit, "_exchange", return_value=_FakeDeribit(markets, tickers)):
        chain = deribit.fetch_chain("SOL-USD")
    assert chain.quotes["mark"].iloc[0] == pytest.approx(2.5)


@pytest.mark.unit
def test_a_network_failure_is_vendor_unavailable_not_no_data():
    fake = _FakeDeribit({}, {}, error=ccxt.NetworkError("timed out"))
    with mock.patch.object(deribit, "_exchange", return_value=fake), mock.patch.object(deribit.time, "sleep"), \
         pytest.raises(VendorUnavailableError):
        deribit.fetch_chain("BTC-USD")


@pytest.mark.unit
def test_a_dropped_request_is_retried_before_the_vendor_gives_up():
    markets = {"BTC/USD:BTC-1-C": _market("BTC", "BTC", 80_000.0, "call")}
    fake = _FakeDeribit(markets, {"BTC": {"BTC/USD:BTC-1-C": _ticker(0.02, 45.0, 80_500.0)}})
    flaky = mock.Mock(side_effect=[ccxt.NetworkError("reset"), fake._tickers["BTC"]])
    with mock.patch.object(deribit, "_exchange", return_value=fake), \
         mock.patch.object(fake, "fetch_option_chain", flaky), \
         mock.patch.object(deribit.time, "sleep") as slept:
        assert len(deribit.fetch_chain("BTC-USD").quotes) == 1
    assert flaky.call_count == 2 and slept.call_count == 1


@pytest.mark.unit
def test_a_failed_market_refresh_falls_back_to_the_list_already_loaded():
    markets = {"BTC/USD:BTC-1-C": _market("BTC", "BTC", 80_000.0, "call")}
    fake = _FakeDeribit(markets, {"BTC": {"BTC/USD:BTC-1-C": _ticker(0.02, 45.0, 80_500.0)}})
    with mock.patch.object(deribit, "_exchange", return_value=fake), \
         mock.patch.object(fake, "load_markets", side_effect=ccxt.NetworkError("get_currencies")), \
         mock.patch.object(deribit.time, "sleep"):
        assert len(deribit.fetch_chain("BTC-USD").quotes) == 1
        fake.markets = {}
        deribit._chains.clear()
        with pytest.raises(VendorUnavailableError):
            deribit.fetch_chain("BTC-USD")


@pytest.mark.unit
def test_snapshot_stands_when_its_supporting_series_fail():
    with mock.patch.object(deribit, "fetch_chain", return_value=_chain()), \
         mock.patch.object(deribit, "fetch_daily_closes", side_effect=VendorUnavailableError("down")), \
         mock.patch.object(deribit, "fetch_dvol", return_value=pd.Series(dtype=float)):
        out = deribit.get_options_snapshot("BTC-USD")
    assert "## Options snapshot: BTC" in out
    assert "Realized vol: unavailable" in out


# ---------------------------------------------------------------------------
# Routing, analyst wiring, CLI
# ---------------------------------------------------------------------------

@pytest.fixture
def default_config_copy():
    config_module._config = copy.deepcopy(default_config.DEFAULT_CONFIG)
    yield
    config_module._config = copy.deepcopy(default_config.DEFAULT_CONFIG)


@pytest.mark.unit
def test_options_tools_route_to_deribit_by_default(default_config_copy):
    assert router.get_category_for_method("get_options_snapshot") == "options_data"
    assert default_config.DEFAULT_CONFIG["data_vendors"]["options_data"] == "deribit"
    with mock.patch.dict(router.VENDOR_METHODS, {"get_options_snapshot": {"deribit": lambda *a: "DERIBIT_OK"}}):
        assert router.route_to_vendor("get_options_snapshot", "BTC-USD", "2026-10-27") == "DERIBIT_OK"


@pytest.mark.unit
def test_a_stock_gets_the_no_data_sentinel_not_an_error(default_config_copy):
    out = router.route_to_vendor("get_options_snapshot", "AAPL", None)
    assert out.startswith("NO_DATA_AVAILABLE")


@pytest.mark.unit
def test_options_analyst_is_wired_with_its_tools():
    spec = build_analyst_execution_plan(["options"]).specs[0]
    assert (spec.agent_node, spec.report_key) == ("Options Analyst", "options_report")
    assert {t.name for t in spec.tools} == {"get_options_snapshot", "get_option_chain"}


@pytest.mark.unit
def test_options_analyst_files_its_report():
    from langchain_core.runnables import RunnableLambda

    from tradingagents.agents.analysts.options_analyst import create_options_analyst

    class _LLM(RunnableLambda):
        def __init__(self):
            super().__init__(lambda _: AIMessage("IV is rich to realized."))

        def bind_tools(self, tools):
            return self

    node = create_options_analyst(_LLM())
    out = node({"trade_date": "2026-10-27", "company_of_interest": "BTC-USD", "asset_type": "crypto",
                "instrument_context": "", "messages": []})
    assert out["options_report"] == "IV is rich to realized."


def _debate_state(options_report):
    return {
        "company_of_interest": "BTC-USD", "trade_date": "2026-10-27", "asset_type": "crypto",
        "instrument_context": "", "market_report": "m", "sentiment_report": "s", "news_report": "n",
        "fundamentals_report": "", "options_report": options_report,
        "investment_debate_state": {"bull_history": "", "bear_history": "", "history": "",
                                    "current_response": "", "count": 0},
    }


@pytest.mark.unit
@pytest.mark.parametrize("factory", ["create_bull_researcher", "create_bear_researcher"])
def test_researchers_read_the_options_report_only_when_there_is_one(factory):
    import tradingagents.agents as agents

    seen = []

    class _LLM:
        def invoke(self, prompt, *a, **k):
            seen.append(prompt)
            return AIMessage("argument")

    node = getattr(agents, factory)(_LLM())
    node(_debate_state("25-delta risk reversal -3 pts"))
    node(_debate_state(""))
    assert "Options market report: 25-delta risk reversal -3 pts" in seen[0]
    assert "Options market report" not in seen[1]


@pytest.mark.unit
def test_options_analyst_is_offered_for_crypto_only():
    assert AnalystType.OPTIONS in filter_analysts_for_asset_type(list(AnalystType), AssetType.CRYPTO)
    assert AnalystType.OPTIONS not in filter_analysts_for_asset_type(list(AnalystType), AssetType.STOCK)
    assert parse_analysts("market,options", AssetType.CRYPTO) == [AnalystType.MARKET, AnalystType.OPTIONS]
    with pytest.raises(ValueError):
        parse_analysts("options", AssetType.STOCK)


@pytest.mark.integration
def test_live_deribit_btc_snapshot():
    out = deribit.get_options_snapshot("BTC-USD")
    assert "## Options snapshot: BTC on Deribit" in out
    assert "| Expiry | DTE |" in out


# ---------------------------------------------------------------------------
# Snapshots
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_a_snapshot_reads_back_as_the_chain_it_was_taken_from(tmp_path):
    from tradingagents.dataflows import options_store

    chain = _chain(open_interest={(10, 80_000.0, "call"): 3.5})
    path = options_store.save_chain(chain, tmp_path)
    assert path == tmp_path / "testvenue" / "BTC" / "2026-10-27T0800Z.csv.gz"
    back = options_store.load_chain(path)
    assert (back.underlying, back.venue, back.spot) == ("BTC", "TestVenue", FORWARD)
    assert back.as_of == AS_OF
    pd.testing.assert_frame_equal(back.quotes, chain.quotes.loc[:, list(om.CHAIN_COLUMNS)])
    assert options_store.list_snapshots(tmp_path, "TestVenue", "btc") == [path]
    assert not list(path.parent.glob("*.partial"))


@pytest.mark.unit
def test_snapshot_command_saves_each_coin_and_reports_a_failure(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    import cli.main as m
    from tradingagents.dataflows import options_store

    def fetch(ticker):
        if ticker == "ETH-USD":
            raise VendorUnavailableError("down")
        return _chain()

    monkeypatch.setitem(m.DEFAULT_CONFIG, "options_snapshot_dir", str(tmp_path))
    monkeypatch.setattr(deribit, "fetch_chain", fetch)
    result = CliRunner().invoke(m.app, ["snapshot-options", "BTC-USD,ETH-USD"])
    assert result.exit_code == 1
    assert "ETH-USD: not saved" in result.output
    assert len(options_store.list_snapshots(tmp_path, "TestVenue", "BTC")) == 1
