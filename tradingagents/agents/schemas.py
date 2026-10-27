"""Pydantic schemas used by agents that produce structured output.

The framework's primary artifact is still prose: each agent's natural-language
reasoning is what users read in the saved markdown reports and what the
downstream agents read as context.  Structured output is layered onto the
three decision-making agents (Research Manager, Trader, Portfolio Manager)
so that:

- Their outputs follow consistent section headers across runs and providers
- Each provider's native structured-output mode is used (json_schema for
  OpenAI/xAI, response_schema for Gemini, tool-use for Anthropic)
- Schema field descriptions become the model's output instructions, freeing
  the prompt body to focus on context and the rating-scale guidance
- A render helper turns the parsed Pydantic instance back into the same
  markdown shape the rest of the system already consumes, so display,
  memory log, and saved reports keep working unchanged
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, field_validator

# LLMs sometimes write a placeholder string ("None", "N/A", ...) into an optional
# numeric field instead of omitting it. Coerce those to None so the structured
# call validates instead of erroring (#1058). Pydantic still parses real numeric
# strings ("189.5") to float.
_NULLISH_FLOAT = {"", "none", "n/a", "na", "null", "nil", "-", "tbd", "unknown"}


def _coerce_optional_float(value):
    """Normalise an LLM-written optional numeric field before validation.

    Three shapes show up in practice: a placeholder string ("None", "N/A") in
    place of an omitted value (#1058); a percentage where a price was asked for
    ("15%", #1288); and a human-formatted price ("$1,234.50"). A percentage
    cannot be salvaged into an absolute level -- reading "15%" as 15 would put a
    stop at $15 on a $600 stock -- so it is dropped like a placeholder, leaving
    one bad field to null out instead of failing the whole proposal. A formatted
    price is reduced to its number.

    Anything that is not a single number is dropped the same way. A range
    ("150-160") or a hedge ("around 150") would otherwise reach pydantic, fail
    validation, and discard the whole decision, losing every field the model got
    right along with the price.
    """
    if not isinstance(value, str):
        return value
    text = value.strip()
    if text.lower() in _NULLISH_FLOAT or text.endswith("%"):
        return None
    cleaned = text.replace(",", "").lstrip("$€£¥").strip()
    try:
        return float(cleaned)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Shared rating types
# ---------------------------------------------------------------------------


class PortfolioRating(StrEnum):
    """5-tier rating used by the Research Manager and Portfolio Manager."""

    BUY = "Buy"
    OVERWEIGHT = "Overweight"
    HOLD = "Hold"
    UNDERWEIGHT = "Underweight"
    SELL = "Sell"


class TraderAction(StrEnum):
    """3-tier transaction direction used by the Trader.

    The Trader's job is to translate the Research Manager's investment plan
    into a concrete transaction proposal: should the desk execute a Buy, a
    Sell, or sit on Hold this round.  Position sizing and the nuanced
    Overweight / Underweight calls happen later at the Portfolio Manager.
    """

    BUY = "Buy"
    HOLD = "Hold"
    SELL = "Sell"


# ---------------------------------------------------------------------------
# Research Manager
# ---------------------------------------------------------------------------


class ResearchPlan(BaseModel):
    """Structured investment plan produced by the Research Manager.

    Hand-off to the Trader: the recommendation pins the directional view,
    the rationale captures which side of the bull/bear debate carried the
    argument, and the strategic actions translate that into concrete
    instructions the trader can execute against.
    """

    recommendation: PortfolioRating = Field(
        description=(
            "The investment recommendation. Exactly one of Buy / Overweight / "
            "Hold / Underweight / Sell. Conflicting arguments alone are not a "
            "reason to Hold: commit to the stronger side, sized by how "
            "decisively it wins. Choose Hold only when the evidence is still "
            "balanced after weighing, or too thin to support a call."
        ),
    )
    rationale: str = Field(
        description=(
            "Conversational summary of the key points from both sides of the "
            "debate, ending with which arguments led to the recommendation. "
            "Speak naturally, as if to a teammate."
        ),
    )
    strategic_actions: str = Field(
        description=(
            "Concrete steps for the trader to implement the recommendation, "
            "including sizing guidance relative to a standard allocation. The "
            "research team does not see the caller's holdings; the trader and "
            "portfolio manager apply the actual position."
        ),
    )


def render_research_plan(plan: ResearchPlan) -> str:
    """Render a ResearchPlan to markdown for storage and the trader's prompt context."""
    return "\n".join([
        f"**Recommendation**: {plan.recommendation.value}",
        "",
        f"**Rationale**: {plan.rationale}",
        "",
        f"**Strategic Actions**: {plan.strategic_actions}",
    ])


# ---------------------------------------------------------------------------
# Trader
# ---------------------------------------------------------------------------


class TraderProposal(BaseModel):
    """Structured transaction proposal produced by the Trader.

    The trader reads the Research Manager's investment plan and the analyst
    reports, then turns them into a concrete transaction: what action to
    take, the reasoning that justifies it, and the practical levels for
    entry, stop-loss, and sizing.
    """

    action: TraderAction = Field(
        description="The transaction direction. Exactly one of Buy / Hold / Sell.",
    )
    reasoning: str = Field(
        description=(
            "The case for this action, anchored in the analysts' reports and "
            "the research plan. Two to four sentences."
        ),
    )
    entry_price: float | None = Field(
        default=None,
        description=(
            "Optional entry price target as an absolute number in the instrument's "
            "quote currency (e.g. 189.5), never a percentage or a range. Omit it "
            "if you cannot state a specific level."
        ),
    )
    stop_loss: float | None = Field(
        default=None,
        description=(
            "Optional stop-loss as an absolute price in the instrument's quote "
            "currency (e.g. 172.0), never a percentage. Convert a percentage "
            "distance to the price level it implies, or omit it."
        ),
    )
    position_sizing: str | None = Field(
        default=None,
        description="Optional sizing guidance, e.g. '5% of portfolio'.",
    )

    @field_validator("entry_price", "stop_loss", mode="before")
    @classmethod
    def _nullish_float_to_none(cls, v):
        return _coerce_optional_float(v)


def render_trader_proposal(proposal: TraderProposal) -> str:
    """Render a TraderProposal to markdown.

    The trailing ``FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL**`` line is
    preserved for backward compatibility with the analyst stop-signal text
    and any external code that greps for it.
    """
    parts = [
        f"**Action**: {proposal.action.value}",
        "",
        f"**Reasoning**: {proposal.reasoning}",
    ]
    # Named even when absent, so a reader can tell a level the trader chose not
    # to give from one the schema never asked for.
    for label, value in (("Entry Price", proposal.entry_price),
                         ("Stop Loss", proposal.stop_loss),
                         ("Position Sizing", proposal.position_sizing)):
        parts.extend(["", f"**{label}**: {value if value is not None and value != '' else 'not provided'}"])
    parts.extend([
        "",
        f"FINAL TRANSACTION PROPOSAL: **{proposal.action.value.upper()}**",
    ])
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Portfolio Manager
# ---------------------------------------------------------------------------


class PortfolioDecision(BaseModel):
    """Structured output produced by the Portfolio Manager.

    The model fills every field as part of its primary LLM call; no separate
    extraction pass is required. Field descriptions double as the model's
    output instructions, so the prompt body only needs to convey context and
    the rating-scale guidance.
    """

    rating: PortfolioRating = Field(
        description=(
            "The final position rating. Exactly one of Buy / Overweight / Hold / "
            "Underweight / Sell, picked based on the analysts' debate. "
            "Conflicting arguments alone are not a reason to Hold: commit to the "
            "stronger side, sized by how decisively it wins. Choose Hold only "
            "when the evidence is still balanced after weighing, or too thin to "
            "support a call."
        ),
    )
    executive_summary: str = Field(
        description=(
            "A concise action plan covering entry strategy, position sizing, "
            "key risk levels, and time horizon. Two to four sentences."
        ),
    )
    investment_thesis: str = Field(
        description=(
            "Detailed reasoning anchored in specific evidence from the analysts' "
            "debate. If prior lessons are referenced in the prompt context, "
            "incorporate them; otherwise rely solely on the current analysis."
        ),
    )
    price_target: float | None = Field(
        default=None,
        description="Optional target price in the instrument's quote currency.",
    )
    time_horizon: str | None = Field(
        default=None,
        description="Optional recommended holding period, e.g. '3-6 months'.",
    )

    @field_validator("price_target", mode="before")
    @classmethod
    def _nullish_float_to_none(cls, v):
        return _coerce_optional_float(v)


def render_pm_decision(decision: PortfolioDecision) -> str:
    """Render a PortfolioDecision back to the markdown shape the rest of the system expects.

    Memory log, CLI display, and saved report files all read this markdown,
    so the rendered output preserves the exact section headers (``**Rating**``,
    ``**Executive Summary**``, ``**Investment Thesis**``) that downstream
    parsers and the report writers already handle.
    """
    parts = [
        f"**Rating**: {decision.rating.value}",
        "",
        f"**Executive Summary**: {decision.executive_summary}",
        "",
        f"**Investment Thesis**: {decision.investment_thesis}",
    ]
    # Named even when absent: a missing line reads as a field nobody asked for,
    # so a reader cannot tell "no target" from "target not reported".
    target = decision.price_target if decision.price_target is not None else "not provided"
    parts.extend(["", f"**Price Target**: {target}"])
    parts.extend(["", f"**Time Horizon**: {decision.time_horizon or 'not provided'}"])
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Sentiment Analyst
# ---------------------------------------------------------------------------


class SentimentBand(StrEnum):
    """Discrete sentiment direction produced by the Sentiment Analyst.

    Six tiers keep the signal granular enough to be actionable while remaining
    small enough for every provider to map reliably from its JSON output.
    """

    BULLISH = "Bullish"
    MILDLY_BULLISH = "Mildly Bullish"
    NEUTRAL = "Neutral"
    MIXED = "Mixed"
    MILDLY_BEARISH = "Mildly Bearish"
    BEARISH = "Bearish"


class SentimentReport(BaseModel):
    """Structured sentiment report produced by the Sentiment Analyst.

    Replaces the previous free-form prose output so downstream consumers
    (dashboards, audit logs, PDF renderers, other agents) can read
    ``overall_band`` and ``overall_score`` without maintaining fragile regex
    fallbacks that drift with every model release. ``narrative`` preserves the
    rich source-by-source analysis; ``render_sentiment_report`` prepends a
    deterministic header so the saved report stays human-readable.
    """

    overall_band: SentimentBand = Field(
        description=(
            "Overall sentiment direction. Exactly one of: "
            "Bullish / Mildly Bullish / Neutral / Mixed / Mildly Bearish / Bearish. "
            "Use Mixed when sources point in clearly different directions. "
            "Use Neutral only when all sources are genuinely silent or non-committal."
        ),
    )
    overall_score: float = Field(
        ge=0.0,
        le=10.0,
        description=(
            "Numeric sentiment intensity on a 0–10 scale. "
            "0 = maximally bearish, 5 = neutral, 10 = maximally bullish. "
            "Guideline for consistency with overall_band: "
            "Bullish ~6.5–10, Mildly Bullish ~5.5–6.4, Neutral/Mixed ~4.5–5.5, "
            "Mildly Bearish ~3.5–4.4, Bearish ~0–3.4. "
            "Only the 0–10 bounds are enforced."
        ),
    )
    confidence: Literal["low", "medium", "high"] = Field(
        description=(
            "Confidence in the assessment based on data quality and sample size. "
            "Use 'low' when one or more sources returned a placeholder or fewer "
            "than 5 data points; 'medium' when data is present but sparse; "
            "'high' when all three sources returned substantive data."
        ),
    )
    narrative: str = Field(
        description=(
            "Full sentiment report covering, in order: "
            "(1) source-by-source breakdown with specific evidence (cite message "
            "counts, ratios, notable posts); "
            "(2) cross-source divergences and alignments; "
            "(3) dominant narrative themes; "
            "(4) catalysts and risks surfaced by the data; "
            "(5) a markdown table summarising key sentiment signals, their "
            "direction, source, and supporting evidence. "
            "Keep it informative and substantive: develop each section thoroughly "
            "with concrete evidence so every point adds new signal for the trader."
        ),
    )


def render_sentiment_report(report: SentimentReport) -> str:
    """Render a SentimentReport to the markdown shape the rest of the system expects.

    The structured header (band + score + confidence) is prepended to the
    narrative so the saved report is both human-readable and machine-parseable
    without regex.
    """
    return "\n".join([
        f"**Overall Sentiment:** **{report.overall_band.value}** "
        f"(Score: {report.overall_score:.1f}/10)",
        f"**Confidence:** {report.confidence.capitalize()}",
        "",
        report.narrative,
    ])


# ---------------------------------------------------------------------------
# Options mode
#
# With ``decision_mode: "options"`` the Research Manager adds a volatility view
# to its directional call, the Trader proposes an options structure instead of a
# buy or sell, and the Portfolio Manager approves or rejects that structure.
# The structure is priced in code against the chain, never by the model.
# ---------------------------------------------------------------------------


class VolatilityView(StrEnum):
    """Whether options are worth more or less than the move they price."""

    LONG = "Long volatility"
    NEUTRAL = "Neutral volatility"
    SHORT = "Short volatility"


class OptionsResearchPlan(ResearchPlan):
    """The research plan plus the volatility view and horizon an options structure needs."""

    volatility_view: VolatilityView = Field(
        description=(
            "Exactly one of Long volatility / Neutral volatility / Short volatility. "
            "Long when the move you expect is larger than the options price (implied "
            "vol cheap against the realized vol you expect); Short when options price "
            "more movement than you expect; Neutral when they look fair."
        ),
    )
    volatility_rationale: str = Field(
        description=(
            "The evidence for the volatility view: implied against realized "
            "volatility, IV rank, term structure, skew and any expected event."
        ),
    )
    horizon_days: int | None = Field(
        default=None,
        description="Days over which the view should play out, e.g. 30. Omit if unclear.",
    )


def render_options_research_plan(plan: OptionsResearchPlan) -> str:
    """Render an OptionsResearchPlan: the research plan's sections plus the volatility view."""
    horizon = f"{plan.horizon_days} days" if plan.horizon_days else "not provided"
    return "\n".join([
        render_research_plan(plan),
        "",
        f"**Volatility View**: {plan.volatility_view.value}",
        "",
        f"**Volatility Rationale**: {plan.volatility_rationale}",
        "",
        f"**Horizon**: {horizon}",
    ])


class OptionsStrategy(StrEnum):
    """The structures the strategist may propose: defined risk, one expiry."""

    LONG_CALL = "Long call"
    LONG_PUT = "Long put"
    BULL_CALL_SPREAD = "Bull call spread"
    BEAR_PUT_SPREAD = "Bear put spread"
    BULL_PUT_SPREAD = "Bull put spread"
    BEAR_CALL_SPREAD = "Bear call spread"
    IRON_CONDOR = "Iron condor"
    IRON_BUTTERFLY = "Iron butterfly"
    LONG_STRADDLE = "Long straddle"
    LONG_STRANGLE = "Long strangle"
    CASH_SECURED_PUT = "Cash-secured put"
    NO_TRADE = "No trade"


class OptionLegSpec(BaseModel):
    """One leg of a proposed structure."""

    side: Literal["buy", "sell"] = Field(description="buy or sell")
    option_type: Literal["call", "put"] = Field(description="call or put")
    strike: float = Field(description="Strike price in USD, e.g. 84000. Use a strike from the strike guide.")
    expiry: str = Field(description="Expiry date YYYY-MM-DD, one of the listed expiries in the strike guide.")
    quantity: float = Field(
        default=1.0,
        description="Units of the underlying per structure, normally 1; 2 for a ratio leg.",
    )

    @field_validator("strike", mode="before")
    @classmethod
    def _plain_number(cls, v):
        # "$84,000" is a strike; a placeholder is not, and must fail the leg.
        return _coerce_optional_float(v) if isinstance(v, str) else v


class OptionsTradeProposal(BaseModel):
    """The strategist's structure: legs priced afterwards in code against the chain."""

    strategy: OptionsStrategy = Field(
        description=(
            "The structure. Pick the one that expresses both the direction and the "
            "volatility view; No trade when neither gives an edge."
        ),
    )
    legs: list[OptionLegSpec] = Field(
        default_factory=list,
        description="One to four legs, all on the same expiry. Empty only for No trade.",
    )
    rationale: str = Field(
        description=(
            "Why this structure, expiry and these strikes fit the research plan's "
            "direction, volatility view and horizon. Two to five sentences."
        ),
    )
    exit_plan: str = Field(
        description=(
            "When to take profit, when to cut the trade, and the date or event by "
            "which to close it if neither happens."
        ),
    )
    risk_budget_pct: float | None = Field(
        default=None,
        description=(
            "Suggested maximum loss for this trade as a percentage of capital, e.g. "
            "1.0 for 1%. Omit if you cannot state one."
        ),
    )

    @field_validator("risk_budget_pct", mode="before")
    @classmethod
    def _nullish_float_to_none(cls, v):
        if isinstance(v, str) and v.strip().endswith("%"):
            v = v.strip()[:-1]
        return _coerce_optional_float(v)


def render_options_proposal(proposal: OptionsTradeProposal) -> str:
    """Render a proposal's own fields; the priced structure check follows it."""
    parts = [f"**Strategy**: {proposal.strategy.value}", ""]
    if proposal.legs:
        parts.append("**Legs**:")
        parts += [
            f"- {leg.side} {leg.quantity:g} x {leg.expiry} {leg.strike:g} {leg.option_type}"
            for leg in proposal.legs
        ]
        parts.append("")
    budget = f"{proposal.risk_budget_pct:g}% of capital" if proposal.risk_budget_pct is not None else "not provided"
    parts += [
        f"**Rationale**: {proposal.rationale}",
        "",
        f"**Exit Plan**: {proposal.exit_plan}",
        "",
        f"**Risk Budget**: {budget}",
    ]
    return "\n".join(parts)


class StructureDecision(StrEnum):
    APPROVE = "Approve"
    REJECT = "Reject"


class OptionsPortfolioDecision(BaseModel):
    """The Portfolio Manager's verdict on the proposed structure, plus the directional rating."""

    rating: PortfolioRating = Field(
        description=(
            "The directional rating on the underlying. Exactly one of Buy / "
            "Overweight / Hold / Underweight / Sell; a non-directional structure "
            "such as an iron condor goes with Hold."
        ),
    )
    structure_decision: StructureDecision = Field(
        description=(
            "Approve the proposed structure as priced, or Reject it. Reject when its "
            "risk, cost, liquidity or fit with the research plan does not hold up; "
            "say in the thesis what would have to change."
        ),
    )
    risk_budget_pct: float | None = Field(
        default=None,
        description=(
            "For an approved structure, the maximum loss to accept as a percentage "
            "of capital, e.g. 1.0 for 1%. Omit when rejecting."
        ),
    )
    executive_summary: str = Field(
        description="The decision and how to act on it. Two to four sentences.",
    )
    investment_thesis: str = Field(
        description=(
            "The evidence that decided it, from the research plan, the priced "
            "structure check and the risk debate; for a rejection, what would make "
            "a structure acceptable. If prior lessons are in the prompt, use them."
        ),
    )
    exit_plan: str = Field(
        description="Profit target, loss limit, and the date or event by which to close.",
    )

    @field_validator("risk_budget_pct", mode="before")
    @classmethod
    def _nullish_float_to_none(cls, v):
        if isinstance(v, str) and v.strip().endswith("%"):
            v = v.strip()[:-1]
        return _coerce_optional_float(v)


def render_options_pm_decision(decision: OptionsPortfolioDecision) -> str:
    """Render the decision with the ``**Rating**`` line first, as every rating reader expects."""
    budget = (f"{decision.risk_budget_pct:g}% of capital at maximum loss"
              if decision.risk_budget_pct is not None else "not provided")
    return "\n".join([
        f"**Rating**: {decision.rating.value}",
        "",
        f"**Structure Decision**: {decision.structure_decision.value}",
        "",
        f"**Risk Budget**: {budget}",
        "",
        f"**Executive Summary**: {decision.executive_summary}",
        "",
        f"**Investment Thesis**: {decision.investment_thesis}",
        "",
        f"**Exit Plan**: {decision.exit_plan}",
    ])
