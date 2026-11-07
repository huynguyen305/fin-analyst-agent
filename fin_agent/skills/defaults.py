"""Built-in skill text, used until the matching vault rule note exists.

These are deliberately short starting rules. The intent is that Huy's vault
notes (``Analyst_Skills/``) replace them; each run reports which source it used.
"""

DEFAULT_SKILLS: dict[tuple[str, str], str] = {
    ("profitability", "all"): """\
- Decompose ROE with DuPont: net margin x asset turnover x equity multiplier. Say which driver moved ROE.
- Rising ROE driven mainly by a higher equity multiplier is leverage, not better operations: flag it.
- Compare gross margin and operating margin trends: diverging trends point to opex (SG&A) changes.
- Revenue growth with falling margins suggests price competition or input-cost pressure.""",
    ("profitability", "bank"): """\
- For banks, use NII / average loans as a NIM proxy, ROE, ROA and the equity multiplier.
- Gross margin, CCC and inventory metrics do not apply to banks: never use them.
- ROA above about 1.5% is strong for a Vietnamese bank; below 1% is weak.""",
    ("health", "all"): """\
- Current ratio below 1.0x or quick ratio below 0.7x: liquidity watch.
- Debt to equity above 2.0x for a non-financial firm is high; say whether it is rising.
- Interest coverage below 3x is a concern; below 1.5x is a red flag.
- Real-estate developers carry high liabilities by design (customer advances); note this before judging leverage.""",
    ("health", "bank"): """\
- LDR (loans / deposits) above about 85% means funding pressure; State Bank of Vietnam caps apply to the regulatory LDR, which differs from this simple ratio.
- NPL ratio above 3% is a warning; reserve coverage below 100% is thin.
- Equity / assets below 7% means thin capital; CAR is not in the statements, so say it is not assessed.""",
    ("operational", "all"): """\
- Cash conversion cycle = DSO + DIO - DPO. A rising CCC ties up cash; find which component drove it.
- Receivables growing faster than revenue (rising DSO) can mean looser credit or aggressive revenue recognition.
- Inventory build-up (rising DIO) ahead of falling demand is a margin risk.
- Capex / revenue rising for several years: check whether revenue growth follows.""",
    ("operational", "bank"): """\
- Compare loan growth with deposit growth: loans outrunning deposits needs other funding.
- Very fast loan growth (above about 20% a year) often precedes asset-quality problems.""",
    ("cashflow_quality", "all"): """\
- CFO / net income consistently below 0.8x: earnings are not turning into cash; investigate accruals.
- A rising accruals ratio is a classic earnings-quality warning.
- Dividends not covered by free cash flow (cover below 1.0x) are funded by debt or cash reserves.
- One weak year is noise; flag multi-year patterns.""",
    ("valuation", "all"): """\
- Multiples use the latest close and the latest fiscal year (or trailing four quarters); state the price date.
- Read P/B against ROE: a high P/B is only justified by a durably high ROE. Read P/E against earnings growth.
- Negative or tiny earnings make P/E meaningless (it is then not computed): use P/B or EV/EBIT instead.
- Compare with the stock's own history or sector peers only if those figures are given; never invent peer multiples.
- For banks, P/B with ROE is the main lens; EV-based multiples do not apply.""",
    ("technical", "all"): """\
- Technicals describe price behaviour, not value. Use them for timing and risk, never as the reason a business is good or bad.
- Trend: price above a rising 200-day average is an uptrend; below it, a downtrend. 50-day above 200-day (golden-cross regime) confirms; below (death-cross regime) warns.
- Momentum: RSI above 70 is overbought, below 30 oversold; in a strong trend RSI can stay extreme, so read it with the trend. A positive MACD histogram means momentum is improving.
- Risk: annualized volatility above about 40% or a 12-month drawdown worse than -30% means a position needs smaller size or wider tolerance.
- Volume: a move on rising volume (20/60-day ratio above 1.2) is more credible than one on thin volume.
- Relative strength: compare returns with the VN-Index; persistent underperformance needs a fundamental explanation.
- Vietnam: daily price limits (HOSE +/-7%, HNX +/-10%, UPCoM +/-15%) and T+2 settlement can exaggerate short-term moves; thinly traded stocks give unreliable signals.
- Never give a price target or say a level "will" be reached. State signals as conditions, e.g. "a close back above the 200-day average would end the downtrend".""",
    ("scenarios", "all"): """\
- These are conditional scenarios, not forecasts. Never call one "the target", "expected" or "most likely", and never say a price "will" be reached.
- State each scenario with its assumption: bear = the year's lowest multiple with weak growth, base = today's multiple with typical growth, bull = the year's highest multiple with strong growth.
- The statistical range is an 80% band that assumes past volatility persists and no drift: it shows how far the stock normally moves, not where it is heading.
- Read the trend projection only with its R-squared: below about 30% there is no reliable trend, so say the projection carries little information.
- Compare the scenarios with the 12-month range: a bull or bear price outside the 80% band needs an unusually large move, so say so.
- A wide gap between bear and bull means high uncertainty; position size, not conviction, should reflect it.
- Growth percentiles come from few years of history: one exceptional year moves them, so say how many years back the rates.
- For banks and loss-makers the driver is book value (P/B), not earnings.
- Vietnam: daily price limits (HOSE +/-7%) and thin trading can make the ranges understate gap risk.""",
    ("risk_governance", "all"): """\
- From the statements alone, flag: rising leverage, deteriorating cash conversion, margin compression, and growth funded by debt.
- Governance (related-party deals, ownership, board) needs disclosures that are not in the data: list what should be checked.""",
    ("recommendation", "all"): """\
- Use one of: Accumulate, Hold, Reduce, Avoid. This is analysis, not licensed advice.
- Confidence (low / medium / high) depends on data coverage and agreement between lenses.
- Give bull, base and bear cases, at least three key risks, and what would change the view.
- If valuation was not assessed, the call is about business quality only: say so explicitly.
- If it was, weigh quality against price: a good business at a stretched multiple is not automatically Accumulate.
- Technicals shape timing and position size, not the call itself: a quality business in a downtrend can still be Accumulate, staged over time.
- When price scenarios were assessed, the bull, base and bear cases may quote their scenario prices as conditional ranges with their assumptions, never as targets.""",
    ("qa", "all"): """\
- Answer the question directly first, then the supporting figures.
- If the data cannot answer the question, say what is missing instead of guessing.""",
}
