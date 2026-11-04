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
- Valuation needs market prices, which the data layer does not provide yet. Say that valuation was not assessed.""",
    ("risk_governance", "all"): """\
- From the statements alone, flag: rising leverage, deteriorating cash conversion, margin compression, and growth funded by debt.
- Governance (related-party deals, ownership, board) needs disclosures that are not in the data: list what should be checked.""",
    ("recommendation", "all"): """\
- Use one of: Accumulate, Hold, Reduce, Avoid. This is analysis, not licensed advice.
- Confidence (low / medium / high) depends on data coverage and agreement between lenses.
- Give bull, base and bear cases, at least three key risks, and what would change the view.
- Without valuation data, a call is about business quality only: say so explicitly.""",
    ("qa", "all"): """\
- Answer the question directly first, then the supporting figures.
- If the data cannot answer the question, say what is missing instead of guessing.""",
}
