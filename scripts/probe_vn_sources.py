"""Probe which Vietnamese financial-statement sources answer from this machine.

Standard library only (no vnstock). Run from the repo root:

    python scripts/probe_vn_sources.py FPT

It prints one line per endpoint and saves the first 4 KB of each successful
response to ``probe_output/`` so the column names can be mapped. Paste the
printed summary (and, if asked, the files) back to the agent thread.
Nothing is sent anywhere except the GET requests listed below.
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

UA = "Mozilla/5.0 (fin-analyst-agent probe)"


def candidates(t: str) -> list[tuple[str, str, dict]]:
    return [
        ("tcbs_income", f"https://apipubaws.tcbs.com.vn/tcanalysis/v1/finance/{t}/incomestatement?yearly=1&isAll=true", {}),
        ("tcbs_balance", f"https://apipubaws.tcbs.com.vn/tcanalysis/v1/finance/{t}/balancesheet?yearly=1&isAll=true", {}),
        ("tcbs_cashflow", f"https://apipubaws.tcbs.com.vn/tcanalysis/v1/finance/{t}/cashflow?yearly=1&isAll=true", {}),
        ("tcbs_ratios", f"https://apipubaws.tcbs.com.vn/tcanalysis/v1/finance/{t}/financialratio?yearly=1&isAll=true", {}),
        ("vndirect_statements", f"https://finfo-api.vndirect.com.vn/v4/financial_statements?q=code:{t}~reportType:ANNUAL&size=50", {}),
        ("vndirect_ratios", f"https://finfo-api.vndirect.com.vn/v4/ratios/latest?filter=code:{t}&size=50", {}),
        ("vietcap_company", f"https://trading.vietcap.com.vn/api/price/symbols/getAll", {}),
        ("cafef_page", f"https://s.cafef.vn/bao-cao-tai-chinh/{t}/IncSta/2024/0/0/0/0/bao-cao-tai-chinh-.chn", {}),
        ("sec_edgar_check", "https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json",
         {"User-Agent": "fin-analyst-agent probe contact@example.com"}),
    ]


def main() -> None:
    ticker = (sys.argv[1] if len(sys.argv) > 1 else "FPT").upper()
    out = Path("probe_output")
    out.mkdir(exist_ok=True)
    print(f"Probing sources for {ticker}\n")
    for name, url, headers in candidates(ticker):
        req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json,text/html", **headers})
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                body = r.read()
                status = r.status
                ctype = r.headers.get("Content-Type", "")
        except Exception as exc:
            print(f"FAIL  {name:22s} {type(exc).__name__}: {str(exc)[:90]}")
            continue
        hint = ""
        if "json" in ctype:
            try:
                doc = json.loads(body)
                if isinstance(doc, list) and doc:
                    hint = f"list[{len(doc)}] keys={list(doc[0])[:8]}" if isinstance(doc[0], dict) else f"list[{len(doc)}]"
                elif isinstance(doc, dict):
                    hint = f"keys={list(doc)[:8]}"
            except ValueError:
                hint = "invalid json"
        (out / f"{name}.txt").write_bytes(body[:4096])
        print(f"OK    {name:22s} HTTP {status} {len(body):>8,d} bytes {ctype[:30]:30s} {hint}")
    print(f"\nSamples saved in {out.resolve()}")


if __name__ == "__main__":
    main()
