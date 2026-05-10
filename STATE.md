# STATE.md

## Current system status
- Repository contains a BIST paper-trading research system
- Database contains OHLCV, indicators, and paper-trading tables
- Existing outputs include daily reports, intraday logs, and model artifacts

## Known concerns
- Encoding issues in some Turkish text output
- Repository is not currently a git repository
- Possible typo under `.claude` folder: `settsings.json`
- Realism of execution, slippage, and cost modeling must be audited
- Data leakage risk must be audited

## Current objective
Turn the repository into a realistic, disciplined, end-to-end paper-trading machine.

## Priority order
1. Repository audit
2. Data leakage audit
3. Execution realism audit
4. Improvement roadmap
5. Risk engine design
6. Transaction cost and slippage model
7. ML signal quality improvements
8. Longer paper-trade validation