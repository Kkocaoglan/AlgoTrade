# ROADMAP.md

## Phase 1: Repository Integrity

### Objective
Establish a clean, auditable, low-friction repository baseline before touching trading logic.

### Files likely involved
- `AGENTS.md`
- `PROJECT.md`
- `STATE.md`
- `TASKS.md`
- `CLAUDE.md`
- `run_daily.bat`
- `.claude/settings.local.json`
- `.claude/settsings.json`

### Why it matters
- Current repo hygiene is weak.
- There are encoding issues, duplicated operational memory, and config ambiguity.
- Later fixes will be riskier without a stable baseline.

### Expected benefit
- Safer iterations
- Clearer ownership of project rules and workflow
- Lower chance of accidental regressions

### Verification method
- Docs agree on current objective and workflow
- Config ambiguity is removed
- Daily command flow is documented and reproducible

## Phase 2: Data And Feature Integrity

### Objective
Remove temporal leakage, enforce causal features, and align training features with live inference features.

### Files likely involved
- `fetch_data.py`
- `indicators.py`
- `ml_train.py`
- `paper_trade.py`

### Why it matters
- Current split logic leaks future label information across boundaries.
- Walk-forward validation is not properly purged.
- Feature code is duplicated between training and live inference.

### Expected benefit
- Trustworthy evaluation
- Lower hidden overfitting risk
- Consistent train/live feature behavior

### Verification method
- Purged split checks around `HORIZON` pass
- Shared feature generation produces identical outputs in train and live paths
- Synthetic tests confirm no future information is used

## Phase 3: Execution Realism

### Objective
Replace optimistic fills and idealized exits with realistic execution and barrier handling.

### Files likely involved
- `backtest.py`
- `paper_trade.py`
- `run_daily.bat`

### Why it matters
- Current entries are effectively same-bar fills.
- Stops and targets are not gap-aware.
- Delayed intraday prices are treated like executable quotes.

### Expected benefit
- More realistic backtests
- More credible paper-trading behavior
- Better alignment between research and execution assumptions

### Verification method
- Entries execute on next bar, not signal bar
- Stop/target logic uses `open/high/low/close` consistently
- Gap-through scenarios are tested explicitly
- Delayed quote feeds no longer auto-execute orders

## Phase 4: Portfolio Risk Engine

### Objective
Add portfolio-level controls beyond simple position count and sector filters.

### Files likely involved
- `paper_trade.py`
- `daily_report.py`
- `backtest.py`

### Why it matters
- Current system controls single-trade risk better than total portfolio risk.
- Missing controls include gross exposure, concentration, and daily loss limits.

### Expected benefit
- Lower blow-up risk
- Better concentration discipline
- More realistic portfolio behavior under stress

### Verification method
- Simulated scenarios trigger portfolio risk blocks as expected
- Reports include portfolio-level risk metrics
- New trades are blocked when exposure or drawdown rules are hit

## Phase 5: ML Signal Quality

### Objective
Improve the usefulness of the ML signal after leakage and execution assumptions are corrected.

### Files likely involved
- `ml_train.py`
- `paper_trade.py`
- `optimize.py`
- `indicators.py`

### Why it matters
- Current reported accuracy is not the same as traded usefulness.
- Confidence thresholds are used live but not evaluated in a fully aligned way offline.
- Signal quality should be judged by tradable precision and portfolio contribution.

### Expected benefit
- Better signal selectivity
- Fewer low-quality trades
- Metrics that better reflect traded outcomes

### Verification method
- Report both full-model and traded-only metrics
- Evaluate precision/recall at actual confidence thresholds
- Review stability by symbol, sector, and regime

## Phase 6: Paper-Trade Validation

### Objective
Turn the system into a disciplined multi-week paper-trading validation process.

### Files likely involved
- `paper_trade.py`
- `daily_report.py`
- `news_filter.py`
- `run_daily.bat`
- `STATE.md`

### Why it matters
- The current workflow runs daily, but validation criteria are not formalized.
- News and macro signals are advisory rather than fully integrated into trading controls.

### Expected benefit
- Measurable paper-trade quality
- Clear go/no-go criteria
- Better operational discipline before any live integration

### Verification method
- Define a fixed validation window and KPI set
- Daily reports include validation metrics and failure conditions
- Risk and news gates can be demonstrated in controlled scenarios

## Phase 7: Live-Readiness Checklist

### Objective
Define the conditions that must be met before any broker/API-based live trading work begins.

### Files likely involved
- `PROJECT.md`
- `STATE.md`
- `TASKS.md`
- `paper_trade.py`
- `daily_report.py`

### Why it matters
- Without a hard checklist, live trading decisions become subjective.
- Current system is not yet ready for real-money execution.

### Expected benefit
- Prevents premature deployment
- Makes readiness evidence-based
- Provides a concrete finish line for the project

### Verification method
- Checklist is binary, specific, and evidence-backed
- Examples:
  - no known leakage
  - next-bar execution only
  - gap-aware stop handling
  - enforced portfolio risk limits
  - stable paper-trade validation over a defined period
  - operational failure handling documented

## Suggested Implementation Order

1. Repository cleanup and baseline rules
2. Leakage fixes and feature parity
3. Execution realism refactor
4. Portfolio risk controls
5. ML metric alignment and signal improvement
6. Formal paper-trade validation
7. Live-readiness checklist and approval gate
