# LIVE_READINESS.md

## Purpose

This checklist defines the minimum conditions that must be satisfied before any broker/API-based live trading work begins.

It is intentionally binary and evidence-driven. A box is not complete unless the required evidence exists and is current.

Current posture based on `AUDIT.md`, `ROADMAP.md`, and repository state on 2026-04-03:

- Live trading is **not ready**
- The checklist below is the approval gate
- Any single failed critical item means **no live deployment**

## Approval Rule

- All `Critical` items must be checked
- No open known issue may exist for data leakage, future information use, or unrealistic execution assumptions
- Paper-trade validation criteria must pass over the full validation window before live deployment is considered
- Final go/no-go must be documented with date, owner, and evidence links

## Evidence Convention

For each completed item, record:

- Evidence:
- Date verified:
- Owner:
- Notes:

---

## 1. Data Integrity Checks

### Critical

- [ ] Daily OHLCV ingestion completes without missing symbols or duplicate rows
- [ ] Indicator generation completes for all active symbols and dates expected for that run
- [ ] No forward-filled future data is used in features or labels
- [ ] Train/test and walk-forward boundaries are purged by at least `HORIZON` rows
- [ ] Validation uses forward-chaining only; no shuffled or mixed-time evaluation remains
- [ ] Train/live feature generation is sourced from one logically identical implementation, or parity is tested and proven
- [ ] Any non-causal indicator logic is removed from the live or ML feature path
- [ ] Latest paper-trade run uses only data available as of decision time

### Strongly Recommended

- [ ] Daily sanity check exists for symbol count, date freshness, null rate, and extreme value spikes
- [ ] A lightweight schema/version check runs before training and paper trading
- [ ] Feature parity test exists comparing training-time and paper-trade feature outputs on the same symbol/date slice

### Required Evidence

- Output from a focused integrity verification run
- Documented purge logic around split boundaries
- Feature parity check result

---

## 2. Model Integrity Checks

### Critical

- [ ] Model artifact contains the expected model, feature list, calibration object, and thresholds
- [ ] Probability outputs are calibrated, or calibration has been tested and shown unnecessary with evidence
- [ ] Thresholds are tuned for traded usefulness, not default classifier accuracy
- [ ] No-trade zone is active and used consistently in offline evaluation and paper trading
- [ ] Evaluation reports traded-only metrics on the full decision universe, including neutral/no-trade outcomes
- [ ] BUY precision is reported at the actual deployed threshold
- [ ] Class imbalance treatment is explicitly documented and validated against the real decision universe
- [ ] Walk-forward model evaluation preserves time order and does not use holdout data for calibration or threshold search
- [ ] Model behavior is stable across symbols and regimes, or deployment is restricted to validated subsets

### Strongly Recommended

- [ ] Reliability curve or calibration-bin summary is archived for the latest approved model
- [ ] Threshold report includes precision, coverage, neutral-rate, and trade counts
- [ ] Feature stability review exists for redundancy, importance drift, and fragile proxy features
- [ ] A minimum trade-count rule exists before accepting tuned buy/sell thresholds

### Required Evidence

- Latest `ml_train.py` verification output
- Saved threshold/calibration metadata
- Latest walk-forward summary with traded metrics

---

## 3. Execution Realism Checks

### Critical

- [ ] Entries execute on next bar or later, never on the signal bar close
- [ ] Stop and target handling is gap-aware and uses realistic OHLC barrier logic
- [ ] Slippage and commissions are applied on both entry and exit
- [ ] Intraday watch/monitor mode does not treat delayed quotes as executable fills
- [ ] Exit logic is consistent between backtest assumptions and paper-trade assumptions, or differences are documented and justified
- [ ] Partial exits and trailing stops use explicit, reproducible fill rules
- [ ] A failed market data fetch cannot silently create or close positions

### Strongly Recommended

- [ ] Slippage assumptions are volatility- or liquidity-aware rather than fixed constants
- [ ] A paper-trade dry run exists for stop-gap, target-gap, and no-fill edge cases
- [ ] Run timing guarantees signals are based only on completed bars

### Required Evidence

- Focused scenario test for gap-through stop and target behavior
- Verified order-timing note showing next-bar execution rule
- Current cost/slippage assumptions documented

---

## 4. Risk Controls

### Critical

- [ ] Per-trade risk sizing is enforced from stop distance, not nominal allocation alone
- [ ] Max position count is enforced
- [ ] Max single-position size is enforced
- [ ] Sector concentration limits are enforced
- [ ] Gross exposure limit is enforced
- [ ] Net exposure limit is enforced
- [ ] Cash buffer / insufficient-cash block is enforced
- [ ] Daily loss kill-switch is enforced
- [ ] Portfolio drawdown kill-switch is enforced
- [ ] New trade entry is blocked when portfolio risk state is breached
- [ ] News/macro risk gates that claim to block trades are truly enforced, not advisory only

### Strongly Recommended

- [ ] Correlated exposure limits exist beyond simple sector caps
- [ ] Position sizing scales down under elevated volatility or drawdown
- [ ] Risk report shows concentration, stop proximity, and regime mismatch daily

### Required Evidence

- Controlled verification showing each major block condition triggers as intended
- Latest daily report with exposure and risk summary
- Documented configured limits and rationale

---

## 5. Operational Monitoring

### Critical

- [ ] Daily report is generated successfully after each paper-trade run
- [ ] Daily report includes realized P&L, unrealized P&L, exposure, and risk summary
- [ ] Open positions include entry reason, current thesis, and exit conditions where available
- [ ] Closed trades include exit reason and fill context
- [ ] Event log captures entries, exits, partial exits, blocks, and informational run outcomes
- [ ] Operational failures are visible: model load failure, missing data, stale data, report failure, or DB write failure cannot pass silently
- [ ] There is a clear operator-facing status command or report for current portfolio state

### Strongly Recommended

- [ ] Regime summary is included in daily monitoring
- [ ] A runbook exists for daily operations, failure handling, and restart steps
- [ ] Artifacts are timestamped and retained for audit review

### Required Evidence

- Latest daily report artifact
- Sample event log output
- One successful end-to-end daily run log

---

## 6. Kill-Switch Conditions

Any of the following should force `NO NEW TRADES`, and some should force full liquidation depending on severity.

### Hard Stop: No New Trades

- [ ] Data missing, stale, or incomplete for any required symbol
- [ ] Model artifact missing, unreadable, or schema-incompatible
- [ ] Feature parity check fails
- [ ] Calibration or threshold metadata missing from the active model artifact
- [ ] Reported daily loss exceeds configured maximum
- [ ] Portfolio drawdown exceeds configured maximum
- [ ] Gross or net exposure limit breach
- [ ] Repeated DB write/read integrity issue
- [ ] Macro/news hard-risk gate breach
- [ ] Manual operator halt flag exists and is respected

### Hard Stop: Flatten Portfolio / Disable System

- [ ] Evidence of data leakage or future-information contamination in the active model
- [ ] Live execution path deviates from tested paper-trade assumptions
- [ ] Price feed or broker acknowledgement becomes unreliable
- [ ] P&L reconciliation fails or position state cannot be trusted
- [ ] Unexplained abnormal loss, fill behavior, or repeated execution anomaly

### Required Evidence

- Documented kill-switch logic and who can trigger it
- Verified simulation or controlled test of at least one portfolio halt path

---

## 7. Paper-Trade Validation Criteria Before Live Deployment

### Minimum Validation Window

- [ ] At least 8 consecutive weeks of paper trading completed
- [ ] Validation window includes both favorable and unfavorable market regimes
- [ ] No material process changes were made mid-window; otherwise validation resets

### Performance / Quality Gates

- [ ] BUY precision at deployed threshold is stable and trade-useful
- [ ] Trade coverage is sufficient for the intended strategy frequency
- [ ] Realized P&L is positive after costs over the validation window, or a documented alternative criterion is approved
- [ ] Max drawdown remains within predefined tolerance
- [ ] Average loss, payoff ratio, and hit rate are acceptable for the strategy design
- [ ] Performance is not driven by one symbol, one week, or one regime only
- [ ] Neutral/no-trade zone meaningfully reduces low-quality trades

### Risk / Operations Gates

- [ ] No unresolved stale-data or failed-run incidents remain
- [ ] Risk blocks and macro/news gates trigger correctly during the window
- [ ] Daily reports are complete and internally consistent throughout the window
- [ ] Entry/exit explanations are available for audit review
- [ ] No unexplained divergence exists between ML offline metrics and paper-trade outcomes

### Required Evidence

- Fixed validation period dates
- Daily reports covering the whole window
- Summary KPI sheet for precision, coverage, drawdown, realized P&L, and incident count

---

## 8. Final Live Go/No-Go Signoff

- [ ] Data integrity approved
- [ ] Model integrity approved
- [ ] Execution realism approved
- [ ] Risk controls approved
- [ ] Operational monitoring approved
- [ ] Kill-switch behavior approved
- [ ] Paper-trade validation approved
- [ ] Final deployment scope is defined:
  - symbols:
  - max capital:
  - max positions:
  - order types:
  - operator:
- [ ] Rollback plan is documented
- [ ] First-live-day supervision plan is documented

## Signoff

- Decision: `GO / NO-GO`
- Date:
- Approved by:
- Notes:

---

## Current Known Gaps To Resolve Before Any Live Work

Based on the current repository audit and roadmap, these items should be treated as open until explicitly verified closed:

- [ ] Purged split integrity is proven in `ml_train.py`
- [ ] Train/live feature parity is proven from a shared path or explicit parity test
- [ ] Next-bar entry timing is enforced end to end
- [ ] Delayed intraday quote usage is removed from any executable path
- [ ] Gross/net exposure kill-switches are implemented and verified
- [ ] Daily loss and portfolio drawdown kill-switches are implemented and verified
- [ ] News/risk gating is hard-enforced where intended
- [ ] Multi-week paper-trade validation is completed and documented

