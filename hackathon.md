# Hackathon Presentation Script

## 1. Scope Note

### Text
This repository is not a single bot. It is a multi-module algorithmic trading research platform with three clearly separated domains:

1. BIST spot paper-trading system
2. Crypto paper-trading system
3. VIOP futures support module

The strongest single written reference is `CLAUDE.md`, but it does not fully capture the entire current repository in one place.

What `CLAUDE.md` covers well:
- BIST spot architecture
- risk stack
- operating loop
- evaluation targets
- core file map

What `CLAUDE.md` does not fully cover by itself:
- the full current crypto architecture
- newer crypto tiered ML design
- crypto weekly retrain and promotion workflow
- crypto-specific logging and reconciliation details
- the latest crypto sentiment, funding, and regime layers

So for a hackathon presentation, the correct message is:

This repo has a documented core operating philosophy in `CLAUDE.md`, and that philosophy is implemented across multiple isolated trading modules, with crypto extending the architecture substantially.

### Figure Prompt
Create a clean executive architecture slide showing one repository branching into three isolated systems:
- BIST Spot Engine
- Crypto Engine
- VIOP Engine
Use a professional fintech style, white background, dark navy text, subtle grid, minimal accents in teal and amber. Show `CLAUDE.md` as the main reference card beside the BIST branch, and annotate crypto as “extended beyond core reference.”


## 2. Executive Summary

### Text
At a high level, this platform is a modular algorithmic trading operating system for research, paper execution, monitoring, and controlled model evolution.

Its design principle is not “just predict price.”

Its design principle is:

1. ingest market data reliably
2. derive features and indicators
3. generate candidate signals
4. filter those signals through multiple risk gates
5. paper-execute trades through an order-management layer
6. journal, reconcile, evaluate, and retrain continuously

This is important for a hackathon audience because the real innovation is not a single model. The innovation is the full decision stack:

- signal generation
- execution realism
- multi-layer risk control
- observability
- post-trade accountability
- retraining governance

### Figure Prompt
Design a horizontal pipeline diagram with the stages:
Data -> Features -> Signals -> Risk Gates -> Execution -> Journaling -> Reconciliation -> Retraining
Each stage should be represented as a modular card with small icons. Style should look like a professional trading platform architecture slide for a technical jury.


## 3. System Philosophy

### Text
The repository is built around five architectural principles:

1. Isolation
Each trading domain is intentionally separated.
The crypto system is isolated from the BIST spot system.
The VIOP system is isolated from both.
This reduces accidental cross-impact and makes experimentation safer.

2. Paper-first realism
The platform is designed to validate strategy logic under realistic constraints before any live deployment.
That includes fees, slippage, position sizing, stop logic, and reconciliation.

3. Risk before prediction
A model signal alone is never enough to open a trade.
Every candidate passes through regime, exposure, sentiment, volatility, cooldown, and consistency filters.

4. Continuous auditability
Everything important is logged, journaled, or reportable:
- signals
- trades
- exits
- health
- reconciliation issues
- retrain decisions

5. Promotion, not blind overwrite
Models are not simply retrained and swapped.
They are staged, evaluated, and only promoted if acceptance criteria pass.

### Figure Prompt
Create a “design principles” slide with five pillars labeled Isolation, Realism, Risk-First, Auditability, Controlled Promotion. Use a premium consulting presentation style, with simple iconography and enough whitespace for stage presentation.


## 4. Top-Level Repository Architecture

### Text
The repository has a layered structure:

Layer 1: Market data access
- BIST data fetchers
- Binance/crypto fetchers
- WebSocket market feeds

Layer 2: Indicator and feature computation
- EMA, RSI, MACD, ATR, Bollinger
- multi-timeframe scoring
- macro/contextual features
- sentiment-aligned features

Layer 3: Signal engines
- rule-based logic
- ML scoring
- direction-specific and tier-specific decisions

Layer 4: Risk and gating
- drawdown protection
- correlation controls
- volatility regime filters
- sentiment blocks
- sizing discipline
- dedup/cooldown controls

Layer 5: Execution and bookkeeping
- OMS
- positions
- orders
- partial exits
- journaling

Layer 6: Monitoring and governance
- status dashboards
- reconciliation reports
- rotating logs
- weekly retrain pipeline

### Figure Prompt
Draw a 6-layer architecture stack from bottom to top:
1. Data Access
2. Indicators & Features
3. Signal Engine
4. Risk & Controls
5. Execution & Journaling
6. Monitoring & Governance
Use a clean fintech enterprise diagram style with soft shadows and labeled boundaries.


## 5. BIST Spot System

### Text
The BIST module is the original core system documented in `CLAUDE.md`.

Its purpose is paper trading in Turkish equities with a structured research-to-execution loop.

Main characteristics:
- universe of 29 BIST stocks
- long and short capability
- ensemble ML model
- rule-based gates around the model
- 60-second production loop
- strict capital and exposure controls
- target-based evaluation window

The BIST system is designed around market hours, unlike crypto.
It has:
- market session awareness
- morning reconciliation
- evening refresh and daily summary
- weekend warning logic

This means BIST behaves like a session-based institutional workflow, not a 24/7 bot.

### Figure Prompt
Create a market-hours workflow diagram for the BIST system:
Startup -> Session Check -> Signal Scan -> Risk Filters -> Trade -> Exit Check -> EOD Refresh -> Daily Summary
Add a visual note that this branch is “session-based / exchange hours dependent.”


## 6. BIST Operating Loop

### Text
The BIST loop works on a repeating operational cadence.

Startup stage:
- load model
- connect broker/data layer
- run reconciliation
- restore health state
- send startup notification

In-session loop:
- recompute drawdown
- check circuit breaker
- refresh signals after skip window
- recompute live features
- evaluate model outputs
- print signal report
- evaluate entries
- evaluate exits periodically

Out-of-session workflow:
- refresh end-of-day data
- prepare daily journal
- compute evaluation summaries

This gives the BIST system a full operational lifecycle rather than a single signal function.

### Figure Prompt
Create a circular loop diagram titled “BIST Operating Lifecycle” with segments:
Startup, Market Session, Signal Evaluation, Risk Control, Execution, Exit Monitoring, End-of-Day, Reporting.
Use a polished product-strategy visual style.


## 7. BIST Model Layer

### Text
The BIST side uses an ensemble ML architecture.

Documented design:
- XGBoost
- LightGBM
- CatBoost
- soft-vote probability averaging

The model is not described as a black box.
It is embedded in a broader feature engineering and validation system.

Key ideas:
- multi-feature market state representation
- walk-forward evaluation
- thresholded tradeability tiers
- performance targets tied to paper-trading evaluation

The system explicitly separates:
- weak signals that can be shown
- tradeable signals that can be executed
- stronger signals that trigger higher-priority alerts

That is a product design choice as much as a modeling choice.

### Figure Prompt
Build a “model in context” diagram showing three model blocks feeding into a probability aggregator, then into confidence tiers labeled Weak, Tradeable, Medium, Strong. Use a modern ML product architecture look.


## 8. BIST Risk Stack

### Text
One of the strongest parts of the platform is the layered risk architecture.

The BIST side includes multiple control layers:

1. kill switch layer
Protects against operational and portfolio-level danger

2. circuit breaker
Responds to drawdown states with graded behavior:
- warn
- halt
- kill

3. sector concentration limits
Prevents clustering too much capital in one theme

4. news and sentiment gating
Avoids opening or holding positions against major negative information flow

5. entry sanity checks
Prevents buying into overstretched conditions

6. cooldown/dedup
Prevents repeated entries too quickly in the same symbol

7. portfolio heat and correlation gating
Controls aggregate risk contribution and co-movement risk

8. volatility regime filtering
Reduces or blocks participation in unstable market states

This is a major hackathon talking point:
the system is not model-centric, it is risk-stack-centric.

### Figure Prompt
Create a “defense-in-depth” slide with eight stacked risk gates around a core signal engine. Visual metaphor should resemble layered shields or concentric security rings, but in a financial trading style.


## 9. BIST Logging, Reconciliation, and Journaling

### Text
The BIST module is observable by design.

It includes:
- structured logging by severity and domain
- trade logs
- risk logs
- system logs
- debug logs

It also includes reconciliation:
- ghost position detection
- missing position detection
- size mismatches
- balance mismatches
- orphan orders
- PnL consistency checks

And it includes end-of-day journaling and evaluation reporting.

The point is simple:
if a trading system cannot explain its own state, it is not production-shaped.

### Figure Prompt
Design a slide titled “Operational Trust Layer” with three columns:
Logging, Reconciliation, Journaling.
Under each column show 3–5 compact bullets as visual tags. Style should feel like enterprise reliability engineering for finance.


## 10. Crypto System Overview

### Text
The crypto module is a separate 24/7 paper-trading engine built with the same architectural philosophy as BIST, but adapted to always-on markets.

Main properties:
- isolated from BIST and VIOP
- Binance-oriented market data
- 24/7 loop
- WebSocket price streaming
- directional ML
- tiered universe
- live status dashboard
- separate crypto logs, crypto journals, and crypto reconciliation

The crypto system is not just a port of the BIST engine.
It adds a more dynamic market-state framework:
- real-time WebSocket-driven exit monitoring
- multi-timeframe scoring
- sentiment and funding context
- BTC dominance context
- tier-specific model management

### Figure Prompt
Create a side-by-side comparison slide:
left = BIST spot system
right = Crypto 24/7 system
Highlight differences in market hours, data source, execution frequency, signal context, and model deployment style.


## 11. Crypto Runtime Architecture

### Text
The crypto runtime is organized around three daemon threads plus a keep-alive main process.

Thread 1: Signal scanner
- every 5 minutes
- computes MTF, ML, sentiment, regime, and entry decisions

Thread 2: Exit monitor
- every 15 seconds
- uses live WebSocket prices first
- falls back to REST if needed
- evaluates stop, target, trailing, and sentiment exits

Thread 3: Health monitor
- every 60 seconds
- prints runtime state
- checks model reload conditions
- supports visibility and uptime diagnostics

This is a strong engineering pattern because it separates:
- lower-frequency entry computation
- higher-frequency risk response
- independent observability

### Figure Prompt
Create a technical architecture diagram for the crypto engine showing:
Main Process
-> Signal Scanner Thread
-> Exit Monitor Thread
-> Health Monitor Thread
-> WebSocket Feed
-> REST Market Data
Use a modern systems diagram style suitable for an engineering jury.


## 12. Crypto Market Universe Design

### Text
The crypto universe is explicitly tiered:

MAJOR tier:
- BTC
- ETH
- BNB
- SOL
- AVAX
- SUI
- DOT
- HYPE

RISKY tier:
- ONDO
- FET

This is a strategic design decision.
The system acknowledges that all coins should not be treated identically.

Tiering affects:
- model assignment
- thresholds
- exposure limits
- risk appetite
- entry discipline

In other words, the platform bakes asset quality into the architecture itself.

### Figure Prompt
Design a clean two-column slide with “MAJOR” and “RISKY” tiers. Show each coin as a card. Add annotations for stricter limits and higher thresholds on the risky side.


## 13. Crypto Data and Context Layer

### Text
The crypto data layer combines several information channels:

1. REST market data
- OHLCV candle retrieval
- historical data for feature computation

2. WebSocket live prices
- near-real-time price updates
- used especially for exit responsiveness

3. Sentiment layer
- Fear & Greed Index
- cached to avoid noisy repeated requests

4. Funding-rate layer
- Binance perpetual funding context
- captures crowd positioning pressure

5. BTC dominance layer
- market-structure context
- helps interpret altcoin participation regime

This means the crypto engine is not price-only.
It is context-aware.

### Figure Prompt
Create a central “Crypto Context Engine” visual with five inbound inputs:
OHLCV, WebSocket Prices, Fear & Greed, Funding Rate, BTC Dominance.
All feed into a central decision layer. Use polished fintech infographic styling.


## 14. Crypto Indicator and MTF Logic

### Text
The crypto indicator layer computes classic technical features in memory:
- EMA
- RSI
- MACD
- Bollinger Bands
- ATR

On top of this, the system builds multi-timeframe scoring using:
- 5m
- 15m
- 1h

This is important because the system separates:
- structural direction
- medium-term confirmation
- entry timing

The platform also includes freshness checks on bars so stale candles after reconnects do not silently drive decisions.

That is a subtle but important production-quality behavior.

### Figure Prompt
Create a multi-timeframe visual with three stacked timeframes:
1h = trend
15m = confirmation
5m = timing
Show arrows merging into a single long-score / short-score output card.


## 15. Crypto Signal Engine

### Text
The crypto entry engine is a layered decision system.

Inputs include:
- MTF score
- model probabilities
- BTC regime
- Fear & Greed
- funding state
- BTC dominance regime
- higher-timeframe bias
- correlation gate
- daily risk mode

A trade is opened only if enough of these layers align.

This means the crypto engine behaves more like a rules-governed policy system than a simple classifier wrapper.

The model proposes.
The gate stack decides.

### Figure Prompt
Design a decision-tree style slide with a center node labeled “Candidate Signal” and surrounding decision boxes:
MTF, ML, BTC Regime, F&G, Funding, Correlation, HTF Bias, Daily Risk, Exposure.
Final outputs should be Execute / Reject / Shadow Only.


## 16. Crypto Regime and Context Controls

### Text
The crypto module contains multiple market-context layers beyond raw indicators.

BTC two-layer regime:
- structural trend using long-horizon EMA logic
- short-term momentum using 1h return behavior

Dynamic correlation gate:
- average BTC-alt correlation across the market
- threshold adapts by market state

Funding sentiment:
- blocks crowded-side entries
- boosts size in the contrarian direction

Fear & Greed:
- influences entry strictness
- also contributes to exit decisions in extreme conditions

BTC dominance:
- helps filter weak altcoin environments

These mechanisms show that the system models market state, not just chart shape.

### Figure Prompt
Create a “market context dashboard” mockup with widgets for:
BTC Regime, Correlation State, Funding Bias, Fear & Greed, BTC Dominance.
Use a serious trading terminal aesthetic, but simplified for presentation.


## 17. Crypto Execution and OMS

### Text
The crypto OMS is a paper order-management layer backed by SQLite tables.

It manages:
- positions
- orders
- open/closed lifecycle
- partial exits
- direction handling
- stop and target persistence

This layer exists so the strategy can be evaluated as an operating system, not just a signal generator.

Key behaviors:
- separate order log and position log
- open and close transitions are explicit
- partial profit-taking is represented
- exits are journaled and reportable

### Figure Prompt
Design a lifecycle diagram:
Signal Approved -> Order Created -> Position Open -> Partial Exit -> Final Exit -> Journaled -> Reconciled
Make it look like an operations flow used by a trading desk.


## 18. Crypto MajorAfter Concept

### Text
The crypto engine includes a daily “MajorAfter” mechanism.

Its purpose is:
once a daily profit objective is hit, stop risking real paper capital and continue tracking hypothetical follow-up trades in a shadow book.

Conceptually, that means:
- lock the achieved daily gain
- freeze real trading
- continue observing what would have happened

This is a research-and-discipline hybrid mechanism.

It does two things:
- protects the achieved day
- still preserves learning opportunities after the lock point

For a hackathon audience, this is a strong “trading operations” feature because it reflects behavior design, not just math.

### Figure Prompt
Create a visual showing:
Morning Equity -> Daily Target Hit -> Real Trading Locked -> Shadow Book Continues
Use a split-lane diagram where the real capital lane stops and the shadow lane continues for research.


## 19. Crypto Journaling and Status Surfaces

### Text
The crypto system is highly observable.

It has:
- a live status dashboard
- signal journals
- trade journals
- daily summaries
- rotating logs by domain

Status coverage includes:
- live prices
- open positions
- MTF state
- freshness state
- BTC regime
- funding bias
- correlation state
- model availability

This is critical in a demo setting because it allows judges to see:
- what the system believes
- why it is blocked
- what risk mode it is in

### Figure Prompt
Build a dashboard-style slide showing compact widgets:
Open Positions, BTC Regime, Funding, Correlation, Model Status, Signal Freshness.
Style should resemble a clean internal trading operations dashboard.


## 20. Crypto Reconciliation

### Text
The crypto module has its own reconciliation layer, separate from the BIST one.

It checks for:
- ghost positions
- missing fills
- size mismatches
- negative-balance inconsistencies
- orphan orders
- PnL mismatches

The design principle is “report, never auto-fix.”

That is an important trust decision.
In a trading system, hidden automatic repair can be more dangerous than a visible inconsistency report.

### Figure Prompt
Create a slide titled “Reconciliation as Internal Audit” showing six issue classes flowing into an alert/report layer. Use a professional compliance-and-controls visual theme.


## 21. Crypto ML Training and Promotion

### Text
The crypto ML workflow has evolved beyond a single generic model.

It now supports:
- directional training
- tier-specific model separation
- staged candidate artifacts
- weekly retrain
- acceptance gates
- controlled promotion
- rejected-model archiving

This means the system treats model deployment as governed operations, not just experimentation.

There are now effectively four strategic model tracks:
- LONG MAJOR
- LONG RISKY
- SHORT MAJOR
- SHORT RISKY

That reflects a core belief:
different market-quality buckets should not necessarily share the same learned decision surface.

### Figure Prompt
Create a 2x2 model governance matrix:
columns = MAJOR / RISKY
rows = LONG / SHORT
Each cell shows Train -> Validate -> Stage -> Promote/Reject.
Use a professional ML operations slide style.


## 22. Crypto Validation and Governance

### Text
The crypto training pipeline contains explicit validation governance.

Examples of governance ideas present in the architecture:
- walk-forward validation
- precision-driven acceptance
- per-coin support awareness
- rejected-model metadata logging
- retrain history tracking

This matters because the repo is not just asking:
Can a model fit?

It is asking:
- Is the model stable enough?
- Is the model sufficiently broad?
- Is it overfitting a few weak coins?
- Should it be promoted at all?

That is a mature research question.

### Figure Prompt
Design a slide titled “Model Governance” with cards:
Walk-Forward, Per-Coin Validation, Acceptance Gates, Rejection Logging, Promotion History.
Use a clean MLOps visual identity.


## 23. VIOP Module

### Text
The VIOP branch is presented as a completely independent system for futures-oriented experimentation.

It has its own:
- data fetch flow
- indicator computation
- trader loop
- OMS
- journal

In the repo architecture, VIOP is strategically important even if it is less documented than BIST:
it proves the platform is intended as a multi-market framework, not a one-strategy script.

### Figure Prompt
Create a lightweight supporting slide showing VIOP as an independent extension branch from the same repository base. Style should be visually consistent with the main architecture slides.


## 24. Persistent Storage Model

### Text
The repository uses SQLite as the operational memory and bookkeeping layer.

Conceptually, storage is used for:
- historical market data
- indicators
- positions
- orders
- equity tracking
- signal journals
- daily journals
- reconciliation context

You do not need to explain every table in a hackathon.
What you should explain is why this matters:

SQLite makes the platform:
- inspectable
- reproducible
- portable
- easy to audit

This is a strong prototype-to-product choice for a local research engine.

### Figure Prompt
Create a storage architecture visual with a central SQLite cylinder and labeled satellites:
Market Data, Indicators, Positions, Orders, Journals, Reconciliation, Metrics.
Use a polished system-design diagram style.


## 25. Logs and Telemetry

### Text
The platform has two logging worlds:

1. Shared/general logging
- BIST-oriented system logs
- trade logs
- risk logs
- debug logs

2. Crypto-specific logging
- crypto trade log
- crypto scan log
- crypto system log
- crypto exit log

Telegram is used as a real-time alert channel.

This means the system supports both:
- forensic analysis after the fact
- live operational awareness during runtime

### Figure Prompt
Create a telemetry slide showing two channels:
File Logs and Telegram Alerts.
Under file logs, split into Spot Logs and Crypto Logs. Use a modern reliability engineering presentation style.


## 26. Why This Project Is More Than a Bot

### Text
In a hackathon, the strongest framing is this:

This project is not a prediction script.
It is a modular trading operating system.

Why?

Because it includes:
- data ingestion
- feature engineering
- decision policy
- risk enforcement
- execution simulation
- journaling
- reconciliation
- reporting
- retraining governance

That is the difference between:
- a model demo
and
- an operational research platform

### Figure Prompt
Create a bold closing slide with the statement:
“Not a Bot. A Trading Operating System.”
Below it, show nine small capability blocks: Data, Features, Signals, Risk, Execution, Logs, Journal, Audit, Retrain.


## 27. Suggested Demo Narrative

### Text
Here is the most professional way to narrate the system live:

Part 1 — Problem
Retail and research trading systems often fail because they focus on prediction but ignore execution discipline, risk controls, and auditability.

Part 2 — Our approach
We built a modular algorithmic trading platform with separate engines for BIST, crypto, and VIOP, each isolated but following the same operational philosophy.

Part 3 — Core differentiator
Our real differentiator is not just ML.
It is the combination of:
- signal generation
- multi-layer risk gating
- paper execution realism
- reconciliation
- model promotion governance

Part 4 — Crypto innovation
On crypto we extended the architecture with:
- real-time WebSocket exits
- tier-specific models
- BTC regime logic
- funding and sentiment overlays
- shadow-book learning after daily target lock

Part 5 — Why it matters
This creates a system that is explainable, auditable, and much closer to a production-shaped trading stack than a notebook-based strategy prototype.

### Figure Prompt
Create a final storyteller slide with five numbered blocks:
Problem, Approach, Differentiator, Crypto Innovation, Impact.
Use a premium keynote style with minimal icons and strong hierarchy.


## 28. Presenter Warnings

### Text
Do not present this as live-money autonomous trading.
Present it as:
- paper-trading research platform
- controlled strategy lab
- production-shaped architecture with governance

Do not oversell accuracy.
Instead, emphasize:
- modularity
- isolation
- observability
- discipline
- model governance

If a judge asks “What is the innovation?”
answer:

The innovation is the full-stack architecture that connects prediction, risk, execution, and auditability into one coherent operating system.

If a judge asks “What is the hardest engineering problem solved here?”
answer:

Keeping a signal-driven trading system explainable and governable while still letting it operate across multiple markets and multiple model styles.

### Figure Prompt
Create a clean appendix-style slide titled “How to Position the Project” with two columns:
Say This / Avoid This.
Use a professional investor-demo tone.
