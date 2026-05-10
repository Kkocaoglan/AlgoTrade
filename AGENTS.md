# AGENTS.md

## Role
You are working on a BIST algorithmic trading research and paper-trading repository.

Your job is to help improve code quality, trading realism, risk control, and research workflow without breaking existing behavior unless explicitly required.

## Core rules
- Use `py -3.12` only.
- Never use `py -3` or `python`.
- Never read `*.db`, `*.pkl`, `*.csv`, or `*.png` files directly.
- If database schema is needed, use:
  - `sqlite3 trade_data.db ".schema"`
- Data exploration should be done only with short Python inspection commands.
- After every code change, run the smallest relevant verification command.
- If verification fails, fix once automatically and rerun.
- Prefer minimal safe diffs.
- Do not make broad refactors unless explicitly requested.
- Preserve current behavior unless the task explicitly requires changes.
- Always explain what changed and why.

## Workflow rules
For every task:
1. Inspect relevant files first
2. Make a short plan
3. Apply minimal code changes
4. Run focused verification
5. Summarize findings and results

## Audit rules
When auditing:
- Look for data leakage
- Look for unrealistic execution assumptions
- Look for weak risk controls
- Look for overfitting or invalid evaluation
- Do not edit code unless explicitly asked

## Trading realism rules
Prioritize:
- transaction cost realism
- slippage realism
- gap-aware stop/target handling
- position sizing discipline
- exposure limits
- regime filtering
- confidence threshold quality

## Output style
Be concise, technical, and explicit.
Always mention:
- files inspected
- risks found
- assumptions
- suggested next steps