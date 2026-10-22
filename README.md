# Gini prep lab

Hands-on rebuild of the credit model process Gini describes in its insights pieces and IFRS 9 case study:
account-data features → tree binning → WoE → MIV selection → logistic scorecard (elastic net challenger) → validation → 12m PD → lifetime PD → SICR threshold → probability-weighted ECL → backtest and overlay.

## Start
```
pip install -r requirements.txt
jupyter lab gini_lab.ipynb
```
Run from this folder (the notebook reads `data/` and imports `checks.py`).

## Rules
- Fill the `TODO` cells top to bottom; run the checkpoint after each.
- Answer every 🎤 question out loud, under a minute. That's the interview rehearsal.
- `_answers/` holds the solutions notebook, the answer key and the data generator. Stay out until you're done or stuck for 20+ minutes.

## If time is short
Core path ≈ 5–6 h: modules 1, 3, 4, 5, 7, 9a, 9c, 9d.
Then 6 (stability), 9e (overlay), 2 (feature synthesis), 8, 9b.

## The data
Synthetic UK unsecured personal-loan book: 40,000 loans originated 2016–2020, monthly performance to Dec 2022,
unemployment history plus base/upside/downside scenarios. It's built with real-world traps planted in it; the
checkpoints confirm when you've found them. Module 10 maps everything onto your Freddie Mac sample.
