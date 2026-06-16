# Cloud Status

## GitHub Repository

- Repository: `https://github.com/AmineSeRaFimmm/a-share-five-layer-pwa`
- Visibility: private
- Branch: `main`
- Streamlit entry point: `app.py`

## Verified GitHub Actions Runs

- `27598847852`: success, `Update Daily Snapshot`, workflow_dispatch, target date `2026-06-15`
- `27599234209`: success, `Update Daily Snapshot`, workflow_dispatch, target date `2026-06-15`

The second run uses:

- `actions/checkout@v6`
- `actions/setup-python@v6`

## Latest Verified Data

- `data/latest_snapshot.json`
  - `trade_date`: `2026-06-15`
  - `status`: `ready`
  - sectors: `31`
  - market breadth: `80.64516129032258`
  - average coverage: `167.67741935483872`
- `data/update_status.json`
  - `target_trade_date`: `2026-06-15`
  - `status`: `ready`
  - sample industry source dates all equal `2026-06-15`
- `data/backtest/strategy_summary.json`
  - `status`: `ready`
  - recent curve rows: `260`
  - strategy comparison rows: `3`
  - window robustness rows: `4`
  - recent signal rows: `30`

## Streamlit Community Cloud Deployment Inputs

Use these values when creating the Streamlit app:

- Repository: `AmineSeRaFimmm/a-share-five-layer-pwa`
- Branch: `main`
- Main file path: `app.py`
- Python version: `3.12`

## Remaining Acceptance Items

These require Streamlit Cloud and a phone/browser session:

1. Deploy the app in Streamlit Community Cloud.
2. Open the deployed cloud URL on a phone.
3. Add it to the phone home screen as a PWA.
4. Confirm first screen shows data date, update time, status, and buy/sell signal area.
5. Observe 3 trading days of scheduled GitHub Actions updates.
