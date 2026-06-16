# PWA / Cloud Deployment Checklist

This project is designed for:

1. GitHub Actions updates market data and model outputs.
2. Streamlit Community Cloud serves the app.
3. The Streamlit app reads cached files only.
4. The phone installs the cloud page as a PWA.

## 1. Repository Setup

The project directory must be pushed to a GitHub repository with these files included:

- `app.py`
- `pages/`
- `utils.py`
- `avix_utils.py`
- `ui_theme.py`
- `snapshot_store.py`
- `scripts/update_daily_snapshot.py`
- `requirements.txt`
- `.github/workflows/update-daily-snapshot.yml`
- `.streamlit/config.toml`
- `static/manifest.webmanifest`
- `static/pwa.js`
- `static/service-worker.js`
- `static/icon-192.png`
- `static/icon-512.png`
- `data/latest_snapshot.json`
- `data/update_status.json`
- `data/history/`
- `data/avix/`
- `data/backtest/`

## 2. GitHub Actions Verification

Open the GitHub repository and run:

1. Actions
2. `Update Daily Snapshot`
3. `Run workflow`
4. Optional `target_date`: `YYYY-MM-DD`

Expected result:

- Workflow completes successfully.
- It commits only when files under `data/` changed.
- `data/update_status.json` contains `target_trade_date`, `last_attempt_at`, `status`, `reason`, and `latest_source_dates`.
- If sample industries are not fully updated, `status` must be `waiting_partial`, and `data/latest_snapshot.json` must not be overwritten.

## 3. Streamlit Community Cloud

Create a Streamlit app from the GitHub repository:

- Main file path: `app.py`
- Python version: `3.12`

After deployment, open the cloud URL and confirm:

- The app loads without waiting for AKShare.
- Header shows data date and update time.
- If the latest target date is missing, the page shows the previous snapshot plus a clear warning.
- Today recommendation is rendered from `latest_snapshot.json.clarity_signal`.

## 4. Phone PWA Check

On the phone, open the Streamlit Cloud URL.

Safari on iPhone:

1. Share
2. Add to Home Screen
3. Open from the new icon

Chrome on Android:

1. Menu
2. Add to Home screen or Install app
3. Open from the new icon

Expected result:

- App opens in standalone/mobile-friendly mode when supported by the browser.
- Home screen icon is visible.
- First screen shows current data date, update status, and buy/sell signal area.

## 5. Daily Data Acceptance

For each trading day after market close:

- `latest_snapshot.json.trade_date` must equal the target trading day before the page claims updated.
- Industry count must be at least 30.
- Required columns must exist: `板块名称`, `对应ETF`, `涨跌幅`, `综合博弈得分`, `逃顶风险分`, `数据日期`.
- `综合博弈得分` and `逃顶风险分` must not be broadly empty.
- `成分股覆盖数` must not be all zero.
- `latest_source_dates` for sample industries must all equal the target trading day.

Observe at least 3 trading days before treating the automation as stable.
