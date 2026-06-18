from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import run_daily_snapshot as runner
import update_daily_snapshot as uds


def _status(target: str | None, status: str, reason: str, latest_source_dates: dict | None = None) -> dict:
    if status == "skipped" and target:
        status = "ready"
        reason = "Snapshot already ready"
    return {
        "target_date": target,
        "target_trade_date": target,
        "last_attempt_at": datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M:%S"),
        "status": status,
        "reason": reason,
        "latest_source_dates": latest_source_dates or {},
    }


def main() -> int:
    uds._status = _status
    return runner.main()


if __name__ == "__main__":
    raise SystemExit(main())
