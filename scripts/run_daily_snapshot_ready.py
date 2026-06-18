from __future__ import annotations

import run_daily_snapshot as runner
import update_daily_snapshot as uds
from status_guard import build_status


def _status(target: str | None, status: str, reason: str, latest_source_dates: dict | None = None) -> dict:
    return build_status(uds._read_json, uds.LATEST_FILE, target, status, reason, latest_source_dates)


def main() -> int:
    uds._status = _status
    return runner.main()


if __name__ == "__main__":
    raise SystemExit(main())
