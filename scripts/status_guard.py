from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo


def build_status(read_json, latest_file, target, state, reason, latest_source_dates=None):
    latest = read_json(latest_file)
    same_ready_snapshot = bool(
        target
        and latest.get('trade_date') == target
        and latest.get('status') == 'ready'
    )
    if state == 'skipped' and same_ready_snapshot:
        state = 'ready'
        reason = 'Snapshot already ready'
    return {
        'target_date': target,
        'target_trade_date': target,
        'last_attempt_at': datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M:%S'),
        'status': state,
        'reason': reason,
        'latest_source_dates': latest_source_dates or {},
    }
