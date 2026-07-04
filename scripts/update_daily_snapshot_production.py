from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
for path in (ROOT, SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

os.environ.setdefault("NO_PROXY", "*")
os.environ.setdefault("SW_PROCESSED_CACHE_TTL_SECONDS", "0")


def _patch_streamlit_cache_for_headless() -> None:
    try:
        import streamlit as st  # type: ignore
        try:
            from streamlit.runtime.scriptrunner import get_script_run_ctx  # type: ignore
            in_streamlit = get_script_run_ctx() is not None
        except Exception:
            in_streamlit = False
        if in_streamlit:
            return

        def _no_cache(*args, **kwargs):
            if args and callable(args[0]):
                return args[0]

            def _decorator(func):
                return func

            return _decorator

        st.cache_data = _no_cache  # type: ignore[attr-defined]
    except Exception:
        return


def _print_failure_status() -> None:
    status_file = ROOT / "data" / "update_status.json"
    if not status_file.exists():
        print("[daily-snapshot] update_status.json was not written", file=sys.stderr, flush=True)
        return
    try:
        payload = json.loads(status_file.read_text(encoding="utf-8"))
        print(
            "[daily-snapshot] failure status:\n" + json.dumps(payload, ensure_ascii=False, indent=2),
            file=sys.stderr,
            flush=True,
        )
    except Exception as exc:
        print(f"[daily-snapshot] failed to read update_status.json: {exc}", file=sys.stderr, flush=True)


def main() -> int:
    _patch_streamlit_cache_for_headless()

    from robust_sw_sources import install

    install()

    import update_daily_snapshot

    code = int(update_daily_snapshot.main())
    if code != 0:
        _print_failure_status()
    return code


if __name__ == "__main__":
    raise SystemExit(main())
