from __future__ import annotations

import os
import time


FIXED_TEST_ASYNC_WORKFLOW_MOCK_SLEEP_SEC = 5.0
ASYNC_WORKFLOW_MOCK_SLEEP_ENV = "TASK_ROUTER_TEST_ASYNC_WORKFLOW_MOCK_SLEEP_SEC"


def _resolve_mock_sleep_sec() -> float:
    raw = str(os.getenv(ASYNC_WORKFLOW_MOCK_SLEEP_ENV, "")).strip()
    if not raw:
        return FIXED_TEST_ASYNC_WORKFLOW_MOCK_SLEEP_SEC
    try:
        parsed = float(raw)
    except ValueError:
        return FIXED_TEST_ASYNC_WORKFLOW_MOCK_SLEEP_SEC
    if parsed < 0:
        return FIXED_TEST_ASYNC_WORKFLOW_MOCK_SLEEP_SEC
    return parsed


def sleep_for_test_async_workflow_mock() -> float:
    # Placeholder delay for mock async workflows to simulate long-running execution.
    sleep_sec = _resolve_mock_sleep_sec()
    if sleep_sec > 0:
        time.sleep(sleep_sec)
    return sleep_sec
