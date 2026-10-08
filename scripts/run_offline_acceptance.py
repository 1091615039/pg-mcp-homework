"""Run the offline acceptance suite and a deterministic, API-key-free demo."""

from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pg_mcp.config.settings import SecurityConfig  # noqa: E402
from pg_mcp.demo import LocalDemo  # noqa: E402
from pg_mcp.observability.metrics import MetricsCollector  # noqa: E402
from pg_mcp.observability.tracing import get_request_id, request_context  # noqa: E402
from pg_mcp.services.sql_validator import SQLValidator  # noqa: E402


def main() -> None:
    """Run all tests that do not require external PostgreSQL/model services."""
    env = os.environ.copy()
    env.pop("OPENAI_API_KEY", None)
    env.pop("PG_MCP_RUN_EXTERNAL_TESTS", None)
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")

    command = [
        sys.executable,
        "-m",
        "pytest",
        "tests/unit",
        "tests/integration",
        "tests/e2e",
        "--cov=src/pg_mcp",
        "--cov-report=term",
        "--cov-fail-under=80",
        "-o",
        "addopts=-q",
        "--tb=short",
    ]
    tests = subprocess.run(  # noqa: S603 -- fixed local pytest arguments, no shell/user command
        command,
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
    )
    output = tests.stdout + tests.stderr
    summary = re.search(r"(?P<passed>\d+) passed(?:, (?P<skipped>\d+) skipped)?", output)
    coverage = re.search(r"Total coverage: (?P<percent>[\d.]+)%", output)
    if tests.returncode or summary is None or coverage is None:
        sys.stderr.write(output)
        raise SystemExit(tests.returncode or 1)

    demo = LocalDemo()
    try:
        user_count = demo.ask("统计用户数量")
        posts_by_user = demo.ask("统计每个用户的文章数量")
        sensitive = demo.check_sensitive_column()
    finally:
        demo.close()

    validator = SQLValidator(
        SecurityConfig(blocked_tables=["public.credentials"], blocked_columns=["users.email"]),
        allow_explain=True,
    )
    blocked_table = not validator.validate("SELECT id FROM credentials")[0]
    blocked_column = not validator.validate("SELECT email FROM users")[0]
    explain_readonly = validator.validate("EXPLAIN (ANALYZE FALSE) SELECT id FROM users")[0]
    explain_analyze_denied = not validator.validate("EXPLAIN ANALYZE SELECT id FROM users")[0]

    async def trace_check() -> str | None:
        async with request_context("offline-acceptance-demo"):
            return get_request_id()

    request_id = asyncio.run(trace_check())
    metrics = MetricsCollector()
    metrics.increment_query_request("success", "sqlite-demo")
    metric_value = metrics.query_requests.labels(
        status="success", database="sqlite-demo"
    )._value.get()
    metrics.reset_all_metrics()

    report: dict[str, Any] = {
        "status": "PASS",
        "runtime": "offline SQLite + mocked/unit components",
        "real_openai_key_configured": bool(os.getenv("OPENAI_API_KEY")),
        "real_postgres_connected": False,
        "tests": {
            "passed": int(summary.group("passed")),
            "external_skipped": int(summary.group("skipped") or 0),
            "coverage_percent": float(coverage.group("percent")),
        },
        "local_demo": {
            "question": "统计用户数量",
            "sql": user_count.sql,
            "rows": user_count.rows,
            "posts_by_user_sql": posts_by_user.sql,
            "posts_by_user_rows": posts_by_user.rows,
            "sensitive_column_blocked": "已拦截" in sensitive,
            "sensitive_rule": "users.email",
        },
        "security": {
            "schema_qualified_table_blocked": blocked_table,
            "sensitive_column_blocked": blocked_column,
            "explain_analyze_false_allowed": explain_readonly,
            "explain_analyze_denied": explain_analyze_denied,
        },
        "observability": {
            "request_id": request_id,
            "query_metric_sample": int(metric_value),
        },
        "test_output_tail": output.splitlines()[-4:],
    }
    checks = [
        blocked_table,
        blocked_column,
        explain_readonly,
        explain_analyze_denied,
        request_id == "offline-acceptance-demo",
        metric_value == 1,
        report["tests"]["coverage_percent"] >= 80,
    ]
    if not all(checks):
        raise SystemExit("Offline acceptance evidence check failed")
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
