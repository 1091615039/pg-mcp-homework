"""Unit tests for request tracing, logging context, and Prometheus metrics."""

import logging

import pytest

from pg_mcp.observability.logging import RequestContextFilter
from pg_mcp.observability.metrics import MetricsCollector
from pg_mcp.observability.tracing import get_request_id, request_context


@pytest.mark.asyncio
async def test_request_context_is_added_to_log_records() -> None:
    """Logs emitted inside a request scope carry the same trace identifier."""
    record = logging.LogRecord("test", logging.INFO, __file__, 1, "message", (), None)
    request_filter = RequestContextFilter()

    assert get_request_id() is None
    async with request_context("trace-for-test"):
        assert request_filter.filter(record)
        assert record.request_id == "trace-for-test"

    assert get_request_id() is None


def test_metrics_reset_does_not_register_duplicate_collectors() -> None:
    """Metrics can be reset repeatedly without re-registering Prometheus names."""
    collector = MetricsCollector()
    collector.reset_all_metrics()

    collector.increment_query_request("success", "test_db")
    collector.increment_retry_attempt("database")
    assert collector.query_requests.labels(status="success", database="test_db")._value.get() == 1
    assert collector.retry_attempts.labels(operation="database")._value.get() == 1

    collector.reset_all_metrics()
    collector.reset_all_metrics()

    assert collector.query_requests.labels(status="success", database="test_db")._value.get() == 0
    assert collector.retry_attempts.labels(operation="database")._value.get() == 0
