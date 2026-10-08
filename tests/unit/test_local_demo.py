"""Tests for the API-key-free local demonstration."""

import pytest

from pg_mcp.demo import LocalDemo
from pg_mcp.models.errors import SecurityViolationError


@pytest.fixture
def demo() -> LocalDemo:
    instance = LocalDemo()
    yield instance
    instance.close()


def test_natural_language_question_generates_and_executes_sql(demo: LocalDemo) -> None:
    answer = demo.ask("统计用户数量")

    assert answer.sql == "SELECT COUNT(*) AS user_count FROM users"
    assert answer.rows == [{"user_count": 3}]


def test_recent_post_count_is_bounded(demo: LocalDemo) -> None:
    answer = demo.ask("查询最近 99 篇文章")

    assert "LIMIT 20" in answer.sql
    assert len(answer.rows) == 4


def test_unsupported_question_is_reported(demo: LocalDemo) -> None:
    with pytest.raises(ValueError, match="离线演示暂不支持"):
        demo.ask("分析下周的营收趋势")


def test_sensitive_column_is_blocked(demo: LocalDemo) -> None:
    with pytest.raises(SecurityViolationError, match="email"):
        demo.validator.validate_or_raise("SELECT email FROM users")


def test_join_query_returns_author_names(demo: LocalDemo) -> None:
    answer = demo.ask("查询已发布文章")

    assert len(answer.rows) == 3
    assert answer.rows[0]["author"] == "陈宇"
