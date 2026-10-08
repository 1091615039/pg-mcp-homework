"""Tests for the API-key-free local demonstration."""

# Chinese punctuation in the display text is intentional.
# ruff: noqa: RUF001

import pytest

from pg_mcp.demo import LocalDemo, _format_rows, main
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


def test_per_user_post_count_uses_grouped_query(demo: LocalDemo) -> None:
    answer = demo.ask("统计每个用户的文章数量")

    assert "GROUP BY u.id, u.name" in answer.sql
    assert answer.rows == [
        {"name": "林晓", "post_count": 2},
        {"name": "陈宇", "post_count": 1},
        {"name": "周宁", "post_count": 1},
    ]


def test_format_empty_rows_is_readable() -> None:
    assert _format_rows([]) == "（没有结果）"


def test_interactive_demo_shows_query_and_security_results(monkeypatch, capsys) -> None:
    questions = iter(["统计每个用户的文章数量", "安全演示", "q"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(questions))

    main()

    output = capsys.readouterr().out
    assert "SQLite" in output
    assert "post_count" in output
    assert "已拦截" in output
    assert "GROUP BY u.id, u.name" in output
