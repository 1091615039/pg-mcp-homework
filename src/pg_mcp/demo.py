"""Offline, template-based demo that does not need OpenAI or PostgreSQL."""

# Chinese punctuation in the interactive prompts is intentional.
# ruff: noqa: RUF001, S608

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from typing import Any

from pg_mcp.config.settings import SecurityConfig
from pg_mcp.services.sql_validator import SQLValidator


@dataclass(frozen=True)
class DemoAnswer:
    """SQL and local sample rows produced for one demo question."""

    sql: str
    rows: list[dict[str, Any]]


class LocalDemo:
    """Small deterministic NL-to-SQL demo using a fixed set of query templates.

    This is intentionally not a general-purpose language model. Every generated
    query is validated before the in-memory SQLite database executes it.
    """

    def __init__(self) -> None:
        self.connection = sqlite3.connect(":memory:")
        self.connection.row_factory = sqlite3.Row
        self.validator = SQLValidator(
            SecurityConfig(
                allowed_tables=["users", "posts"],
                blocked_columns=["email", "users.email"],
            )
        )
        self._seed_database()

    def _seed_database(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE users (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE posts (
                id INTEGER PRIMARY KEY,
                user_id INTEGER NOT NULL REFERENCES users(id),
                title TEXT NOT NULL,
                status TEXT NOT NULL,
                published_at TEXT,
                view_count INTEGER NOT NULL DEFAULT 0
            );
            INSERT INTO users (id, name, email, created_at) VALUES
                (1, '林晓', 'lin@example.test', '2026-09-12'),
                (2, '陈宇', 'chen@example.test', '2026-09-20'),
                (3, '周宁', 'zhou@example.test', '2026-10-01');
            INSERT INTO posts (id, user_id, title, status, published_at, view_count) VALUES
                (1, 1, '数据库安全入门', 'published', '2026-09-15', 128),
                (2, 1, 'SQL 查询实践', 'published', '2026-09-25', 96),
                (3, 2, '本地开发环境搭建', 'published', '2026-10-02', 204),
                (4, 3, '草稿: 数据治理', 'draft', NULL, 0);
            """
        )

    @staticmethod
    def generate_sql(question: str) -> str:
        """Map a supported demo question to a SQL template."""
        normalized = re.sub(r"\s+", "", question).lower()

        if any(word in normalized for word in ("每个用户", "按用户", "peruser")) and any(
            word in normalized for word in ("文章", "帖子", "post")
        ):
            return (
                "SELECT u.name, COUNT(p.id) AS post_count "
                "FROM users AS u LEFT JOIN posts AS p ON p.user_id = u.id "
                "GROUP BY u.id, u.name ORDER BY post_count DESC, u.id"
            )

        if any(word in normalized for word in ("用户", "user")) and any(
            word in normalized for word in ("多少", "数量", "count", "统计")
        ):
            return "SELECT COUNT(*) AS user_count FROM users"

        if any(word in normalized for word in ("已发布", "发布的", "published")) and any(
            word in normalized for word in ("文章", "帖子", "post")
        ):
            return (
                "SELECT p.id, p.title, u.name AS author, p.published_at "
                "FROM posts AS p JOIN users AS u ON u.id = p.user_id "
                "WHERE p.status = 'published' ORDER BY p.published_at DESC"
            )

        if any(word in normalized for word in ("最近", "最新", "recent", "latest")) and any(
            word in normalized for word in ("文章", "帖子", "post")
        ):
            match = re.search(r"(\d+)", normalized)
            limit = min(max(int(match.group(1)), 1), 20) if match else 5
            return (
                "SELECT p.id, p.title, u.name AS author, p.status, p.published_at "
                "FROM posts AS p JOIN users AS u ON u.id = p.user_id "
                f"ORDER BY COALESCE(p.published_at, '') DESC, p.id DESC LIMIT {limit}"
            )

        if any(word in normalized for word in ("文章", "帖子", "post")):
            return (
                "SELECT p.id, p.title, u.name AS author, p.status, p.published_at "
                "FROM posts AS p JOIN users AS u ON u.id = p.user_id ORDER BY p.id"
            )

        if any(word in normalized for word in ("用户", "user")):
            return "SELECT id, name, created_at FROM users ORDER BY id"

        raise ValueError(
            "这个离线演示暂不支持该问题。可试：统计用户数量、查询所有用户、"
            "查询已发布文章、查询最近 3 篇文章、统计每个用户的文章数量。"
        )

    def ask(self, question: str) -> DemoAnswer:
        """Generate, validate, and execute one supported question locally."""
        sql = self.generate_sql(question)
        self.validator.validate_or_raise(sql)
        rows = [dict(row) for row in self.connection.execute(sql).fetchall()]
        return DemoAnswer(sql=sql, rows=rows)

    def check_sensitive_column(self) -> str:
        """Describe the validator's rejection of a deliberately sensitive query."""
        sql = "SELECT email FROM users"
        is_valid, error = self.validator.validate(sql)
        outcome = "通过" if is_valid else "已拦截"
        return f"安全校验结果：{outcome}；SQL：{sql}；原因：{error}"

    def close(self) -> None:
        self.connection.close()


def _format_rows(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "（没有结果）"
    columns = list(rows[0])
    widths = {
        column: max(len(column), *(len(str(row.get(column, ""))) for row in rows))
        for column in columns
    }
    header = " | ".join(column.ljust(widths[column]) for column in columns)
    divider = "-+-".join("-" * widths[column] for column in columns)
    body = [
        " | ".join(str(row.get(column, "")).ljust(widths[column]) for column in columns)
        for row in rows
    ]
    return "\n".join([header, divider, *body])


def main() -> None:
    """Run the interactive local demonstration."""
    demo = LocalDemo()
    print("PG-MCP 本地演示（离线模板模式）")
    print("无需 OpenAI API Key、PostgreSQL 或网络；使用内存 SQLite 样例数据。")
    print("支持范围有限，输入 q 退出，输入“安全演示”查看敏感列拦截。")
    try:
        while True:
            question = input("\n自然语言问题> ").strip()
            if question.lower() in {"q", "quit", "exit", "退出"}:
                break
            if question == "安全演示":
                print(demo.check_sensitive_column())
                continue
            try:
                answer = demo.ask(question)
            except ValueError as error:
                print(error)
                continue
            print(f"\n生成 SQL：\n{answer.sql}")
            print("\n本地查询结果：")
            print(_format_rows(answer.rows))
    finally:
        demo.close()


if __name__ == "__main__":
    main()
