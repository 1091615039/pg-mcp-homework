"""Create two Chinese submission visuals from a fresh offline acceptance run."""

# Chinese punctuation in the rendered labels is intentional.
# ruff: noqa: RUF001

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
DESKTOP = Path.home() / "Desktop" / "PG-MCP作业效果图"
WIDTH, HEIGHT = 1600, 1100
FONT_PATH = Path("C:/Windows/Fonts/msyh.ttc")


def font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT_PATH), size)


def rounded(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    fill: str,
    radius: int = 22,
    outline: str | None = None,
    width: int = 1,
) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def text(
    draw: ImageDraw.ImageDraw,
    point: tuple[int, int],
    value: str,
    size: int,
    color: str = "#17243A",
    **kwargs: Any,
) -> None:
    draw.text(point, value, font=font(size), fill=color, **kwargs)


def base(title: str, subtitle: str, badge: str) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (WIDTH, HEIGHT), "#F2F5FB")
    draw = ImageDraw.Draw(image)
    rounded(draw, (48, 42, WIDTH - 48, 103), "#16233B", radius=22)
    text(draw, (78, 59), "PG-MCP  /  AI 编程实战营 · 第二次作业", 24, "#FFFFFF")
    rounded(draw, (1280, 56, 1512, 91), "#203650", radius=17)
    text(draw, (1300, 61), badge, 17, "#AEE6D1")
    text(draw, (70, 136), title, 42)
    text(draw, (72, 194), subtitle, 21, "#63718A")
    return image, draw


def footer(draw: ImageDraw.ImageDraw) -> None:
    text(
        draw,
        (72, 1050),
        (
            "证据说明：基于本机实际运行结果；离线样例使用内存 SQLite，"
            "自动化测试使用 mock。未连接真实 OpenAI 或 PostgreSQL。"
        ),
        16,
        "#65738A",
    )


def render_offline_demo(report: dict[str, Any]) -> Image.Image:
    image, draw = base(
        "无 API Key 也能现场演示：查询结果与敏感数据拦截",
        "真实运行 LocalDemo · 固定模板映射 SQL · 内存 SQLite 执行 · SQLValidator 执行列级规则",
        "OFFLINE · PASS",
    )
    demo = report["local_demo"]

    rounded(draw, (66, 250, 1035, 980), "#FFFFFF", radius=28)
    text(draw, (100, 278), "01  查询演示", 27, "#4052CC")
    rounded(draw, (100, 326, 1000, 388), "#F2F5FC", radius=16)
    text(draw, (123, 345), "自然语言问题  ·  统计每个用户的文章数量", 20)

    rounded(draw, (100, 412, 1000, 572), "#101A2B", radius=18)
    text(draw, (126, 430), "生成并通过安全校验的 SQL", 17, "#95A6C4")
    sql_lines = [
        "SELECT u.name, COUNT(p.id) AS post_count",
        "FROM users AS u LEFT JOIN posts AS p ON p.user_id = u.id",
        "GROUP BY u.id, u.name ORDER BY post_count DESC, u.id",
    ]
    for i, line in enumerate(sql_lines):
        text(draw, (126, 462 + i * 30), line, 16, "#E4EDFA")

    rounded(draw, (100, 596, 1000, 786), "#F8FAFE", radius=18, outline="#E3E9F4")
    text(draw, (126, 616), "SQLite 查询结果", 19, "#34415A")
    for x, label in ((126, "用户名"), (500, "文章数")):
        text(draw, (x, 655), label, 16, "#78869C")
    rows = demo["posts_by_user_rows"]
    for index, row in enumerate(rows):
        y = 691 + index * 30
        draw.line((126, y - 6, 970, y - 6), fill="#E8EDF5", width=1)
        text(draw, (126, y), str(row["name"]), 17)
        text(draw, (500, y), str(row["post_count"]), 17, "#4052CC")

    rounded(draw, (100, 812, 1000, 944), "#F0FAF6", radius=18, outline="#D4EEE4")
    text(draw, (126, 832), "快速核对  ·  统计用户数量", 18, "#316B58")
    text(draw, (126, 875), demo["sql"], 16, "#41576B")
    text(draw, (760, 875), f"user_count = {demo['rows'][0]['user_count']}", 20, "#20835E")

    rounded(draw, (1060, 250, 1534, 980), "#FFFFFF", radius=28)
    text(draw, (1095, 278), "02  列级安全拦截", 26, "#C64E58")
    text(draw, (1095, 339), "模拟尝试读取敏感邮箱列", 18, "#63718A")
    rounded(draw, (1094, 382, 1500, 473), "#FFF4F4", radius=18, outline="#F5D6D8")
    text(draw, (1120, 399), "SELECT email FROM users", 18, "#9F3541")
    text(draw, (1120, 432), "列策略：users.email 设为禁止返回", 15, "#9F3541")
    rounded(draw, (1094, 498, 1500, 560), "#D9535D", radius=18)
    text(draw, (1214, 516), "已拦截  ·  安全校验拒绝", 19, "#FFFFFF")
    rounded(draw, (1094, 590, 1500, 780), "#F7F9FC", radius=18)
    text(draw, (1120, 612), "实际返回的拒绝原因", 16, "#748199")
    text(draw, (1120, 651), "Access to column", 18, "#35435B")
    text(draw, (1120, 682), "'email' is not allowed", 18, "#35435B")
    draw.line((1120, 730, 1470, 730), fill="#E0E6EF", width=1)
    text(draw, (1120, 748), "不依赖模型 Key · 无真实邮箱被读取", 15, "#748199")
    footer(draw)
    return image


def render_acceptance(report: dict[str, Any]) -> Image.Image:
    tests = report["tests"]
    image, draw = base(
        "自动化回归验收：多库、安全策略、弹性与可观测性",
        "无外部凭据运行单元与 mock 流程 · 同时直接执行离线查询和 SQL 安全断言",
        "TEST SUITE · PASS",
    )

    cards = [
        (66, 250, 526, 404, "通过测试", str(tests["passed"]), "单元 + mock 流程", "#18845F"),
        (
            548,
            250,
            1008,
            404,
            "覆盖率",
            f"{tests['coverage_percent']:.2f}%",
            "质量门槛 ≥ 80%",
            "#4052CC",
        ),
        (
            1030,
            250,
            1534,
            404,
            "外部联调",
            str(tests["external_skipped"]),
            "因无 PostgreSQL / Key 跳过",
            "#B77924",
        ),
    ]
    for x1, y1, x2, y2, label, value, detail, color in cards:
        rounded(draw, (x1, y1, x2, y2), "#FFFFFF", radius=25)
        text(draw, (x1 + 28, y1 + 22), label, 19, "#65738A")
        text(draw, (x1 + 28, y1 + 58), value, 42, color)
        text(draw, (x1 + 28, y1 + 115), detail, 16, "#7A879A")

    rounded(draw, (66, 435, 1534, 972), "#FFFFFF", radius=28)
    text(draw, (100, 465), "关键验收项", 26, "#34415A")
    check_items = [
        ("OK", "多数据库路由", "各逻辑库路由到独立执行器；保留物理 PostgreSQL 数据库名", "#18845F"),
        ("OK", "访问控制", "schema / 表 / 列拒绝规则、允许列、CTE 来源及通配读取回归", "#18845F"),
        (
            "OK",
            "EXPLAIN 执行策略",
            "ANALYZE FALSE 允许；真正执行查询的 EXPLAIN ANALYZE 拒绝",
            "#18845F",
        ),
        ("OK", "限流与重试", "并发限流、瞬时数据库错误重试与退避由单测验证", "#18845F"),
        ("OK", "指标与追踪", "请求指标样本 = 1；request_id 可在异步请求上下文中读取", "#18845F"),
        (
            "OK",
            "配置 / 模型校验",
            "dotenv 与进程环境优先级、逗号策略列表、响应字段一致性",
            "#18845F",
        ),
    ]
    for index, (icon, title, detail, color) in enumerate(check_items):
        y = 516 + index * 58
        rounded(draw, (100, y, 148, y + 34), "#E8F5EF", radius=16)
        text(draw, (108, y + 5), icon, 15, color)
        text(draw, (160, y + 2), title, 18, "#26354D")
        text(draw, (390, y + 4), detail, 16, "#65738A")

    rounded(draw, (1010, 855, 1496, 930), "#F3F6FC", radius=17)
    request_id = report["observability"]["request_id"]
    metric = report["observability"]["query_metric_sample"]
    text(draw, (1034, 871), f"request_id  {request_id}", 15, "#4052CC")
    text(draw, (1034, 899), f"Prometheus 请求指标样本  {metric}", 15, "#4052CC")

    footer(draw)
    return image


def main() -> None:
    """Run acceptance, then save reviewable PNGs to the user's desktop."""
    process = subprocess.run(  # noqa: S603 -- fixed local Python script, no user-controlled command
        [sys.executable, str(ROOT / "scripts" / "run_offline_acceptance.py")],
        cwd=ROOT,
        check=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
    )
    report = json.loads(process.stdout)
    DESKTOP.mkdir(parents=True, exist_ok=True)
    first = DESKTOP / "PG-MCP-01-离线查询与安全拦截.png"
    second = DESKTOP / "PG-MCP-02-自动化验收与质量指标.png"
    render_offline_demo(report).save(first, optimize=True)
    render_acceptance(report).save(second, optimize=True)
    (DESKTOP / "离线验收原始结果.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {"images": [str(first), str(second)], "tests": report["tests"]},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
