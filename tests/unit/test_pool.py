"""Tests for logical database routing during pool construction."""

from typing import TYPE_CHECKING, cast
from unittest.mock import patch

import pytest

from pg_mcp.config.settings import DatabaseConfig
from pg_mcp.db.pool import create_pools

if TYPE_CHECKING:
    from asyncpg import Pool


@pytest.mark.asyncio
async def test_named_pool_uses_route_key_and_preserves_physical_database_name() -> None:
    pool = cast("Pool", object())
    config = DatabaseConfig(name="warehouse_prod", host="db.example")

    with patch("pg_mcp.db.pool.create_pool", return_value=pool) as create_pool:
        pools = await create_pools({"analytics": config})

    assert pools == {"analytics": pool}
    create_pool.assert_awaited_once_with(config)
    assert config.name == "warehouse_prod"


@pytest.mark.asyncio
async def test_legacy_pool_list_is_keyed_by_physical_database_name() -> None:
    pool = cast("Pool", object())
    config = DatabaseConfig(name="legacy_db")

    with patch("pg_mcp.db.pool.create_pool", return_value=pool):
        pools = await create_pools([config])

    assert pools == {"legacy_db": pool}
