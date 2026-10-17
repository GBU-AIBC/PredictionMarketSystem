from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
import inspect
from typing import Any, Protocol, cast

import asyncpg


@dataclass(frozen=True, slots=True)
class FactorCatalogEntry:
    factor_id: str
    name: str
    description: str
    input_schema_hash: str
    output_type: str
    direction: str
    owner: str


FACTOR_CATALOG_ROWS: tuple[FactorCatalogEntry, ...] = (
    FactorCatalogEntry(
        factor_id="anchoring_lag_divergence",
        name="Anchoring Lag Divergence",
        description=(
            "Signed H2 divergence between LLM posterior and market YES price "
            "after news. Positive means buy YES; negative means buy NO. "
            "Includes 24h linear decay."
        ),
        input_schema_hash="51c833482880179372982cec8f88eb9a7c094ece9e52be91bce487a1c70e9050",
        output_type="scalar",
        direction="neutral",
        owner="system",
    ),
    FactorCatalogEntry(
        factor_id="favorite_longshot_bias",
        name="Favorite-Longshot Bias",
        description=(
            "Signed H1 contrarian bucket signal: negative means buy NO in "
            "overpriced low-YES longshots, positive means buy YES in underpriced "
            "high-YES favorites."
        ),
        input_schema_hash="c4288b992546eb39e3a5e71f660d2a381a45e07f5a267349855f010c59785f6b",
        output_type="scalar",
        direction="neutral",
        owner="system",
    ),
    FactorCatalogEntry(
        factor_id="orderbook_imbalance",
        name="Orderbook Imbalance",
        description="Normalized bid-versus-ask depth imbalance from the current orderbook signal.",
        input_schema_hash="97e885bf8b2edd8ce9fff149334dbe1706358eb4fb8b8c51a4b42561878c5963",
        output_type="scalar",
        direction="neutral",
        owner="system",
    ),
    FactorCatalogEntry(
        factor_id="fair_value_spread",
        name="Fair Value Spread",
        description="Signed difference between external fair value and the current YES price.",
        input_schema_hash="adb923abb80bbd30efa4db61ba846660317f138ef12c3ae521891df2831d64f9",
        output_type="scalar",
        direction="neutral",
        owner="system",
    ),
    FactorCatalogEntry(
        factor_id="subset_pricing_violation",
        name="Subset Pricing Violation",
        description="Signed difference between subset and superset prices from external signals.",
        input_schema_hash="c9e66b836e6fe6a9981ee6419aa38acb39de607e84fb1ff643b46bb9ac446891",
        output_type="scalar",
        direction="neutral",
        owner="system",
    ),
    FactorCatalogEntry(
        factor_id="subset_relation",
        name="Subset Relation",
        description="Raw cross-market subset/superset relation confidence from MarketRelationService.",
        input_schema_hash="27ec38b9caa18da4ad771cdfac2bb3c1727dd44da8d50c7d9a9e2bbd84f552a4",
        output_type="scalar",
        direction="neutral",
        owner="deferred",
    ),
    FactorCatalogEntry(
        factor_id="contradiction_relation",
        name="Contradiction Relation",
        description="Raw cross-market contradiction confidence from MarketRelationService.",
        input_schema_hash="f471eaed6525a85335173285f4fa692193678d027106d8d60458751c839acabc",
        output_type="scalar",
        direction="neutral",
        owner="deferred",
    ),
    FactorCatalogEntry(
        factor_id="semantic_similarity",
        name="Semantic Similarity",
        description="TF-IDF cosine similarity for a related market pair.",
        input_schema_hash="078ca015846d22223ad7cdce31e1cae9a75bc8b44919e0ee2fdb6d8733d7852d",
        output_type="scalar",
        direction="neutral",
        owner="deferred",
    ),
    FactorCatalogEntry(
        factor_id="cross_market_mispricing",
        name="Cross-Market Mispricing",
        description="YES-price sum above one for mutually exclusive market pairs.",
        input_schema_hash="dafee0e995dadb1c8dc140a3d2b7a1186690ae33f4d9c21361b837a44c504385",
        output_type="scalar",
        direction="neutral",
        owner="deferred",
    ),
    FactorCatalogEntry(
        factor_id="metaculus_prior",
        name="Metaculus Prior",
        description="Raw Metaculus probability from the external signal payload.",
        input_schema_hash="4f62fec15fd5abaf2ff76810596268d1e14b46d346ff6e9f38b259c370a3ed71",
        output_type="probability",
        direction="neutral",
        owner="system",
    ),
    FactorCatalogEntry(
        factor_id="yes_count",
        name="Yes Count",
        description="Raw external yes_count observation count from the signal payload.",
        input_schema_hash="afbc921285acc81f1289beca8dd64114c18f49068a8904c651a887c5ba8c178f",
        output_type="scalar",
        direction="neutral",
        owner="system",
    ),
    FactorCatalogEntry(
        factor_id="no_count",
        name="No Count",
        description="Raw external no_count observation count from the signal payload.",
        input_schema_hash="2871d6bf945e3ed4407b8b1f1beeb484cd8bd455a156e939094fbcc6a455c317",
        output_type="scalar",
        direction="neutral",
        owner="system",
    ),
)


class _AsyncAcquireContext(Protocol):
    async def __aenter__(self) -> object: ...

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: object,
    ) -> object: ...


def _select_entries(
    factor_ids: Sequence[str] | None,
) -> tuple[FactorCatalogEntry, ...]:
    # Relation factors are intentionally not advertised until a production
    # writer/lifecycle is wired (F-13).  Keeping their definitions here makes
    # the storage format forwards-compatible while preventing catalog users
    # from depending on values that can never be populated.
    available = {
        entry.factor_id: entry
        for entry in FACTOR_CATALOG_ROWS
        if entry.owner != "deferred"
    }
    selected_ids = tuple(factor_ids) if factor_ids is not None else tuple(available)
    missing_ids = sorted(set(selected_ids) - set(available))
    if missing_ids:
        msg = f"unknown factor catalog ids: {', '.join(missing_ids)}"
        raise ValueError(msg)
    return tuple(available[factor_id] for factor_id in selected_ids)


def validate_registered_factor_catalog() -> None:
    """Fail loudly if a compute-capable factor is absent from the catalog."""

    from pms.factors.definitions import REGISTERED

    registered_ids = {factor.factor_id for factor in REGISTERED}
    catalog_ids = {entry.factor_id for entry in FACTOR_CATALOG_ROWS}
    missing_ids = sorted(registered_ids - catalog_ids)
    if missing_ids:
        msg = "registered factors missing from catalog: " + ", ".join(missing_ids)
        raise ValueError(msg)


async def seed_factor_catalog(
    connection: asyncpg.Connection,
    *,
    factor_ids: Sequence[str] | None = None,
) -> None:
    insert_query = """
    INSERT INTO factors (
        factor_id,
        name,
        description,
        input_schema_hash,
        output_type,
        direction,
        owner
    ) VALUES ($1, $2, $3, $4, $5, $6, $7)
    ON CONFLICT (factor_id) DO UPDATE
    SET
        name = EXCLUDED.name,
        description = EXCLUDED.description,
        input_schema_hash = EXCLUDED.input_schema_hash,
        output_type = EXCLUDED.output_type,
        direction = EXCLUDED.direction,
        owner = EXCLUDED.owner
    """
    for entry in _select_entries(factor_ids):
        await connection.execute(
            insert_query,
            entry.factor_id,
            entry.name,
            entry.description,
            entry.input_schema_hash,
            entry.output_type,
            entry.direction,
            entry.owner,
        )


@asynccontextmanager
async def _acquire_connection(pool: asyncpg.Pool) -> AsyncIterator[object]:
    acquired = pool.acquire()
    if hasattr(acquired, "__aenter__") and hasattr(acquired, "__aexit__"):
        async with cast(_AsyncAcquireContext, acquired) as managed_connection:
            yield managed_connection
        return

    if not inspect.isawaitable(acquired):
        msg = "pool.acquire() must return an awaitable or async context manager"
        raise TypeError(msg)

    raw_connection: object = await acquired
    try:
        yield raw_connection
    finally:
        release = getattr(pool, "release", None)
        if callable(release):
            released = release(raw_connection)
            if inspect.isawaitable(released):
                await cast(Any, released)


async def ensure_factor_catalog(
    pool: asyncpg.Pool,
    *,
    factor_ids: Sequence[str] | None = None,
) -> None:
    async with _acquire_connection(pool) as acquired_connection:
        connection = cast(asyncpg.Connection, acquired_connection)
        async with connection.transaction():
            await seed_factor_catalog(connection, factor_ids=factor_ids)
