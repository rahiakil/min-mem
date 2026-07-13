"""Cloud storage cost projection from measured byte growth."""

from __future__ import annotations

from dataclasses import asdict, dataclass


# Assumption: S3 Standard–class object storage (documented in paper).
DEFAULT_USD_PER_GB_MONTH = 0.023


@dataclass
class CloudProjection:
    usd_per_gb_month: float
    bytes_per_day_per_agent: float
    bytes_30_day: float
    bytes_1_year: float
    gb_month_cost_identity: float
    gb_month_cost_minified: float
    monthly_savings_usd: float
    annual_savings_usd: float


def project_cloud_cost(
    bytes_per_memory: float,
    memories_per_day: float = 50.0,
    agents: int = 10,
    usd_per_gb_month: float = DEFAULT_USD_PER_GB_MONTH,
    identity_bytes: int | None = None,
    minified_bytes: int | None = None,
    reduction_pct: float | None = None,
) -> CloudProjection:
    daily = bytes_per_memory * memories_per_day * agents
    b30 = daily * 30
    b365 = daily * 365

    if reduction_pct is not None:
        gb_id = b365 / (1024**3)
        gb_min = gb_id * (1 - reduction_pct / 100.0)
    elif identity_bytes is not None and minified_bytes is not None:
        ratio = minified_bytes / identity_bytes if identity_bytes else 1.0
        gb_id = b365 / (1024**3)
        gb_min = gb_id * ratio
    else:
        gb_id = b365 / (1024**3)
        gb_min = gb_id * 0.8

    cost_id = gb_id * usd_per_gb_month
    cost_min = gb_min * usd_per_gb_month
    monthly_save = cost_id - cost_min
    annual_save = monthly_save * 12

    return CloudProjection(
        usd_per_gb_month=usd_per_gb_month,
        bytes_per_day_per_agent=bytes_per_memory * memories_per_day,
        bytes_30_day=b30,
        bytes_1_year=b365,
        gb_month_cost_identity=cost_id,
        gb_month_cost_minified=cost_min,
        monthly_savings_usd=monthly_save,
        annual_savings_usd=annual_save,
    )


def projection_to_dict(p: CloudProjection) -> dict:
    return asdict(p)
