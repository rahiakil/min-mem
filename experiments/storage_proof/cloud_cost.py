"""Cloud storage + egress cost projection from measured byte growth.

Model: each employee runs ``agents_per_employee`` agents (default 20), each agent
writes ``memories_per_day`` memories of ``bytes_per_memory`` bytes. Memory is
retained for ``retention_years`` (cumulative store), so the billable store
compounds rather than being just the yearly increment. Cost has two components:

  * **Storage** on the cumulative retained store at S3 Standard-class pricing.
  * **Egress** from one full-store sync per week (52/yr) for backup / DR /
    cross-region replication at AWS-egress-class pricing. Egress dominates
    because a byte saved on the wire is ~4x a byte saved on disk and recurs
    every sync.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


# S3 Standard object storage (documented in paper).
DEFAULT_USD_PER_GB_MONTH = 0.023
# AWS egress to internet / cross-region (documented in paper).
DEFAULT_USD_PER_GB_EGRESS = 0.09
# Full-store syncs per year (weekly backup / DR replication).
FULL_STORE_SYNCS_PER_YEAR = 52
DEFAULT_AGENTS_PER_EMPLOYEE = 20
DEFAULT_RETENTION_YEARS = 3
DEFAULT_MEMORIES_PER_DAY = 50.0


@dataclass
class CloudProjection:
    usd_per_gb_month: float
    usd_per_gb_egress: float
    employees: int
    agents_per_employee: int
    agents: int
    memories_per_day: float
    retention_years: float
    bytes_per_memory: float
    reduction_pct: float
    yearly_growth_bytes: float
    cumulative_store_bytes: float
    cumulative_store_gb: float
    annual_storage_cost_identity: float
    annual_storage_cost_minified: float
    annual_egress_cost_identity: float
    annual_egress_cost_minified: float
    annual_cost_identity: float
    annual_cost_minified: float
    annual_savings_usd: float
    monthly_savings_usd: float
    bytes_30_day: float
    bytes_1_year: float


def project_cloud_cost(
    bytes_per_memory: float,
    memories_per_day: float = DEFAULT_MEMORIES_PER_DAY,
    agents: int | None = None,
    employees: int = 1000,
    agents_per_employee: int = DEFAULT_AGENTS_PER_EMPLOYEE,
    retention_years: float = DEFAULT_RETENTION_YEARS,
    usd_per_gb_month: float = DEFAULT_USD_PER_GB_MONTH,
    usd_per_gb_egress: float = DEFAULT_USD_PER_GB_EGRESS,
    full_store_syncs_per_year: int = FULL_STORE_SYNCS_PER_YEAR,
    identity_bytes: int | None = None,
    minified_bytes: int | None = None,
    reduction_pct: float | None = None,
) -> CloudProjection:
    if agents is None:
        agents = employees * agents_per_employee
    daily = bytes_per_memory * memories_per_day * agents
    b30 = daily * 30
    b365 = daily * 365

    if reduction_pct is not None:
        ratio = 1 - reduction_pct / 100.0
    elif identity_bytes and minified_bytes is not None:
        ratio = minified_bytes / identity_bytes
    else:
        ratio = 0.8

    cumulative = b365 * retention_years
    cum_gb = cumulative / (1024 ** 3)

    storage_id = cum_gb * usd_per_gb_month * 12
    storage_min = storage_id * ratio
    egress_id = full_store_syncs_per_year * cum_gb * usd_per_gb_egress
    egress_min = egress_id * ratio
    cost_id = storage_id + egress_id
    cost_min = storage_min + egress_min
    annual_save = cost_id - cost_min
    monthly_save = annual_save / 12

    return CloudProjection(
        usd_per_gb_month=usd_per_gb_month,
        usd_per_gb_egress=usd_per_gb_egress,
        employees=employees,
        agents_per_employee=agents_per_employee,
        agents=agents,
        memories_per_day=memories_per_day,
        retention_years=retention_years,
        bytes_per_memory=bytes_per_memory,
        reduction_pct=reduction_pct if reduction_pct is not None else (1 - ratio) * 100,
        yearly_growth_bytes=b365,
        cumulative_store_bytes=cumulative,
        cumulative_store_gb=cum_gb,
        annual_storage_cost_identity=storage_id,
        annual_storage_cost_minified=storage_min,
        annual_egress_cost_identity=egress_id,
        annual_egress_cost_minified=egress_min,
        annual_cost_identity=cost_id,
        annual_cost_minified=cost_min,
        annual_savings_usd=annual_save,
        monthly_savings_usd=monthly_save,
        bytes_30_day=b30,
        bytes_1_year=b365,
    )


STANDARD_COMPANY_SIZES = [50, 200, 1000, 5000, 50000]


def project_company_table(
    bytes_per_memory: float,
    reduction_pct: float,
    memories_per_day: float = DEFAULT_MEMORIES_PER_DAY,
    agents_per_employee: int = DEFAULT_AGENTS_PER_EMPLOYEE,
    retention_years: float = DEFAULT_RETENTION_YEARS,
    usd_per_gb_month: float = DEFAULT_USD_PER_GB_MONTH,
    usd_per_gb_egress: float = DEFAULT_USD_PER_GB_EGRESS,
    full_store_syncs_per_year: int = FULL_STORE_SYNCS_PER_YEAR,
    company_sizes: list[int] | None = None,
) -> list[dict]:
    rows: list[dict] = []
    for emp in (company_sizes or STANDARD_COMPANY_SIZES):
        p = project_cloud_cost(
            bytes_per_memory=bytes_per_memory,
            memories_per_day=memories_per_day,
            employees=emp,
            agents_per_employee=agents_per_employee,
            retention_years=retention_years,
            usd_per_gb_month=usd_per_gb_month,
            usd_per_gb_egress=usd_per_gb_egress,
            full_store_syncs_per_year=full_store_syncs_per_year,
            reduction_pct=reduction_pct,
        )
        rows.append({
            "employees": emp,
            "agents": p.agents,
            "cumulative_store_gb": round(p.cumulative_store_gb, 1),
            "annual_cost_identity_usd": round(p.annual_cost_identity, 2),
            "annual_cost_minified_usd": round(p.annual_cost_minified, 2),
            "annual_savings_usd": round(p.annual_savings_usd, 2),
            "monthly_savings_usd": round(p.monthly_savings_usd, 2),
        })
    return rows


def projection_to_dict(p: CloudProjection) -> dict:
    return asdict(p)
