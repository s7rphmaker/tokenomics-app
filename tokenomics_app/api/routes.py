import asyncio

from fastapi import APIRouter, HTTPException

from tokenomics_app.schemas import MonteCarloInput, ProjectInput
from tokenomics_app.services.deterministic import compute_sell_pressure_table
from tokenomics_app.services.monte_carlo import run_monte_carlo
from tokenomics_app.utils.json_safety import sanitize_floats


router = APIRouter()


@router.post("/sell-pressure")
async def api_sell_pressure(data: ProjectInput):
    total_alloc = sum(c.allocation_pct for c in data.categories)
    if abs(total_alloc - 100.0) > 0.01:
        raise HTTPException(
            status_code=422,
            detail=f"Total allocation must equal 100%. Current total: {total_alloc:.2f}%",
        )

    result = compute_sell_pressure_table(
        data.categories,
        data.total_supply,
        data.initial_price,
        data.horizon_months,
    )
    return sanitize_floats(result)


@router.post("/monte-carlo")
async def api_monte_carlo(data: MonteCarloInput):
    total_alloc = sum(c.allocation_pct for c in data.categories)
    if abs(total_alloc - 100.0) > 0.01:
        raise HTTPException(
            status_code=422,
            detail=f"Total allocation must equal 100%. Current total: {total_alloc:.2f}%",
        )

    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, run_monte_carlo, data)
    return sanitize_floats(result)


@router.get("/health")
async def health():
    return {"status": "ok"}
