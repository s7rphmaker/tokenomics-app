from typing import List

import numpy as np

from tokenomics_app.schemas import Category


def compute_release_schedule(categories: List[Category], total_supply: float, horizon: int):
    """Compute monthly token releases per category."""
    result = {}
    total_monthly = np.zeros(horizon, dtype=float)

    for cat in categories:
        amount = total_supply * (cat.allocation_pct / 100.0)
        tge_pct = cat.tge_unlock_pct / 100.0
        pressure = cat.sell_pressure_pct / 100.0
        arr = np.zeros(horizon, dtype=float)

        remaining_pct = 1.0 - tge_pct
        monthly_vest_pct = (remaining_pct / cat.vesting_months) if cat.vesting_months > 0 else 0.0

        for m in range(horizon):
            if m == 0:
                arr[m] = tge_pct * amount
            elif cat.vesting_months > 0 and m > cat.cliff_months:
                vesting_month = m - cat.cliff_months
                if vesting_month <= cat.vesting_months:
                    arr[m] = monthly_vest_pct * amount

        total_monthly += arr
        result[cat.name] = {
            "monthly_release": arr.tolist(),
            "amount": amount,
            "pressure": pressure,
        }

    result["__total__"] = total_monthly.tolist()
    return result


def compute_sell_pressure_table(categories: List[Category], total_supply: float, initial_price: float, horizon: int):
    """Compute deterministic sell pressure at constant price."""
    releases = compute_release_schedule(categories, total_supply, horizon)
    total_release = np.array(releases["__total__"], dtype=float)
    cumulative_supply = np.cumsum(total_release)

    pressure_by_cat = {}
    total_pressure_tokens = np.zeros(horizon, dtype=float)

    for cat in categories:
        r = releases[cat.name]
        monthly_arr = np.array(r["monthly_release"], dtype=float)
        pressure_arr = monthly_arr * r["pressure"]
        pressure_by_cat[cat.name] = pressure_arr.tolist()
        total_pressure_tokens += pressure_arr

    total_pressure_usd = total_pressure_tokens * initial_price
    cumulative_pressure_tokens = np.cumsum(total_pressure_tokens)
    cumulative_pressure_usd = np.cumsum(total_pressure_usd)

    rows = []
    for m in range(horizon):
        row = {
            "month": m,
            "label": f"T+{m}M" if m > 0 else "TGE",
            "price": initial_price,
            "monthly_unlock": float(total_release[m]),
            "cumulative_supply": float(cumulative_supply[m]),
            "circulating_pct": float(cumulative_supply[m] / total_supply) * 100,
            "pressure_tokens": float(total_pressure_tokens[m]),
            "pressure_usd": float(total_pressure_usd[m]),
            "cumulative_pressure_tokens": float(cumulative_pressure_tokens[m]),
            "cumulative_pressure_usd": float(cumulative_pressure_usd[m]),
            "pressure_pct_of_unlock": (float(total_pressure_tokens[m]) / float(total_release[m]) * 100)
            if total_release[m] > 0
            else 0,
            "category_breakdown": {
                cat.name: float(pressure_by_cat[cat.name][m]) for cat in categories
            },
        }
        rows.append(row)

    category_releases = {}
    for cat in categories:
        monthly = np.array(releases[cat.name]["monthly_release"], dtype=float)
        category_releases[cat.name] = {
            "monthly": monthly.tolist(),
            "cumulative": np.cumsum(monthly).tolist(),
        }

    avg_monthly = float(np.mean(total_pressure_usd)) if horizon > 0 else 0.0
    peak_month_idx = int(np.argmax(total_pressure_usd)) if horizon > 0 else 0
    peak_month_label = "TGE" if peak_month_idx == 0 else f"T+{peak_month_idx}M"

    # TGE-specific metrics
    tge_pressure_tokens = float(total_pressure_tokens[0]) if horizon > 0 else 0.0
    tge_pressure_usd = float(total_pressure_usd[0]) if horizon > 0 else 0.0
    tge_unlock_tokens = float(total_release[0]) if horizon > 0 else 0.0

    return {
        "rows": rows,
        "category_releases": category_releases,
        "summary": {
            "total_supply": total_supply,
            "initial_price": initial_price,
            "cumulative_sell_pressure_usd": float(cumulative_pressure_usd[-1]) if horizon > 0 else 0.0,
            "peak_monthly_pressure_usd": float(np.max(total_pressure_usd)) if horizon > 0 else 0.0,
            "peak_month_idx": peak_month_idx,
            "peak_month_label": peak_month_label,
            "avg_monthly_pressure_usd": avg_monthly,
            "tge_unlock_tokens": tge_unlock_tokens,
            "tge_pressure_tokens": tge_pressure_tokens,
            "tge_pressure_usd": tge_pressure_usd,
        },
    }
