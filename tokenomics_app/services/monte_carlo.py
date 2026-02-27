import math
from typing import Dict, List

import numpy as np

from tokenomics_app.schemas import Category, MonteCarloInput
from tokenomics_app.services.deterministic import compute_release_schedule


GRANULARITY_STEPS = {
    "monthly": 1,
    "weekly": 4,
    "daily": 30,
}
QUANTILES = [0.05, 0.25, 0.50, 0.75, 0.95]

# ── Market Layer constants ──
# Impact model: r_perm = alpha * sign(u) * |u|^beta, u = F_t / L_t
# Concavity: beta < 1 → large trades have diminishing marginal impact
_IMP_BETA_DEFAULT = 0.6          # default concavity exponent (overridden by params.impact_beta)

# Hard cap on single-step log-return from impact (prevents numerical blow-up)
# exp(0.405) ≈ 1.5 → up 50%; exp(-0.693) ≈ 0.5 → down 50%
_IMPACT_LOG_CAP = 0.405

# Liquidity regime factors: L_target = L0 * factor
_L_REGIME_FACTORS = {"normal": 1.0, "crash": 0.40, "euphoria": 1.30}

# Probability of regime transitions per step (tuned for weekly granularity)
_L_CRASH_PROB = 0.008       # ~once per ~2.5 years at weekly steps
_L_RECOVERY_PROB = 0.15     # crash mean duration ~7 steps (~1.75 months)
_L_EUPHORIA_PROB = 0.004    # ~once per ~5 years
_L_EUPHORIA_DECAY = 0.10    # euphoria mean duration ~10 steps (~2.5 months)

# GBM noise: minimum 1% daily vol floor
_GBM_VOL_FLOOR = 0.01

# ADV → daily vol calibration (Amihud illiquidity proxy)
# sigma_daily ≈ _ADV_VOL_SCALE / sqrt(ADV_daily / initial_mcap)
# Anchor: ADV/mcap=1.5% → sigma≈5.7%/day
_ADV_VOL_SCALE = 0.007

# Default execution cap fraction for buyback
_BUYBACK_EXEC_FRAC_DEFAULT = 0.20

# Absolute minimum floor for L_t (prevents division by zero)
_L_MIN_USD = 1_000.0

# Momentum window (unchanged)
_MOMENTUM_WINDOW_STEPS = 4   # окно для расчёта momentum (4 недели ≈ 1 месяц)


# Профили запуска — определяют глубину ликвидности и чувствительность к давлению
PRELAUNCH_PROFILES: Dict[str, Dict[str, float]] = {
    # depth_frac:          circ mcap fraction available as AMM/CEX liquidity (±2% range)
    # adv_frac:            daily volume / circ mcap
    # impact_free_sell_ratio: absorption threshold (fraction of circulating) for KPI counting
    # alpha_mod:           multiplier on impact_alpha_perm schema field
    # sigma_mod:           multiplier on ADV-calibrated sigma_daily
    "conservative": {
        "depth_frac": 0.008,
        "adv_frac": 0.008,
        "impact_free_sell_ratio": 0.0018,
        "alpha_mod": 1.30,   # more sensitive: shallow market, strong impact
        "sigma_mod": 1.20,   # higher vol: less liquid, more choppy
    },
    "balanced": {
        "depth_frac": 0.015,
        "adv_frac": 0.015,
        "impact_free_sell_ratio": 0.0012,
        "alpha_mod": 1.00,
        "sigma_mod": 1.00,
    },
    "aggressive": {
        "depth_frac": 0.025,
        "adv_frac": 0.025,
        "impact_free_sell_ratio": 0.0009,
        "alpha_mod": 0.75,   # less sensitive: deeper market absorbs more
        "sigma_mod": 0.85,   # lower vol: more liquid
    },
}

EXCHANGE_CONFIG: Dict[str, Dict[str, float]] = {
    # Tier-4 is count-based and can be selected multiple times.
    "tier4": {"min": 0.0, "max": 3_000.0, "usd": 1_500.0, "l0": 500.0, "units": 0.08},
    # Individual exchanges
    "mexc": {"min": 20_000.0, "max": 30_000.0, "usd": 25_000.0, "l0": 2_200.0, "units": 0.42},
    "gate": {"min": 20_000.0, "max": 30_000.0, "usd": 25_000.0, "l0": 2_200.0, "units": 0.42},
    "kucoin": {"min": 20_000.0, "max": 30_000.0, "usd": 25_000.0, "l0": 2_200.0, "units": 0.42},
    "bingx": {"min": 5_000.0, "max": 15_000.0, "usd": 10_000.0, "l0": 1_200.0, "units": 0.22},
    "htx": {"min": 5_000.0, "max": 15_000.0, "usd": 10_000.0, "l0": 1_200.0, "units": 0.22},
    "binance_alpha": {"min": 400_000.0, "max": 400_000.0, "usd": 400_000.0, "l0": 8_000.0, "units": 1.20},
    "okx": {"min": 600_000.0, "max": 600_000.0, "usd": 600_000.0, "l0": 12_000.0, "units": 1.60},
    "bybit": {"min": 600_000.0, "max": 600_000.0, "usd": 600_000.0, "l0": 12_000.0, "units": 1.60},
    "binance": {"min": 3_000_000.0, "max": 3_000_000.0, "usd": 3_000_000.0, "l0": 30_000.0, "units": 3.00},
}

# ── Hype v2 calibration tables ──
# Monthly ongoing buy pressure as fraction of initial circulating mcap.
# Exponential scaling: hype creates FOMO feedback loops.
HYPE_BASE_FRACTIONS: Dict[int, float] = {
    1: 0.0005,   # dead project — 0.05%/month
    2: 0.0015,   # very low — 0.15%/month
    3: 0.0035,   # low — 0.35%/month
    4: 0.0070,   # below average — 0.70%/month
    5: 0.0120,   # medium — 1.20%/month
    6: 0.0200,   # above average — 2.00%/month
    7: 0.0350,   # high — 3.50%/month
    8: 0.0550,   # very high — 5.50%/month
    9: 0.0850,   # extreme — 8.50%/month
    10: 0.1300,  # viral FOMO — 13.00%/month
}

# Multiplicative modifiers from hype checkboxes.
# With all 4 active: 1.25 × 1.20 × 1.15 × 1.15 ≈ 1.98×
HYPE_CHECKBOX_MULTIPLIERS: Dict[str, float] = {
    "kol": 1.25,        # KOL/influencer support
    "narrative": 1.20,  # strong narrative / meta alignment
    "team": 1.15,       # known team / previous success
    "vc": 1.15,         # VC / institutional backing
}

# Backward-compatible grouped tiers for older payloads.
EXCHANGE_TIER_CONFIG: Dict[str, Dict[str, float]] = {
    "tier4": EXCHANGE_CONFIG["tier4"],
    "bingx_htx": {"min": 5_000.0, "max": 15_000.0, "usd": 10_000.0, "l0": 1_200.0, "units": 0.22},
    "mexc_gate_kucoin": {"min": 20_000.0, "max": 30_000.0, "usd": 25_000.0, "l0": 2_200.0, "units": 0.42},
    "binance_alpha": EXCHANGE_CONFIG["binance_alpha"],
    "okx_bybit": {"min": 600_000.0, "max": 600_000.0, "usd": 600_000.0, "l0": 12_000.0, "units": 1.60},
    "binance": EXCHANGE_CONFIG["binance"],
}


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _marketing_coef_from_budget(budget_usd: float) -> float:
    b = max(float(budget_usd or 0.0), 0.0)
    if b <= 10_000.0:
        return 0.2 * (b / 10_000.0)
    if b <= 300_000.0:
        return 0.2 + ((b - 10_000.0) / 290_000.0) * 0.45
    if b <= 1_000_000.0:
        return 0.65 + ((b - 300_000.0) / 700_000.0) * 0.05
    return 0.70


def _legacy_score_from_tge_usd(tge_usd: float) -> float:
    x = max(float(tge_usd or 0.0), 0.0)
    max_ref = 3_000_000.0
    return _clamp(10.0 * (math.log10(1.0 + x / 3_000.0) / math.log10(1.0 + max_ref / 3_000.0)), 0.0, 10.0)


def _resolve_exchange_aggregate(params: MonteCarloInput) -> Dict[str, object]:
    selected_raw = getattr(params, "selected_exchanges", None) or []
    selected = []
    seen = set()
    for raw in selected_raw:
        key = str(raw).lower().strip()
        if key in EXCHANGE_CONFIG and key != "tier4" and key not in seen:
            seen.add(key)
            selected.append(key)

    tier4_count = max(int(getattr(params, "tier4_exchange_count", 0) or 0), 0)

    if selected or tier4_count > 0:
        min_usd = 0.0
        max_usd = 0.0
        usd = 0.0
        l0 = 0.0
        units = 0.0
        for key in selected:
            cfg = EXCHANGE_CONFIG[key]
            min_usd += cfg["min"]
            max_usd += cfg["max"]
            usd += cfg["usd"]
            l0 += cfg["l0"]
            units += cfg["units"]
        if tier4_count > 0:
            cfg = EXCHANGE_CONFIG["tier4"]
            min_usd += cfg["min"] * tier4_count
            max_usd += cfg["max"] * tier4_count
            usd += cfg["usd"] * tier4_count
            l0 += cfg["l0"] * tier4_count
            units += cfg["units"] * tier4_count
        return {
            "mode": "multi_exchange",
            "selected_exchanges": selected,
            "tier4_exchange_count": tier4_count,
            "exchange_count": len(selected) + tier4_count,
            "min_usd": float(min_usd),
            "max_usd": float(max_usd),
            "usd": float(usd),
            "l0": float(l0),
            "units": float(units),
            "exchange_tier": "tier4" if tier4_count > 0 and len(selected) == 0 else "custom_multi",
        }

    # New UI can explicitly submit no exchanges (tge_buy_pressure_usd=0).
    # In that case do not inject legacy tier defaults.
    tge_resolved_input = getattr(params, "tge_buy_pressure_usd", None)
    exchange_manual_input = getattr(params, "exchange_tge_buy_usd", None)
    if exchange_manual_input is None and tge_resolved_input is not None:
        return {
            "mode": "no_exchange",
            "selected_exchanges": [],
            "tier4_exchange_count": 0,
            "exchange_count": 0,
            "min_usd": 0.0,
            "max_usd": 0.0,
            "usd": 0.0,
            "l0": 0.0,
            "units": 0.0,
            "exchange_tier": "none",
        }

    # Legacy fallback: single grouped tier.
    exchange_tier = (params.exchange_tier or "tier4").lower()
    if exchange_tier not in EXCHANGE_TIER_CONFIG:
        exchange_tier = "tier4"
    cfg = EXCHANGE_TIER_CONFIG[exchange_tier]
    return {
        "mode": "legacy_tier",
        "selected_exchanges": [],
        "tier4_exchange_count": 0,
        "exchange_count": 1,
        "min_usd": float(cfg["min"]),
        "max_usd": float(cfg["max"]),
        "usd": float(cfg["usd"]),
        "l0": float(cfg["l0"]),
        "units": float(cfg["units"]),
        "exchange_tier": exchange_tier,
    }


def _resolve_buy_pressure_model(params: MonteCarloInput, initial_circ_mcap: float = 0.0) -> Dict[str, object]:
    model = (params.buy_pressure_model or "score_v1").lower()
    if model not in {"score_v1", "tge_hype_v1", "hype_v2"}:
        model = "score_v1"

    exchange_info = _resolve_exchange_aggregate(params)

    # ── hype_v2: slider + checkboxes; exchanges = TGE only ──
    if model == "hype_v2":
        # TGE = raw exchange aggregate USD (no multiplier, no marketing coef)
        tge_buy_pressure_resolved = float(exchange_info["usd"])
        if params.tge_buy_pressure_usd is not None:
            tge_buy_pressure_resolved = max(float(params.tge_buy_pressure_usd), 0.0)

        # Ongoing = hype_base_fraction[level] × initial_circ_mcap × checkbox_multiplier
        hype_level = _clamp(int(params.hype_level or 5), 1, 10)
        base_frac = HYPE_BASE_FRACTIONS.get(hype_level, HYPE_BASE_FRACTIONS[5])

        checkbox_mult = 1.0
        if getattr(params, "hype_kol", False):
            checkbox_mult *= HYPE_CHECKBOX_MULTIPLIERS["kol"]
        if getattr(params, "hype_narrative", False):
            checkbox_mult *= HYPE_CHECKBOX_MULTIPLIERS["narrative"]
        if getattr(params, "hype_team", False):
            checkbox_mult *= HYPE_CHECKBOX_MULTIPLIERS["team"]
        if getattr(params, "hype_vc", False):
            checkbox_mult *= HYPE_CHECKBOX_MULTIPLIERS["vc"]

        ongoing_buy_monthly_resolved = max(initial_circ_mcap, 0.0) * base_frac * checkbox_mult
        if params.ongoing_buy_pressure_monthly_usd is not None:
            ongoing_buy_monthly_resolved = max(float(params.ongoing_buy_pressure_monthly_usd), 0.0)

        legacy_score_proxy = _legacy_score_from_tge_usd(tge_buy_pressure_resolved)

        return {
            "buy_pressure_model": model,
            "exchange_tier": str(exchange_info["exchange_tier"]),
            "exchange_mode": str(exchange_info["mode"]),
            "selected_exchanges": list(exchange_info["selected_exchanges"]),
            "tier4_exchange_count": int(exchange_info["tier4_exchange_count"]),
            "exchange_count": int(exchange_info["exchange_count"]),
            "exchange_range_min_usd": float(exchange_info["min_usd"]),
            "exchange_range_max_usd": float(exchange_info["max_usd"]),
            "exchange_l0_proxy_usd": float(exchange_info["l0"]),
            "exchange_listing_units": float(exchange_info["units"]),
            "exchange_tge_buy_usd": float(exchange_info["usd"]),
            "exchange_multiplier": 1.0,
            "marketing_budget_usd": 0.0,
            "marketing_coef": 0.0,
            "hype_multiplier": 1.0,
            "hype_level": hype_level,
            "hype_kol": bool(getattr(params, "hype_kol", False)),
            "hype_narrative": bool(getattr(params, "hype_narrative", False)),
            "hype_team": bool(getattr(params, "hype_team", False)),
            "hype_vc": bool(getattr(params, "hype_vc", False)),
            "hype_checkbox_multiplier": float(checkbox_mult),
            "tge_buy_pressure_usd": float(tge_buy_pressure_resolved),
            "ongoing_buy_pressure_monthly_usd": float(ongoing_buy_monthly_resolved),
            "legacy_score_proxy": float(legacy_score_proxy),
        }

    # ── tge_hype_v1 / score_v1: original logic ──
    exchange_tge_buy_usd = (
        float(params.exchange_tge_buy_usd)
        if params.exchange_tge_buy_usd is not None and params.exchange_tge_buy_usd >= 0
        else float(exchange_info["usd"])
    )
    exchange_multiplier = _clamp(float(params.exchange_multiplier or 1.0), 0.0, 50.0)
    marketing_budget_usd = max(float(params.marketing_budget_usd or 0.0), 0.0)
    marketing_coef = (
        float(params.marketing_coef)
        if params.marketing_coef is not None
        else _marketing_coef_from_budget(marketing_budget_usd)
    )
    marketing_coef = _clamp(marketing_coef, 0.0, 2.0)

    tge_buy_pressure_resolved = exchange_tge_buy_usd * exchange_multiplier * (1.0 + marketing_coef)
    if params.tge_buy_pressure_usd is not None:
        tge_buy_pressure_resolved = max(float(params.tge_buy_pressure_usd), 0.0)

    hype_multiplier = _clamp(float(params.hype_multiplier or 1.0), 0.5, 10.0)
    ongoing_buy_monthly_resolved = tge_buy_pressure_resolved * hype_multiplier
    if params.ongoing_buy_pressure_monthly_usd is not None:
        ongoing_buy_monthly_resolved = max(float(params.ongoing_buy_pressure_monthly_usd), 0.0)

    legacy_score_proxy = _legacy_score_from_tge_usd(tge_buy_pressure_resolved)

    return {
        "buy_pressure_model": model,
        "exchange_tier": str(exchange_info["exchange_tier"]),
        "exchange_mode": str(exchange_info["mode"]),
        "selected_exchanges": list(exchange_info["selected_exchanges"]),
        "tier4_exchange_count": int(exchange_info["tier4_exchange_count"]),
        "exchange_count": int(exchange_info["exchange_count"]),
        "exchange_range_min_usd": float(exchange_info["min_usd"]),
        "exchange_range_max_usd": float(exchange_info["max_usd"]),
        "exchange_l0_proxy_usd": float(exchange_info["l0"]),
        "exchange_listing_units": float(exchange_info["units"]),
        "exchange_tge_buy_usd": float(exchange_tge_buy_usd),
        "exchange_multiplier": float(exchange_multiplier),
        "marketing_budget_usd": float(marketing_budget_usd),
        "marketing_coef": float(marketing_coef),
        "hype_multiplier": float(hype_multiplier),
        "tge_buy_pressure_usd": float(tge_buy_pressure_resolved if model == "tge_hype_v1" else 0.0),
        "ongoing_buy_pressure_monthly_usd": float(ongoing_buy_monthly_resolved if model == "tge_hype_v1" else 0.0),
        "legacy_score_proxy": float(legacy_score_proxy),
    }


def _behavior_archetype(cat: Category) -> str:
    profile = (cat.holder_profile or "other").lower()
    if profile in {"team", "advisors", "treasury", "project"}:
        return "slow"
    if profile in {"airdrop", "community", "ecosystem", "kol"}:
        # KOL продают как фармеры аирдропа — быстро и почти всё
        return "emission"
    if profile in {"investor_round", "public_sale"}:
        return "investor"
    return "default"


def _default_behavior(cat: Category) -> Dict[str, float]:
    profile = _behavior_archetype(cat)

    if profile == "slow":
        # Team/advisors/treasury: медленные продажи, предсказуемое поведение
        # half_life 90 дней — команда растягивает продажи на 3+ месяца,
        # репутационные риски и внутренние lock-up не позволяют дампить быстро
        defaults = {
            "half_life_days": 90.0,
            "concentration": 32.0,
            "half_life_sigma": 0.40,
            "speed_vol": 0.20,
            "momentum_beta": 0.25,
            "drawdown_beta": -0.15,
        }
    elif profile == "emission":
        # Airdrop/community/KOL: продают почти всё в первые 2-3 дня после получения.
        # half_life 2 дня — 50% уходит за 2 дня, 75% за 4 дня.
        # Высокая sigma = большой разброс: фармеры продают в день 1,
        # настоящие пользователи могут держать неделями.
        defaults = {
            "half_life_days": 2.0,
            "concentration": 14.0,
            "half_life_sigma": 0.80,
            "speed_vol": 0.45,
            "momentum_beta": 0.60,
            "drawdown_beta": -0.70,
        }
    elif profile == "investor":
        # VC/investor_round/public_sale: продают за ~1 неделю после cliff.
        # half_life 7 дней — быстрее команды, медленнее KOL.
        # Низкая concentration = большой разброс между фондами
        # (одни дампят сразу, другие держат месяцами).
        defaults = {
            "half_life_days": 7.0,
            "concentration": 16.0,
            "half_life_sigma": 0.65,
            "speed_vol": 0.40,
            "momentum_beta": 1.10,
            "drawdown_beta": -0.85,
        }
    else:
        defaults = {
            "half_life_days": 14.0,
            "concentration": 20.0,
            "half_life_sigma": 0.55,
            "speed_vol": 0.30,
            "momentum_beta": 0.75,
            "drawdown_beta": -0.50,
        }

    mean_pct = cat.terminal_sell_mean_pct if cat.terminal_sell_mean_pct is not None else cat.sell_pressure_pct
    if mean_pct == 0:
        terminal_mean = 0.0
    else:
        terminal_mean = _clamp(mean_pct / 100.0, 0.001, 0.999)

    concentration = cat.terminal_sell_concentration if cat.terminal_sell_concentration is not None else defaults["concentration"]
    half_life_days = cat.half_life_days_median if cat.half_life_days_median is not None else defaults["half_life_days"]
    half_life_sigma = cat.half_life_sigma if cat.half_life_sigma is not None else defaults["half_life_sigma"]
    speed_vol = cat.sell_speed_vol if cat.sell_speed_vol is not None else defaults["speed_vol"]
    momentum_beta = cat.momentum_beta if cat.momentum_beta is not None else defaults["momentum_beta"]
    drawdown_beta = cat.drawdown_beta if cat.drawdown_beta is not None else defaults["drawdown_beta"]

    return {
        "terminal_mean": float(terminal_mean),
        "concentration": float(max(concentration, 2.0)),
        "half_life_days": float(max(half_life_days, 0.25)),
        "half_life_sigma": float(max(half_life_sigma, 0.05)),
        "speed_vol": float(max(speed_vol, 0.01)),
        "momentum_beta": float(momentum_beta),
        "drawdown_beta": float(drawdown_beta),
    }




def _effective_steps_per_month(params: MonteCarloInput, n_categories: int):
    requested = GRANULARITY_STEPS[params.time_granularity]
    effective = params.time_granularity

    est_ops = params.num_simulations * params.horizon_months * requested * max(n_categories, 1)
    if requested == 30 and est_ops > 45_000_000:
        requested = 4
        effective = "weekly"

    est_ops = params.num_simulations * params.horizon_months * requested * max(n_categories, 1)
    if requested == 4 and est_ops > 140_000_000:
        requested = 1
        effective = "monthly"

    return requested, effective


def _build_step_unlocks(releases: dict, categories: List[Category], horizon: int, steps_per_month: int):
    n_categories = len(categories)
    n_steps = horizon * steps_per_month

    unlocks = np.zeros((n_categories, n_steps), dtype=float)

    for i, cat in enumerate(categories):
        monthly = np.array(releases[cat.name]["monthly_release"], dtype=float)
        unlocks[i, 0] = monthly[0]  # TGE at step 0
        for m in range(1, horizon):
            start = m * steps_per_month
            end = start + steps_per_month
            unlocks[i, start:end] = monthly[m] / steps_per_month

    total_unlocks = np.sum(unlocks, axis=0)
    return unlocks, total_unlocks


def _to_monthly_close(values: np.ndarray, horizon: int, steps_per_month: int) -> np.ndarray:
    # Берём цену на конец каждого месяца (последний шаг)
    return values.reshape(horizon, steps_per_month)[:, -1]


def _prepend_listing_price(monthly: np.ndarray, p0: float) -> np.ndarray:
    """Добавляет листинговую цену как точку month=0 (TGE),
    сдвигая остальные месяцы на +1. Итоговый массив length = horizon+1."""
    return np.concatenate([[p0], monthly])


def _to_monthly_sum(values: np.ndarray, horizon: int, steps_per_month: int) -> np.ndarray:
    return values.reshape(horizon, steps_per_month).sum(axis=1)


def _to_monthly_mean(values: np.ndarray, horizon: int, steps_per_month: int) -> np.ndarray:
    return values.reshape(horizon, steps_per_month).mean(axis=1)


def _to_monthly_max(values: np.ndarray, horizon: int, steps_per_month: int) -> np.ndarray:
    return values.reshape(horizon, steps_per_month).max(axis=1)


def _quantile_table(arr: np.ndarray) -> Dict[str, List[float]]:
    out = {}
    for q in QUANTILES:
        key = f"p{int(q * 100):02d}"
        out[key] = np.quantile(arr, q, axis=0).tolist()
    out["mean"] = np.mean(arr, axis=0).tolist()
    return out


def _tge_circulating_tokens(params: MonteCarloInput) -> float:
    return sum(
        params.total_supply * (cat.allocation_pct / 100.0) * (cat.tge_unlock_pct / 100.0)
        for cat in params.categories
    )


def _calibrate_market_layer(params: MonteCarloInput, initial_circ_mcap: float) -> dict:
    """
    Derive Market Layer parameters from user inputs + prelaunch profile priors.

    Returns dict with:
      L0                 — USD per 1% move at launch
      adv_daily_usd      — average daily volume USD
      sigma_daily        — daily log-return volatility (GBM noise)
      impact_alpha_perm  — permanent impact coefficient (a in r_perm = a*sign(u)*|u|^β)
      impact_beta        — concavity exponent (β)
      liquidity_tau_steps — mean-reversion speed for L_t
      liquidity_shock_sigma — per-step L_t volatility
      initial_circ_mcap  — for reference
    """
    profile = PRELAUNCH_PROFILES.get(params.prelaunch_profile, PRELAUNCH_PROFILES["balanced"])

    # ── L0: USD per 1% move (long-run, persistent depth) ──
    # Priority: explicit liquidity_usd_per_1pct > l0_proxy_usd (from tiers) > profile-inferred.
    # Note: launch_liquidity_usd is NOT in this chain — it feeds TGE boost only (computed below).
    l0_from_proxy = False
    if params.liquidity_usd_per_1pct is not None and params.liquidity_usd_per_1pct > 0:
        L0 = float(params.liquidity_usd_per_1pct)
    elif getattr(params, 'l0_proxy_usd', None) and params.l0_proxy_usd > 0:
        # Proxy from frontend listing tier checkboxes:
        # T1=$5k, T2=$1.2k, T3=$350, DEX=$250 (× MM mult 1.5 if MM checked)
        L0 = max(float(params.l0_proxy_usd), 5_000.0)
        l0_from_proxy = True
    elif params.liquidity_depth_usd is not None and params.liquidity_depth_usd > 0:
        L0 = max(params.liquidity_depth_usd / 10.0, 10_000.0)
    else:
        # Almgren-Chriss: L0 = ADV_daily * eta_liquidity
        # New listing eta ≈ 1.0 (1 day of ADV moves price 1%)
        # ADV_daily = adv_frac * initial_circ_mcap (computed below)
        # But ADV not computed yet — derive directly from mcap × adv_frac × eta_liq
        # eta_liq for new listings: conservative=0.8, balanced=1.0, aggressive=1.5
        adv_frac = profile.get("adv_frac", 0.015)
        eta_liq_map = {"conservative": 0.8, "balanced": 1.0, "aggressive": 1.5}
        eta_liq = eta_liq_map.get(params.prelaunch_profile, 1.0)
        adv_inferred = max(initial_circ_mcap * adv_frac, 10_000.0)
        L0 = max(adv_inferred * eta_liq, _L_MIN_USD * 5)

    # ── ADV: average daily volume ──
    if params.adv_daily_usd is not None and params.adv_daily_usd > 0:
        adv = float(params.adv_daily_usd)
    elif getattr(params, 'listing_units', None) and params.listing_units > 0:
        # Proxy from exchange tier: listing_units × adv_frac × circ_mcap × demand_mult
        # demand_mult uses legacy score scale (0..10). For tge_hype_v1 this score is resolved
        # from TGE buy pressure as a compatibility proxy.
        demand_mult = 0.4 + 0.11 * _clamp(params.buy_pressure_score, 0.0, 10.0)
        adv_frac = profile.get("adv_frac", 0.012)
        adv = max(params.listing_units * adv_frac * initial_circ_mcap * demand_mult, 1_000.0)
    else:
        adv_frac = profile.get("adv_frac", 0.015)
        adv = max(initial_circ_mcap * adv_frac, 1_000.0)

    # ── Sigma (daily volatility) ──
    if params.sigma_daily is not None and params.sigma_daily > 0:
        sigma_daily = float(params.sigma_daily)
    else:
        # Amihud illiquidity proxy: higher ADV/mcap → lower vol
        adv_ratio = adv / max(initial_circ_mcap, 1.0)
        sigma_daily = _ADV_VOL_SCALE / max(math.sqrt(adv_ratio), 1e-6)
        sigma_daily = _clamp(sigma_daily, _GBM_VOL_FLOOR, 0.25)
        # Profile modulates: conservative market = more volatile
        sigma_daily *= profile.get("sigma_mod", 1.0)
        sigma_daily = _clamp(sigma_daily, _GBM_VOL_FLOOR, 0.30)

    # ── impact_alpha_perm ──
    # Calibration anchor: u=0.10 (10% of L_t in one step), beta=0.6
    #   → r_perm = alpha * 0.10^0.6 = alpha * 0.251
    #   Want 5% price impact → alpha = 0.05 / 0.251 ≈ 0.20
    # Default schema value 0.5 is intentionally higher (prelaunch = illiquid = more sensitive)
    alpha_base = params.impact_alpha_perm
    alpha_mod = profile.get("alpha_mod", 1.0)
    impact_alpha_perm = alpha_base * alpha_mod

    # ── TGE L0 boost from launch_liquidity_usd ──
    # If provided, treat it as a temporary L0 boost that decays to L0_base over tge_tau steps.
    # Physical meaning: at TGE market makers deploy more capital (tighter book),
    # then gradually withdraw to normal depth after a few weeks.
    # L0_tge = max(launch_liquidity_usd / 10.0, L0)  — boost ≥ base
    # Per-step: L0_eff(t) = L0_base + (L0_tge - L0_base) * exp(-t / tge_tau)
    tge_tau = float(getattr(params, 'launch_liquidity_tge_tau_steps', 3.0))
    if params.launch_liquidity_usd is not None and params.launch_liquidity_usd > 0:
        L0_tge = max(params.launch_liquidity_usd / 10.0, L0)
    else:
        L0_tge = L0  # no boost — L0_eff(t) = L0_base always

    return {
        "L0": float(L0),
        "L0_tge": float(L0_tge),
        "tge_tau_steps": float(tge_tau),
        "adv_daily_usd": float(adv),
        "sigma_daily": float(sigma_daily),
        "impact_alpha_perm": float(impact_alpha_perm),
        "impact_beta": float(params.impact_beta),
        "liquidity_tau_steps": float(params.liquidity_tau_steps),
        "liquidity_shock_sigma": float(params.liquidity_shock_sigma),
        "initial_circ_mcap": float(initial_circ_mcap),
        "buyback_exec_frac": float(params.buyback_exec_frac),
        "buyback_sink_mode": str(params.buyback_sink_mode),
        # Source tracking for model_info
        "l0_source": "explicit" if (params.liquidity_usd_per_1pct and params.liquidity_usd_per_1pct > 0)
                     else ("listing_proxy" if l0_from_proxy else "profile_inferred"),
        "listing_units": float(getattr(params, 'listing_units', None) or 0),
    }


def _buy_flow_profile(
    score: float,
    depth_usd: float,
    buyback_usd_per_month: float = 0.0,
    ongoing_buy_usd_per_month: float = 0.0,
    steps_per_month: int = 4,
) -> Dict[str, float]:
    """
    Mapping score 0..10 + deterministic monthly budgets -> exogenous buy flow за шаг.

    buy_load_base:          score-based спрос как доля circulating/шаг
    buy_load_max:           hard cap по токенам/шаг
    buy_usd_cap_step_base:  hard cap по USD/шаг (из depth)
    buyback_usd_per_step:   явный manual buyback бюджет USD/шаг (детерминирован, без шума)
    ongoing_buy_usd_per_step: ongoing market/hype buy USD/шаг (детерминирован, без шума)
    noise_sigma:            разброс между симуляциями для score-части
    """
    score_clamped = _clamp(score, 0.0, 10.0)

    # Deterministic budgets — convert monthly to per-step
    buyback_usd_per_step = max(buyback_usd_per_month, 0.0) / max(steps_per_month, 1)
    ongoing_buy_usd_per_step = max(ongoing_buy_usd_per_month, 0.0) / max(steps_per_month, 1)

    if score_clamped <= 0:
        return {
            "score": 0.0,
            "buy_load_base": 0.0,
            "buy_load_max": 0.0,
            "buy_depth_cap_frac": 0.0,
            "buy_usd_cap_step_base": 0.0,
            "buyback_usd_per_step": float(buyback_usd_per_step),
            "buyback_usd_per_month": float(buyback_usd_per_month),
            "ongoing_buy_usd_per_step": float(ongoing_buy_usd_per_step),
            "ongoing_buy_usd_per_month": float(ongoing_buy_usd_per_month),
            "noise_sigma": 0.0,
        }

    x = score_clamped / 10.0
    sat = math.tanh(1.25 * x) / math.tanh(1.25)  # насыщение при высоком score

    # Якоря: score 5 ≈ 0.19%/step, score 10 ≈ 0.34%/step
    buy_load_base = 0.0034 * (sat ** 1.35)
    buy_load_max = 0.0010 + 0.0038 * (x ** 1.15)  # ~0.10%..0.48%/step
    buy_depth_cap_frac = 0.15 * (x ** 1.20)       # до 15% depth на шаг
    buy_usd_cap_step_base = max(depth_usd, 0.0) * buy_depth_cap_frac

    # При более высоком score разброс между симуляциями чуть меньше.
    noise_sigma = _clamp(0.22 - 0.01 * score_clamped, 0.10, 0.22)

    return {
        "score": float(score_clamped),
        "buy_load_base": float(max(buy_load_base, 0.0)),
        "buy_load_max": float(max(buy_load_max, 0.0)),
        "buy_depth_cap_frac": float(max(buy_depth_cap_frac, 0.0)),
        "buy_usd_cap_step_base": float(max(buy_usd_cap_step_base, 0.0)),
        "buyback_usd_per_step": float(buyback_usd_per_step),
        "buyback_usd_per_month": float(buyback_usd_per_month),
        "ongoing_buy_usd_per_step": float(ongoing_buy_usd_per_step),
        "ongoing_buy_usd_per_month": float(ongoing_buy_usd_per_month),
        "noise_sigma": float(noise_sigma),
    }


def _sample_buy_flow(profile: Dict[str, float], rng: np.random.Generator) -> Dict[str, float]:
    # Buyback — детерминирован, шума нет (реальный контракт на фиксированную сумму)
    if profile["buy_load_base"] <= 0 or profile["score"] <= 0:
        return {
            **profile,
            "noise_mult": 1.0,
            "buy_load_target": 0.0,
            "buy_usd_cap_step": 0.0,
        }

    sigma = profile["noise_sigma"]
    noise_mult = float(rng.lognormal(mean=-0.5 * sigma * sigma, sigma=sigma))
    return {
        **profile,
        "noise_mult": noise_mult,
        "buy_load_target": float(profile["buy_load_base"] * noise_mult),
        "buy_usd_cap_step": float(profile["buy_usd_cap_step_base"] * noise_mult),
    }


def _simulate_one_path(
    rng: np.random.Generator,
    params: MonteCarloInput,
    release_steps: np.ndarray,
    total_release_steps: np.ndarray,
    cumulative_release_steps: np.ndarray,
    unlock_caps: np.ndarray,
    steps_per_month: int,
    behavior: Dict[str, np.ndarray],
    buy_flow_profile: Dict[str, float],
    tge_buy_pressure_usd: float,
    market_params: Dict,
) -> Dict[str, np.ndarray]:
    """
    Один прогон симуляции — Market Layer ABM.

    ABM sell-side (half-life, logit rates, Beta terminal fractions) сохранён.
    Price layer заменён: F_t = (buy_usd - sell_usd), L_t = stateful liquidity.
    r_perm = alpha * sign(u) * |u|^beta, u = F_t / L_t.
    P_{t+1} = P_t * exp(r_perm + eps_t), eps_t ~ N(0, sigma_step^2).
    """
    n_categories, n_steps = release_steps.shape
    horizon = params.horizon_months
    step_days = 30.0 / steps_per_month

    # ── Buy flow на этот путь (с логнормальным шумом) ──
    buy_flow = _sample_buy_flow(buy_flow_profile, rng)

    # ── Macro sentiment shock (один раз на симуляцию) ──
    # Моделирует состояние рынка при листинге: медвежий / нейтральный / бычий.
    # sigma=0.45 → P5..P95 ≈ 0.49x..2.03x давления, E[shock]=1.0 (LogN correction).
    _macro_sigma = 0.45
    macro_sell_shock = float(rng.lognormal(mean=-0.5 * _macro_sigma ** 2, sigma=_macro_sigma))

    # ── Market Layer state (инициализация на путь) ──
    L0_base = market_params["L0"]      # base (long-run) liquidity USD per 1% move
    L0_tge  = market_params["L0_tge"]  # TGE-period boosted L0 (≥ L0_base)
    tge_tau = market_params["tge_tau_steps"]  # decay constant in steps (0 = no boost)
    L_t = L0_tge                       # start at TGE boost level; decays to L0_base
    L0 = L0_base                       # regime logic uses L0 as the long-run anchor
    sigma_step = market_params["sigma_daily"] * math.sqrt(step_days)
    sigma_step = max(sigma_step, 0.005)  # floor: 0.5% per step
    L_regime = "normal"                # liquidity regime: normal | crash | euphoria
    buyback_burned_tokens = 0.0        # cumulative burned tokens (sink_mode="burn")
    buyback_treasury_tokens = 0.0      # cumulative treasury-locked tokens

    # ── Инициализация случайных параметров держателей (один раз на симуляцию) ──

    # 1. Сколько токенов держатель в итоге продаст — Beta-распределение
    zero_sell_mask = behavior["terminal_mean"] == 0.0
    alpha = np.maximum(behavior["terminal_mean"] * behavior["concentration"], 1e-3)
    beta_param = np.maximum((1.0 - behavior["terminal_mean"]) * behavior["concentration"], 1e-3)
    terminal_fraction = rng.beta(alpha, beta_param)
    terminal_fraction = np.where(zero_sell_mask, 0.0, terminal_fraction)
    # Macro shock масштабирует terminal_fraction (сколько решат продать в зависимости от настроения)
    terminal_fraction = np.clip(terminal_fraction * macro_sell_shock, 0.0, 1.0)
    terminal_caps = unlock_caps * terminal_fraction  # максимум токенов к продаже

    # 2. Скорость продаж — логнормальное распределение вокруг медианы half_life
    half_life_days = np.exp(rng.normal(np.log(np.maximum(behavior["half_life_days"], 1e-3)), behavior["half_life_sigma"]))
    half_life_steps = np.maximum(half_life_days / max(step_days, 1e-6), 1e-3)
    base_rate = 1.0 - np.exp(-math.log(2.0) / half_life_steps)
    base_rate = np.clip(base_rate, 1e-6, 1 - 1e-6)
    base_logit = np.log(base_rate / (1.0 - base_rate))

    # ── Состояние симуляции ──
    inventory = np.zeros(n_categories, dtype=float)  # токены в "кошельке" каждой категории
    sold_total = np.zeros(n_categories, dtype=float)  # сколько уже продано

    prices = np.zeros(n_steps, dtype=float)
    pressure_tokens_steps = np.zeros(n_steps, dtype=float)
    pressure_usd_steps = np.zeros(n_steps, dtype=float)
    sell_load_steps = np.zeros(n_steps, dtype=float)
    buy_tokens_steps = np.zeros(n_steps, dtype=float)
    buy_usd_steps = np.zeros(n_steps, dtype=float)
    buy_load_steps = np.zeros(n_steps, dtype=float)
    net_load_steps = np.zeros(n_steps, dtype=float)
    impact_steps = np.zeros(n_steps, dtype=float)
    unlock_ratio_steps = np.zeros(n_steps, dtype=float)

    p0 = params.initial_price
    price_floor = max(p0 * params.price_floor_ratio, 1e-12)
    prices[0] = p0
    peak_price = p0
    max_drawdown = 0.0

    # ── Основной цикл по шагам ──
    for t in range(n_steps):
        price_t = max(prices[t - 1] if t > 0 else p0, price_floor)
        step_price_ref = price_t
        prev_peak = peak_price

        # Сигналы для поведения держателей
        if t == 0:
            momentum = 0.0
        else:
            lookback = max(0, t - _MOMENTUM_WINDOW_STEPS)
            ref_price = max(prices[lookback], price_floor)
            momentum = math.log(max(price_t, price_floor) / ref_price)

        drawdown_signal = (price_t / prev_peak - 1.0) if prev_peak > 0 else 0.0

        # Разлок токенов на этом шаге
        inventory += release_steps[:, t]

        # Сколько можно продать (не превысить terminal_cap)
        remaining_cap = np.maximum(terminal_caps - sold_total, 0.0)
        sellable = np.minimum(inventory, remaining_cap)

        # Ставка продаж — логит с ценовыми сигналами и шумом
        if np.any(sellable > 0):
            noise = rng.normal(0.0, behavior["speed_vol"], size=n_categories)
            if params.market_response:
                logits = base_logit + behavior["momentum_beta"] * momentum + behavior["drawdown_beta"] * drawdown_signal + noise
                logits = np.clip(logits, -60.0, 60.0)
                rates = 1.0 / (1.0 + np.exp(-logits))
            else:
                rates = base_rate * np.exp(noise)
            rates = np.clip(rates, 0.0, 1.0)
            sold = sellable * rates
        else:
            sold = np.zeros(n_categories, dtype=float)

        sold_total += sold
        inventory -= sold

        sold_tokens_t = float(np.sum(sold))
        pressure_tokens_steps[t] = sold_tokens_t

        # Метрики нагрузки на рынок
        circulating_tokens_t = max(cumulative_release_steps[t], 1.0)
        prev_circulating = cumulative_release_steps[t - 1] if t > 0 else circulating_tokens_t
        unlock_ratio_steps[t] = total_release_steps[t] / max(prev_circulating, 1.0)
        sell_load = sold_tokens_t / circulating_tokens_t
        sell_load_steps[t] = sell_load

        # ── Exogenous buy flow (score-based stochastic component) ──
        buy_load_target = min(buy_flow["buy_load_target"], buy_flow["buy_load_max"])
        buy_tokens_t = 0.0
        if buy_load_target > 0:
            buy_tokens_t = buy_load_target * circulating_tokens_t
            if buy_flow["buy_usd_cap_step"] > 0 and step_price_ref > 0:
                tokens_from_usd_cap = buy_flow["buy_usd_cap_step"] / max(step_price_ref, price_floor)
                buy_tokens_t = min(buy_tokens_t, tokens_from_usd_cap)

        score_buy_tokens = buy_tokens_t  # score-based (no exec cap applied here)
        score_buy_usd = score_buy_tokens * step_price_ref

        # ── Deterministic ongoing buy + explicit buyback with shared execution cap ──
        # Both flows are constrained by execution cap, but sink mode applies only to buyback.
        exec_cap_usd_total = market_params["buyback_exec_frac"] * max(L_t, _L_MIN_USD)
        remaining_exec_cap_usd = exec_cap_usd_total

        ongoing_buy_usd_executed = 0.0
        if buy_flow["ongoing_buy_usd_per_step"] > 0 and step_price_ref > 0 and remaining_exec_cap_usd > 0:
            ongoing_buy_usd_executed = min(buy_flow["ongoing_buy_usd_per_step"], remaining_exec_cap_usd)
            remaining_exec_cap_usd -= ongoing_buy_usd_executed
            ongoing_tokens_t = ongoing_buy_usd_executed / max(step_price_ref, price_floor)
            buy_tokens_t += ongoing_tokens_t

        buyback_usd_executed = 0.0
        if buy_flow["buyback_usd_per_step"] > 0 and step_price_ref > 0 and remaining_exec_cap_usd > 0:
            buyback_usd_executed = min(buy_flow["buyback_usd_per_step"], remaining_exec_cap_usd)
            buyback_tokens_t = buyback_usd_executed / max(step_price_ref, price_floor)
            buy_tokens_t += buyback_tokens_t

        # ── Liquidity regime transition (per-step, path-level) ──
        u_roll = float(rng.random())
        if L_regime == "normal":
            if u_roll < _L_CRASH_PROB:
                L_regime = "crash"
            elif u_roll > 1.0 - _L_EUPHORIA_PROB:
                L_regime = "euphoria"
        elif L_regime == "crash":
            if u_roll < _L_RECOVERY_PROB:
                L_regime = "normal"
        elif L_regime == "euphoria":
            if u_roll < _L_EUPHORIA_DECAY:
                L_regime = "normal"

        # TGE decay: L0_eff decreases from L0_tge → L0_base as steps progress.
        # After 5×tau steps the residual boost is <1% of the initial boost.
        if tge_tau > 0 and L0_tge > L0_base:
            tge_decay = math.exp(-t / tge_tau)
            L0_eff = L0_base + (L0_tge - L0_base) * tge_decay
        else:
            L0_eff = L0_base

        L_target = L0_eff * _L_REGIME_FACTORS[L_regime]

        # ── Liquidity mean-reversion + shock ──
        tau = market_params["liquidity_tau_steps"]
        shock_L = float(rng.normal(0.0, market_params["liquidity_shock_sigma"]))
        L_t = L_t + (L_target - L_t) / tau + L_t * shock_L
        L_t = max(L_t, _L_MIN_USD)

        # ── Market Layer price update ──
        # F_t = net USD flow (positive = net buy pressure, negative = net sell)
        sell_usd_t = sold_tokens_t * step_price_ref
        tge_buy_usd_executed = tge_buy_pressure_usd if t == 0 else 0.0
        buy_usd_t = score_buy_usd + ongoing_buy_usd_executed + buyback_usd_executed + tge_buy_usd_executed
        buy_tokens_t_final = buy_usd_t / max(step_price_ref, price_floor)
        F_t = buy_usd_t - sell_usd_t

        # Dimensionless net flow: u = F_t / L_t
        u_t = F_t / max(L_t, _L_MIN_USD)

        # Permanent impact log-return (concave power law)
        alpha_perm = market_params["impact_alpha_perm"]
        beta_imp = market_params["impact_beta"]
        if u_t >= 0.0:
            r_perm = alpha_perm * (u_t ** beta_imp)
        else:
            r_perm = -alpha_perm * ((-u_t) ** beta_imp)
        r_perm = _clamp(r_perm, -_IMPACT_LOG_CAP, _IMPACT_LOG_CAP)

        # GBM noise: market microstructure
        eps_t = float(rng.normal(0.0, sigma_step))

        # Log-return price update (combined cap prevents single-step blow-up)
        total_lr = _clamp(r_perm + eps_t, -_IMPACT_LOG_CAP, _IMPACT_LOG_CAP)
        price_t = price_t * math.exp(total_lr)
        price_t = max(price_t, price_floor)

        # Impact fraction for KPI (signed: negative = upward)
        impact_fraction = math.exp(abs(r_perm)) - 1.0
        impact_steps[t] = math.copysign(impact_fraction, -r_perm)

        # ── Buyback sink effect ──
        sink_mode = market_params["buyback_sink_mode"]
        if buyback_usd_executed > 0 and sink_mode != "none":
            burned_this_step = buyback_usd_executed / max(step_price_ref, price_floor)
            if sink_mode == "burn":
                buyback_burned_tokens += burned_this_step
            elif sink_mode == "treasury":
                buyback_treasury_tokens += burned_this_step

        # ── Update circulating denominator for loads ──
        effective_circ = max(circulating_tokens_t - buyback_burned_tokens, 1.0)
        buy_tokens_t = buy_tokens_t_final
        buy_load = buy_tokens_t / effective_circ
        sell_load_adj = sold_tokens_t / effective_circ
        net_load = sell_load_adj - buy_load

        buy_tokens_steps[t] = buy_tokens_t
        buy_usd_steps[t] = buy_usd_t
        buy_load_steps[t] = buy_load
        sell_load_steps[t] = sell_load_adj
        net_load_steps[t] = net_load

        prices[t] = price_t
        pressure_usd_steps[t] = sold_tokens_t * step_price_ref

        if price_t > peak_price:
            peak_price = price_t
        if peak_price > 0:
            dd = price_t / peak_price - 1.0
            if dd < max_drawdown:
                max_drawdown = dd

    return {
        # Prepend listing price (p0) как точка month=0 (TGE, до impact).
        # Итоговый массив: [p0, price_month1, price_month2, ..., price_monthN]
        # Длина = horizon + 1, соответствует months = [0, 1, ..., horizon]
        "prices_monthly": _prepend_listing_price(_to_monthly_close(prices, horizon, steps_per_month), p0),
        "prices_twap_monthly": _prepend_listing_price(_to_monthly_mean(prices, horizon, steps_per_month), p0),
        "pressure_tokens_monthly": _to_monthly_sum(pressure_tokens_steps, horizon, steps_per_month),
        "pressure_usd_monthly": _to_monthly_sum(pressure_usd_steps, horizon, steps_per_month),
        "buy_tokens_monthly": _to_monthly_sum(buy_tokens_steps, horizon, steps_per_month),
        "buy_support_usd_monthly": _to_monthly_sum(buy_usd_steps, horizon, steps_per_month),
        "sell_load_monthly": _to_monthly_max(sell_load_steps, horizon, steps_per_month),
        "sell_load_monthly_mean": _to_monthly_mean(sell_load_steps, horizon, steps_per_month),
        "buy_load_monthly": _to_monthly_max(buy_load_steps, horizon, steps_per_month),
        "buy_load_monthly_mean": _to_monthly_mean(buy_load_steps, horizon, steps_per_month),
        "net_load_monthly": _to_monthly_max(net_load_steps, horizon, steps_per_month),
        "net_load_monthly_mean": _to_monthly_mean(net_load_steps, horizon, steps_per_month),
        "impact_monthly": _to_monthly_max(impact_steps, horizon, steps_per_month),
        "unlock_ratio_monthly": _to_monthly_max(unlock_ratio_steps, horizon, steps_per_month),
        "max_drawdown": np.array([max_drawdown]),
        "max_sell_load": np.array([float(np.max(sell_load_steps))]),
        "max_buy_load": np.array([float(np.max(buy_load_steps))]),
        "max_net_load": np.array([float(np.max(net_load_steps))]),
        "max_buy_support_usd_step": np.array([float(np.max(buy_usd_steps))]),
        "max_impact": np.array([float(np.max(impact_steps))]),
        "max_unlock_ratio": np.array([float(np.max(unlock_ratio_steps))]),
        "steps_sell_load_gt_threshold": np.array([int(np.sum(sell_load_steps > params.impact_free_absorption))], dtype=float),
        "steps_net_load_gt_threshold": np.array([int(np.sum(net_load_steps > params.impact_free_absorption))], dtype=float),
        "buy_flow_noise_mult": np.array([buy_flow["noise_mult"]], dtype=float),
    }


def run_monte_carlo(params: MonteCarloInput):
    """Run stochastic sell-pressure Monte Carlo — Market Layer ABM.

    ABM sell-side: half-life exponential decay, logit sell rates, Beta terminal fractions.
    Market layer: stateful liquidity L_t, USD net flow F_t, log-return price update.
    """
    horizon = params.horizon_months
    n_sims = params.num_simulations

    # Compute initial circulating mcap early (needed by hype_v2 resolver).
    initial_circ_tokens = _tge_circulating_tokens(params)
    if initial_circ_tokens <= 0:
        initial_circ_tokens = max(params.total_supply * 0.005, 1.0)
    initial_circ_mcap = initial_circ_tokens * params.initial_price

    # Resolve buy-pressure model inputs before calibration/simulation.
    buy_pressure_state = _resolve_buy_pressure_model(params, initial_circ_mcap=initial_circ_mcap)
    resolved_score_for_calibration = (
        float(params.buy_pressure_score)
        if buy_pressure_state["buy_pressure_model"] == "score_v1"
        else float(buy_pressure_state["legacy_score_proxy"])
    )
    resolved_l0_proxy = params.l0_proxy_usd
    if resolved_l0_proxy is None and buy_pressure_state["exchange_l0_proxy_usd"] > 0:
        resolved_l0_proxy = (
            float(buy_pressure_state["exchange_l0_proxy_usd"])
            * max(float(buy_pressure_state["exchange_multiplier"]), 0.1)
        )
    resolved_listing_units = params.listing_units
    if resolved_listing_units is None and buy_pressure_state["exchange_listing_units"] > 0:
        resolved_listing_units = (
            float(buy_pressure_state["exchange_listing_units"])
            * max(float(buy_pressure_state["exchange_multiplier"]), 0.1)
        )
    params = params.model_copy(update={
        "buy_pressure_model": buy_pressure_state["buy_pressure_model"],
        "buy_pressure_score": resolved_score_for_calibration,  # proxy used for calibration/compat fields
        "exchange_tier": buy_pressure_state["exchange_tier"],
        "selected_exchanges": buy_pressure_state["selected_exchanges"],
        "tier4_exchange_count": buy_pressure_state["tier4_exchange_count"],
        "exchange_tge_buy_usd": buy_pressure_state["exchange_tge_buy_usd"],
        "exchange_multiplier": buy_pressure_state["exchange_multiplier"],
        "marketing_budget_usd": buy_pressure_state["marketing_budget_usd"],
        "marketing_coef": buy_pressure_state["marketing_coef"],
        "hype_multiplier": buy_pressure_state["hype_multiplier"],
        "tge_buy_pressure_usd": buy_pressure_state["tge_buy_pressure_usd"],
        "ongoing_buy_pressure_monthly_usd": buy_pressure_state["ongoing_buy_pressure_monthly_usd"],
        "l0_proxy_usd": resolved_l0_proxy,
        "listing_units": resolved_listing_units,
    })

    market_params = _calibrate_market_layer(params, initial_circ_mcap)

    # Apply prelaunch profile threshold for KPI counting (impact_free_absorption)
    profile = PRELAUNCH_PROFILES.get(params.prelaunch_profile, PRELAUNCH_PROFILES["balanced"])
    params = params.model_copy(update={
        "impact_free_absorption": profile["impact_free_sell_ratio"],
    })

    prelaunch_info = {
        "prelaunch_profile_effective": params.prelaunch_profile,
        "inferred_initial_circulating_tokens": float(initial_circ_tokens),
        "inferred_initial_circulating_mcap_usd": float(initial_circ_mcap),
        "inferred_liquidity_L0_usd": float(market_params["L0"]),
    }

    releases = compute_release_schedule(params.categories, params.total_supply, horizon)
    steps_per_month, effective_granularity = _effective_steps_per_month(params, len(params.categories))
    release_steps, total_release_steps = _build_step_unlocks(releases, params.categories, horizon, steps_per_month)
    cumulative_release_steps = np.cumsum(total_release_steps)
    unlock_caps = np.sum(release_steps, axis=1)

    behavior_defaults = [_default_behavior(cat) for cat in params.categories]
    behavior = {
        "terminal_mean": np.array([b["terminal_mean"] for b in behavior_defaults], dtype=float),
        "concentration": np.array([b["concentration"] for b in behavior_defaults], dtype=float),
        "half_life_days": np.array([b["half_life_days"] for b in behavior_defaults], dtype=float),
        "half_life_sigma": np.array([b["half_life_sigma"] for b in behavior_defaults], dtype=float),
        "speed_vol": np.array([b["speed_vol"] for b in behavior_defaults], dtype=float),
        "momentum_beta": np.array([b["momentum_beta"] for b in behavior_defaults], dtype=float),
        "drawdown_beta": np.array([b["drawdown_beta"] for b in behavior_defaults], dtype=float),
    }

    depth_usd = market_params["L0"] * 100  # back-convert L0 to pool depth for buy_flow_profile
    buyback_monthly = params.buyback_usd_per_month if params.buyback_usd_per_month is not None else 0.0
    score_buy_input = params.buy_pressure_score if params.buy_pressure_model == "score_v1" else 0.0
    ongoing_buy_monthly = params.ongoing_buy_pressure_monthly_usd if params.buy_pressure_model in ("tge_hype_v1", "hype_v2") else 0.0
    tge_buy_pressure_usd = params.tge_buy_pressure_usd if params.buy_pressure_model in ("tge_hype_v1", "hype_v2") else 0.0

    buy_flow_profile = _buy_flow_profile(
        score=score_buy_input,
        depth_usd=depth_usd,
        buyback_usd_per_month=buyback_monthly,
        ongoing_buy_usd_per_month=ongoing_buy_monthly,
        steps_per_month=steps_per_month,
    )
    baseline_params = params.model_copy(update={
        "buy_pressure_score": 0.0,
        "buyback_usd_per_month": None,
        "tge_buy_pressure_usd": 0.0,
        "ongoing_buy_pressure_monthly_usd": 0.0,
    })
    baseline_buy_flow_profile = _buy_flow_profile(
        score=0.0,
        depth_usd=depth_usd,
        buyback_usd_per_month=0.0,
        ongoing_buy_usd_per_month=0.0,
        steps_per_month=steps_per_month,
    )
    run_baseline = score_buy_input > 0 or buyback_monthly > 0 or ongoing_buy_monthly > 0 or tge_buy_pressure_usd > 0

    all_prices = np.zeros((n_sims, horizon + 1), dtype=float)  # +1 для TGE (month=0 = listing price)
    all_twap_prices = np.zeros((n_sims, horizon + 1), dtype=float)  # TWAP (mean within month)
    all_pressure_tokens = np.zeros((n_sims, horizon), dtype=float)
    all_pressure_usd = np.zeros((n_sims, horizon), dtype=float)
    all_buy_tokens = np.zeros((n_sims, horizon), dtype=float)
    all_buy_support_usd = np.zeros((n_sims, horizon), dtype=float)
    all_sell_load = np.zeros((n_sims, horizon), dtype=float)
    all_sell_load_mean = np.zeros((n_sims, horizon), dtype=float)
    all_buy_load = np.zeros((n_sims, horizon), dtype=float)
    all_buy_load_mean = np.zeros((n_sims, horizon), dtype=float)
    all_net_load = np.zeros((n_sims, horizon), dtype=float)
    all_net_load_mean = np.zeros((n_sims, horizon), dtype=float)
    all_impact = np.zeros((n_sims, horizon), dtype=float)
    all_unlock_ratio = np.zeros((n_sims, horizon), dtype=float)

    all_max_drawdown = np.zeros(n_sims, dtype=float)
    all_max_sell_load = np.zeros(n_sims, dtype=float)
    all_max_buy_load = np.zeros(n_sims, dtype=float)
    all_max_net_load = np.zeros(n_sims, dtype=float)
    all_max_buy_support_usd_step = np.zeros(n_sims, dtype=float)
    all_max_impact = np.zeros(n_sims, dtype=float)
    all_max_unlock_ratio = np.zeros(n_sims, dtype=float)
    all_steps_sell_load_gt_threshold = np.zeros(n_sims, dtype=float)
    all_steps_net_load_gt_threshold = np.zeros(n_sims, dtype=float)
    all_buy_noise_mult = np.zeros(n_sims, dtype=float)

    baseline_end_prices = np.zeros(n_sims, dtype=float)
    baseline_max_drawdown = np.zeros(n_sims, dtype=float)

    effective_seed = params.random_seed if params.random_seed is not None else int(np.random.default_rng().integers(0, 2**31))
    seed_rng = np.random.default_rng(effective_seed)
    path_seeds = seed_rng.integers(0, 2**31 - 1, size=n_sims, dtype=np.int64)

    for i in range(n_sims):
        path_seed = int(path_seeds[i])
        sim = _simulate_one_path(
            rng=np.random.default_rng(path_seed),
            params=params,
            release_steps=release_steps,
            total_release_steps=total_release_steps,
            cumulative_release_steps=cumulative_release_steps,
            unlock_caps=unlock_caps,
            steps_per_month=steps_per_month,
            behavior=behavior,
            buy_flow_profile=buy_flow_profile,
            tge_buy_pressure_usd=tge_buy_pressure_usd,
            market_params=market_params,
        )

        all_prices[i] = sim["prices_monthly"]
        all_twap_prices[i] = sim["prices_twap_monthly"]
        all_pressure_tokens[i] = sim["pressure_tokens_monthly"]
        all_pressure_usd[i] = sim["pressure_usd_monthly"]
        all_buy_tokens[i] = sim["buy_tokens_monthly"]
        all_buy_support_usd[i] = sim["buy_support_usd_monthly"]
        all_sell_load[i] = sim["sell_load_monthly"]
        all_sell_load_mean[i] = sim["sell_load_monthly_mean"]
        all_buy_load[i] = sim["buy_load_monthly"]
        all_buy_load_mean[i] = sim["buy_load_monthly_mean"]
        all_net_load[i] = sim["net_load_monthly"]
        all_net_load_mean[i] = sim["net_load_monthly_mean"]
        all_impact[i] = sim["impact_monthly"]
        all_unlock_ratio[i] = sim["unlock_ratio_monthly"]

        all_max_drawdown[i] = float(sim["max_drawdown"][0])
        all_max_sell_load[i] = float(sim["max_sell_load"][0])
        all_max_buy_load[i] = float(sim["max_buy_load"][0])
        all_max_net_load[i] = float(sim["max_net_load"][0])
        all_max_buy_support_usd_step[i] = float(sim["max_buy_support_usd_step"][0])
        all_max_impact[i] = float(sim["max_impact"][0])
        all_max_unlock_ratio[i] = float(sim["max_unlock_ratio"][0])
        all_steps_sell_load_gt_threshold[i] = float(sim["steps_sell_load_gt_threshold"][0])
        all_steps_net_load_gt_threshold[i] = float(sim["steps_net_load_gt_threshold"][0])
        all_buy_noise_mult[i] = float(sim["buy_flow_noise_mult"][0])

        if run_baseline:
            # Baseline market_params: same L_t dynamics but no buyback sink
            baseline_market_params = {**market_params, "buyback_sink_mode": "none"}
            sim_base = _simulate_one_path(
                rng=np.random.default_rng(path_seed),
                params=baseline_params,
                release_steps=release_steps,
                total_release_steps=total_release_steps,
                cumulative_release_steps=cumulative_release_steps,
                unlock_caps=unlock_caps,
                steps_per_month=steps_per_month,
                behavior=behavior,
                buy_flow_profile=baseline_buy_flow_profile,
                tge_buy_pressure_usd=0.0,
                market_params=baseline_market_params,
            )
            baseline_end_prices[i] = float(sim_base["prices_monthly"][-1])
            baseline_max_drawdown[i] = float(sim_base["max_drawdown"][0])
        else:
            baseline_end_prices[i] = float(sim["prices_monthly"][-1])
            baseline_max_drawdown[i] = float(sim["max_drawdown"][0])

    all_cumulative_pressure_usd = np.cumsum(all_pressure_usd, axis=1)
    all_pressure_usd_listing = all_pressure_tokens * params.initial_price
    all_cumulative_pressure_tokens = np.cumsum(all_pressure_tokens, axis=1)
    all_cumulative_pressure_usd_listing = np.cumsum(all_pressure_usd_listing, axis=1)
    all_cumulative_buy_support_usd = np.cumsum(all_buy_support_usd, axis=1)

    price_quantiles = _quantile_table(all_prices)
    twap_price_quantiles = _quantile_table(all_twap_prices)
    pressure_tokens_quantiles = _quantile_table(all_pressure_tokens)
    pressure_usd_quantiles = _quantile_table(all_pressure_usd)
    pressure_usd_listing_quantiles = _quantile_table(all_pressure_usd_listing)
    cumulative_pressure_quantiles = _quantile_table(all_cumulative_pressure_usd)
    cumulative_pressure_tokens_quantiles = _quantile_table(all_cumulative_pressure_tokens)
    cumulative_pressure_usd_listing_quantiles = _quantile_table(all_cumulative_pressure_usd_listing)
    buy_tokens_quantiles = _quantile_table(all_buy_tokens)
    buy_support_usd_quantiles = _quantile_table(all_buy_support_usd)
    cumulative_buy_support_quantiles = _quantile_table(all_cumulative_buy_support_usd)
    sell_load_quantiles = _quantile_table(all_sell_load)
    sell_load_mean_quantiles = _quantile_table(all_sell_load_mean)
    buy_load_quantiles = _quantile_table(all_buy_load)
    buy_load_mean_quantiles = _quantile_table(all_buy_load_mean)
    net_load_quantiles = _quantile_table(all_net_load)
    net_load_mean_quantiles = _quantile_table(all_net_load_mean)
    impact_quantiles = _quantile_table(all_impact)
    unlock_ratio_quantiles = _quantile_table(all_unlock_ratio)

    n_sample = min(80, n_sims)
    sample_indices = seed_rng.choice(n_sims, size=n_sample, replace=False)
    spaghetti_prices = [all_prices[int(idx)].tolist() for idx in sample_indices]
    spaghetti_pressure = [all_pressure_usd[int(idx)].tolist() for idx in sample_indices]
    spaghetti_pressure_tokens = [all_pressure_tokens[int(idx)].tolist() for idx in sample_indices]

    # monthly_stats: horizon точек (m=0..horizon-1).
    # all_prices[:,0] = p0 (TGE listing), all_prices[:,m+1] = цена конца месяца m.
    # all_pressure_usd[:,m] = давление за месяц m (как и раньше).
    monthly_stats = []
    for m in range(horizon):
        monthly_stats.append({
            "month": m,
            "label": f"T+{m}M" if m > 0 else "TGE",
            "price_mean": float(np.mean(all_prices[:, m + 1])),
            "price_p05": float(np.quantile(all_prices[:, m + 1], 0.05)),
            "price_p25": float(np.quantile(all_prices[:, m + 1], 0.25)),
            "price_p50": float(np.quantile(all_prices[:, m + 1], 0.50)),
            "price_p75": float(np.quantile(all_prices[:, m + 1], 0.75)),
            "price_p95": float(np.quantile(all_prices[:, m + 1], 0.95)),
            "pressure_usd_mean": float(np.mean(all_pressure_usd[:, m])),
            "pressure_usd_p05": float(np.quantile(all_pressure_usd[:, m], 0.05)),
            "pressure_usd_p25": float(np.quantile(all_pressure_usd[:, m], 0.25)),
            "pressure_usd_p50": float(np.quantile(all_pressure_usd[:, m], 0.50)),
            "pressure_usd_p75": float(np.quantile(all_pressure_usd[:, m], 0.75)),
            "pressure_usd_p95": float(np.quantile(all_pressure_usd[:, m], 0.95)),
            "pressure_tokens_p50": float(np.quantile(all_pressure_tokens[:, m], 0.50)),
            "pressure_usd_listing_p50": float(np.quantile(all_pressure_usd_listing[:, m], 0.50)),
            "cumulative_pressure_tokens_p50": float(np.quantile(all_cumulative_pressure_tokens[:, m], 0.50)),
            "cumulative_pressure_p50": float(np.quantile(all_cumulative_pressure_usd[:, m], 0.50)),
            "cumulative_pressure_usd_listing_p50": float(np.quantile(all_cumulative_pressure_usd_listing[:, m], 0.50)),
            "buy_support_usd_p50": float(np.quantile(all_buy_support_usd[:, m], 0.50)),
            "sell_load_ratio_max_p50": float(np.quantile(all_sell_load[:, m], 0.50)),
            "sell_load_ratio_mean_p50": float(np.quantile(all_sell_load_mean[:, m], 0.50)),
            "buy_load_ratio_max_p50": float(np.quantile(all_buy_load[:, m], 0.50)),
            "buy_load_ratio_mean_p50": float(np.quantile(all_buy_load_mean[:, m], 0.50)),
            "net_load_ratio_max_p50": float(np.quantile(all_net_load[:, m], 0.50)),
            "net_load_ratio_mean_p50": float(np.quantile(all_net_load_mean[:, m], 0.50)),
            "impact_p50": float(np.quantile(all_impact[:, m], 0.50)),
            "unlock_ratio_p50": float(np.quantile(all_unlock_ratio[:, m], 0.50)),
            # TWAP (mean price within month)
            "price_twap_p50": float(np.quantile(all_twap_prices[:, m + 1], 0.50)),
            "price_twap_mean": float(np.mean(all_twap_prices[:, m + 1])),
            # Backward compatibility aliases (all max-based now for consistency)
            "sell_load_ratio_p50": float(np.quantile(all_sell_load[:, m], 0.50)),
            "buy_load_ratio_p50": float(np.quantile(all_buy_load[:, m], 0.50)),
            "net_load_ratio_p50": float(np.quantile(all_net_load[:, m], 0.50)),
        })

    all_end_prices = all_prices[:, -1]
    max_monthly_pressure_tokens = np.max(all_pressure_tokens, axis=1)
    total_cumulative_tokens = all_cumulative_pressure_tokens[:, -1]
    total_cumulative_listing_usd = all_cumulative_pressure_usd_listing[:, -1]
    max_monthly_pressure = np.max(all_pressure_usd, axis=1)
    max_monthly_buy_support = np.max(all_buy_support_usd, axis=1)
    total_cumulative = all_cumulative_pressure_usd[:, -1]
    total_buy_support = all_cumulative_buy_support_usd[:, -1]
    baseline_median_end_price = float(np.median(baseline_end_prices))
    baseline_prob_drawdown_50pct = float(np.mean(baseline_max_drawdown < -0.5))
    baseline_prob_drawdown_80pct = float(np.mean(baseline_max_drawdown < -0.8))
    paired_end_price_delta = all_end_prices - baseline_end_prices
    # Symmetric percent difference is bounded to [-200, +200] and remains stable
    # when baseline prices are near-zero.
    denom = np.maximum(np.abs(all_end_prices) + np.abs(baseline_end_prices), 1e-12)
    paired_end_price_delta_pct_sym = 200.0 * (all_end_prices - baseline_end_prices) / denom
    delta_end_price_vs_baseline_pct = float(np.median(paired_end_price_delta_pct_sym))
    delta_pct_method = "symmetric"
    baseline_floor = max(params.initial_price * 0.01, 1e-10)
    baseline_collapse_share = float(np.mean(baseline_end_prices < baseline_floor))
    planned_monthly_buy_support_usd = float(ongoing_buy_monthly + buyback_monthly)
    planned_total_buy_support_usd = float(tge_buy_pressure_usd + planned_monthly_buy_support_usd * horizon)

    summary = {
        "prelaunch_profile": params.prelaunch_profile,
        "buy_pressure_model": params.buy_pressure_model,
        "num_simulations": n_sims,
        "horizon_months": horizon,
        "median_end_price": float(np.median(all_end_prices)),
        "mean_end_price": float(np.mean(all_end_prices)),
        "p05_end_price": float(np.quantile(all_end_prices, 0.05)),
        "p95_end_price": float(np.quantile(all_end_prices, 0.95)),
        "median_peak_pressure_tokens": float(np.median(max_monthly_pressure_tokens)),
        "median_total_pressure_tokens": float(np.median(total_cumulative_tokens)),
        "mean_total_pressure_tokens": float(np.mean(total_cumulative_tokens)),
        "median_total_pressure_usd_at_listing": float(np.median(total_cumulative_listing_usd)),
        "mean_total_pressure_usd_at_listing": float(np.mean(total_cumulative_listing_usd)),
        "median_peak_pressure_usd": float(np.median(max_monthly_pressure)),
        "median_total_pressure_usd": float(np.median(total_cumulative)),
        "mean_total_pressure_usd": float(np.mean(total_cumulative)),
        "median_peak_buy_support_usd": float(np.median(max_monthly_buy_support)),
        "median_peak_buy_support_usd_step": float(np.median(all_max_buy_support_usd_step)),
        "p95_peak_buy_support_usd_step": float(np.quantile(all_max_buy_support_usd_step, 0.95)),
        "median_total_buy_support_usd": float(np.median(total_buy_support)),
        "mean_total_buy_support_usd": float(np.mean(total_buy_support)),
        "median_max_drawdown": float(np.median(all_max_drawdown)),
        "p05_max_drawdown": float(np.quantile(all_max_drawdown, 0.05)),
        "prob_price_above_initial": float(np.mean(all_end_prices > params.initial_price)),
        "prob_drawdown_50pct": float(np.mean(all_max_drawdown < -0.5)),
        "prob_drawdown_80pct": float(np.mean(all_max_drawdown < -0.8)),
        "baseline_median_end_price": baseline_median_end_price,
        "baseline_prob_drawdown_50pct": baseline_prob_drawdown_50pct,
        "baseline_prob_drawdown_80pct": baseline_prob_drawdown_80pct,
        "delta_end_price_vs_baseline": float(np.median(paired_end_price_delta)),
        "delta_end_price_vs_baseline_pct": delta_end_price_vs_baseline_pct,
        "delta_end_price_vs_baseline_pct_method": delta_pct_method,
        "baseline_collapse_share": baseline_collapse_share,
        "delta_prob_drawdown_50pct_vs_baseline": float(np.mean(all_max_drawdown < -0.5) - baseline_prob_drawdown_50pct),
        "delta_prob_drawdown_80pct_vs_baseline": float(np.mean(all_max_drawdown < -0.8) - baseline_prob_drawdown_80pct),
        "price_change_median_pct": float((np.median(all_end_prices) / params.initial_price - 1.0) * 100),
        "median_peak_sell_load_ratio": float(np.median(all_max_sell_load)),
        "median_peak_buy_load_ratio": float(np.median(all_max_buy_load)),
        "median_peak_net_load_ratio": float(np.median(all_max_net_load)),
        "prob_peak_load_gt_threshold": float(np.mean(all_max_net_load > params.impact_free_absorption)),
        "median_steps_load_gt_threshold": float(np.median(all_steps_net_load_gt_threshold)),
        "prob_peak_sell_load_gt_threshold": float(np.mean(all_max_sell_load > params.impact_free_absorption)),
        "prob_peak_net_load_gt_threshold": float(np.mean(all_max_net_load > params.impact_free_absorption)),
        "median_steps_sell_load_gt_threshold": float(np.median(all_steps_sell_load_gt_threshold)),
        "median_steps_net_load_gt_threshold": float(np.median(all_steps_net_load_gt_threshold)),
        "median_peak_impact_step_pct": float(np.median(all_max_impact) * 100.0),
        "median_peak_unlock_ratio_pct": float(np.median(all_max_unlock_ratio) * 100.0),
        "time_granularity_requested": params.time_granularity,
        "time_granularity_used": effective_granularity,
        "steps_per_month": steps_per_month,
        "random_seed_used": effective_seed,
        "tge_buy_pressure_usd": float(tge_buy_pressure_usd),
        "ongoing_buy_pressure_monthly_usd": float(ongoing_buy_monthly),
        "manual_buyback_usd_per_month": float(buyback_monthly),
        "planned_monthly_buy_support_usd": planned_monthly_buy_support_usd,
        "planned_total_buy_support_usd": planned_total_buy_support_usd,
        # Market Layer calibration info
        "market_L0_usd": float(market_params["L0"]),
        "market_sigma_daily_pct": float(market_params["sigma_daily"] * 100),
        "market_adv_daily_usd": float(market_params["adv_daily_usd"]),
        "market_impact_alpha_perm": float(market_params["impact_alpha_perm"]),
        "market_impact_beta": float(market_params["impact_beta"]),
        "buyback_sink_mode": market_params["buyback_sink_mode"],
    }

    release_schedule = {
        cat.name: releases[cat.name]["monthly_release"]
        for cat in params.categories
    }

    agent_profiles = []
    for i, cat in enumerate(params.categories):
        bd = behavior_defaults[i]
        archetype = _behavior_archetype(cat)
        agent_profiles.append({
            "name": cat.name,
            "holder_profile": cat.holder_profile,
            "profile": archetype,
            "terminal_sell_pct": round(bd["terminal_mean"] * 100, 1),
            "half_life_days": round(bd["half_life_days"], 1),
            "momentum_beta": round(bd["momentum_beta"], 2),
            "drawdown_beta": round(bd["drawdown_beta"], 2),
        })

    return {
        "summary": summary,
        "monthly_stats": monthly_stats,
        "price_quantiles": price_quantiles,
        "pressure_tokens_quantiles": pressure_tokens_quantiles,
        "pressure_usd_quantiles": pressure_usd_quantiles,
        "pressure_usd_listing_quantiles": pressure_usd_listing_quantiles,
        "cumulative_pressure_tokens_quantiles": cumulative_pressure_tokens_quantiles,
        "cumulative_pressure_quantiles": cumulative_pressure_quantiles,
        "cumulative_pressure_usd_listing_quantiles": cumulative_pressure_usd_listing_quantiles,
        "buy_support_usd_quantiles": buy_support_usd_quantiles,
        "cumulative_buy_support_quantiles": cumulative_buy_support_quantiles,
        "sell_load_quantiles": sell_load_quantiles,
        "sell_load_mean_quantiles": sell_load_mean_quantiles,
        "buy_load_quantiles": buy_load_quantiles,
        "buy_load_mean_quantiles": buy_load_mean_quantiles,
        "net_load_quantiles": net_load_quantiles,
        "net_load_mean_quantiles": net_load_mean_quantiles,
        "impact_quantiles": impact_quantiles,
        "unlock_ratio_quantiles": unlock_ratio_quantiles,
        "spaghetti_prices": spaghetti_prices,
        "spaghetti_pressure": spaghetti_pressure,
        "spaghetti_pressure_tokens": spaghetti_pressure_tokens,
        "buy_tokens_quantiles": buy_tokens_quantiles,
        "monthly_buy_support_usd": np.quantile(all_buy_support_usd, 0.50, axis=0).tolist(),
        "monthly_buy_load": np.quantile(all_buy_load_mean, 0.50, axis=0).tolist(),
        "monthly_buy_load_max": np.quantile(all_buy_load, 0.50, axis=0).tolist(),
        "monthly_sell_load_max": np.quantile(all_sell_load, 0.50, axis=0).tolist(),
        "monthly_net_load_mean": np.quantile(all_net_load_mean, 0.50, axis=0).tolist(),
        "monthly_net_load_max": np.quantile(all_net_load, 0.50, axis=0).tolist(),
        "release_schedule": release_schedule,
        "agent_profiles": agent_profiles,
        "months": list(range(horizon + 1)),  # 0=TGE, 1..horizon = конец каждого месяца
        "twap_price_quantiles": twap_price_quantiles,
        "model_info": {
            "prelaunch_profile": params.prelaunch_profile,
            "simulation_mode": "tokenomics_stress_market_layer",
            "buy_flow_mode": (
                "tge_oneoff_plus_ongoing_hype_plus_buyback"
                if params.buy_pressure_model == "tge_hype_v1"
                else "score_stochastic_plus_buyback"
            ),
            "buy_pressure_model": params.buy_pressure_model,
            "buy_pressure_score": params.buy_pressure_score,
            "buy_pressure_score_legacy_proxy": buy_pressure_state["legacy_score_proxy"],
            "exchange_tier": params.exchange_tier,
            "exchange_mode": buy_pressure_state["exchange_mode"],
            "selected_exchanges": buy_pressure_state["selected_exchanges"],
            "tier4_exchange_count": buy_pressure_state["tier4_exchange_count"],
            "exchange_count": buy_pressure_state["exchange_count"],
            "exchange_range_min_usd": buy_pressure_state["exchange_range_min_usd"],
            "exchange_range_max_usd": buy_pressure_state["exchange_range_max_usd"],
            "exchange_tge_buy_usd": params.exchange_tge_buy_usd,
            "exchange_multiplier": params.exchange_multiplier,
            "marketing_budget_usd": params.marketing_budget_usd,
            "marketing_coef": params.marketing_coef,
            "hype_multiplier": params.hype_multiplier,
            "tge_buy_pressure_usd": float(tge_buy_pressure_usd),
            "ongoing_buy_pressure_monthly_usd": float(ongoing_buy_monthly),
            "market_response": params.market_response,
            "impact_enabled": params.impact_enabled,
            "impact_free_absorption": params.impact_free_absorption,
            # Market Layer (replaces impact_eta)
            "market_L0_usd": float(market_params["L0"]),
            "market_sigma_daily_pct": float(market_params["sigma_daily"] * 100),
            "market_adv_daily_usd": float(market_params["adv_daily_usd"]),
            "market_impact_alpha_perm": float(market_params["impact_alpha_perm"]),
            "market_impact_beta": float(market_params["impact_beta"]),
            "liquidity_tau_steps": float(market_params["liquidity_tau_steps"]),
            "liquidity_shock_sigma": float(market_params["liquidity_shock_sigma"]),
            "buyback_sink_mode": market_params["buyback_sink_mode"],
            "buyback_exec_frac": float(market_params["buyback_exec_frac"]),
            "buy_load_base_per_step": buy_flow_profile["buy_load_base"],
            "buy_load_max_per_step": buy_flow_profile["buy_load_max"],
            "buy_depth_cap_fraction": buy_flow_profile["buy_depth_cap_frac"],
            "buy_usd_cap_step_base": buy_flow_profile["buy_usd_cap_step_base"],
            "buyback_usd_per_month": buy_flow_profile["buyback_usd_per_month"],
            "buyback_usd_per_step": buy_flow_profile["buyback_usd_per_step"],
            "ongoing_buy_usd_per_month": buy_flow_profile["ongoing_buy_usd_per_month"],
            "ongoing_buy_usd_per_step": buy_flow_profile["ongoing_buy_usd_per_step"],
            "buy_noise_sigma": buy_flow_profile["noise_sigma"],
            "buy_flow_noise_mult_mean": float(np.mean(all_buy_noise_mult)),
            "buy_flow_noise_mult_p05": float(np.quantile(all_buy_noise_mult, 0.05)),
            "buy_flow_noise_mult_p95": float(np.quantile(all_buy_noise_mult, 0.95)),
            "deprecated_metrics": [
                "prob_peak_sell_load_gt_threshold",
                "median_steps_sell_load_gt_threshold",
                "prob_price_above_initial",
            ],
            "inferred_initial_circulating_tokens": prelaunch_info.get("inferred_initial_circulating_tokens"),
            "inferred_initial_circulating_mcap_usd": prelaunch_info.get("inferred_initial_circulating_mcap_usd"),
            "inferred_liquidity_L0_usd": prelaunch_info.get("inferred_liquidity_L0_usd"),
            "l0_source": market_params.get("l0_source", "profile_inferred"),
            "listing_units": market_params.get("listing_units", 0.0),
            # TGE L0 boost
            "launch_liquidity_tge_boost": float(market_params["L0_tge"]) if market_params["L0_tge"] > market_params["L0"] else 0.0,
            "launch_liquidity_tge_tau_steps": float(market_params["tge_tau_steps"]),
        },
    }
