from typing import List

from pydantic import BaseModel, Field, field_validator


class Category(BaseModel):
    name: str
    holder_profile: str = "other"
    allocation_pct: float
    tge_unlock_pct: float
    cliff_months: int
    vesting_months: int
    sell_pressure_pct: float

    # Optional advanced behavior overrides for Monte Carlo
    terminal_sell_mean_pct: float = None
    terminal_sell_concentration: float = None
    half_life_days_median: float = None
    half_life_sigma: float = None
    sell_speed_vol: float = None
    momentum_beta: float = None
    drawdown_beta: float = None

    @field_validator("allocation_pct", "tge_unlock_pct", "sell_pressure_pct")
    @classmethod
    def pct_range(cls, v, info):
        if v < 0 or v > 100:
            raise ValueError(f"{info.field_name} must be between 0 and 100")
        return v

    @field_validator("cliff_months", "vesting_months")
    @classmethod
    def non_negative_months(cls, v):
        if v < 0:
            raise ValueError("months must be >= 0")
        return v

    @field_validator("holder_profile")
    @classmethod
    def valid_holder_profile(cls, v):
        vv = str(v).lower().strip()
        allowed = {
            "investor_round",
            "public_sale",
            "airdrop",
            "community",
            "ecosystem",
            "team",
            "advisors",
            "treasury",
            "project",
            "kol",
            "liquidity",
            "other",
        }
        if vv not in allowed:
            raise ValueError(
                "holder_profile must be one of: investor_round, public_sale, airdrop, community, ecosystem, "
                "team, advisors, treasury, project, kol, liquidity, other"
            )
        return vv

    def model_post_init(self, __context):
        if self.vesting_months == 0 and self.tge_unlock_pct < 100:
            raise ValueError(
                f"Category '{self.name}': vesting_months=0 but tge_unlock_pct={self.tge_unlock_pct}% "
                f"— the remaining {100 - self.tge_unlock_pct}% of tokens would never unlock. "
                f"Set tge_unlock_pct=100 or add vesting_months > 0."
            )

    @field_validator("terminal_sell_mean_pct")
    @classmethod
    def optional_pct_range(cls, v):
        if v is None:
            return v
        if v < 0 or v > 100:
            raise ValueError("terminal_sell_mean_pct must be between 0 and 100")
        return v

    @field_validator(
        "terminal_sell_concentration",
        "half_life_days_median",
        "half_life_sigma",
        "sell_speed_vol",
    )
    @classmethod
    def optional_positive(cls, v, info):
        if v is None:
            return v
        if v <= 0:
            raise ValueError(f"{info.field_name} must be > 0")
        return v


class ProjectInput(BaseModel):
    project_name: str
    total_supply: float
    initial_price: float
    horizon_months: int
    categories: List[Category]

    @field_validator("total_supply")
    @classmethod
    def supply_positive(cls, v):
        if v <= 0:
            raise ValueError("total_supply must be > 0")
        return v

    @field_validator("initial_price")
    @classmethod
    def price_positive(cls, v):
        if v <= 0:
            raise ValueError("initial_price must be > 0")
        return v

    @field_validator("horizon_months")
    @classmethod
    def horizon_range(cls, v):
        if v < 1 or v > 120:
            raise ValueError("horizon_months must be between 1 and 120")
        return v


class MonteCarloInput(BaseModel):
    project_name: str
    total_supply: float
    initial_price: float
    horizon_months: int
    num_simulations: int
    categories: List[Category]

    # Pre-listing mode — всегда stress test, параметры рынка не нужны
    prelaunch_profile: str = "balanced"  # conservative | balanced | aggressive
    launch_liquidity_usd: float = None   # опционально: реальная глубина стакана на старте

    # Simulation controls
    time_granularity: str = "weekly"  # daily | weekly | monthly
    market_response: bool = True

    # Price-impact model (калибруется автоматически из ликвидности)
    impact_enabled: bool = True
    impact_free_absorption: float = 0.0012
    impact_eta: float = 0.08

    # Liquidity depth (опционально — если задано, калибрует impact_eta)
    liquidity_depth_usd: float = None

    # Buy pressure legacy score 0..10 (backward compatibility)
    # 0 = чистый стресс-тест (нет покупателей), 10 = максимальный спрос
    buy_pressure_score: float = 0.0

    # Buy pressure engine model selector:
    # score_v1     — legacy score-based flow
    # tge_hype_v1  — exchange tier + marketing (TGE one-off) + hype (ongoing monthly)
    # hype_v2      — hype slider (1-10) + modifier checkboxes; exchanges = TGE only
    buy_pressure_model: str = "score_v1"

    # Exchange-tier inputs for tge_hype_v1
    exchange_tier: str = "tier4"
    selected_exchanges: List[str] = Field(default_factory=list)
    tier4_exchange_count: int = 0
    exchange_tge_buy_usd: float = None               # optional manual override; >= 0
    exchange_multiplier: float = 1.0                 # global multiplier; >= 0

    # Marketing inputs for tge_hype_v1
    marketing_budget_usd: float = 0.0                # >= 0
    marketing_coef: float = None                     # optional override; >= 0

    # Project hype multiplier (ongoing monthly flow)
    hype_multiplier: float = 1.0                     # [0.5, 10]

    # hype_v2 inputs — slider-based hype level + modifier checkboxes
    hype_level: int = 5                              # 1-10 hype slider
    hype_kol: bool = False                           # KOL/influencer backing
    hype_narrative: bool = False                     # strong narrative / meta alignment
    hype_team: bool = False                          # known team / previous success
    hype_vc: bool = False                            # VC / institutional backing

    # Pre-resolved values (frontend may send resolved numbers directly)
    tge_buy_pressure_usd: float = None               # one-off at TGE; >= 0
    ongoing_buy_pressure_monthly_usd: float = None   # ongoing monthly buy; >= 0

    # Explicit buyback budget (USD/month) — added on top of ongoing buy support.
    # Converted into per-step executed buy via execution-cap constraints.
    buyback_usd_per_month: float = None

    # ── Market Layer calibration anchors ──
    # L0: USD per 1% move at launch. If None, inferred from launch_liquidity_usd or prelaunch profile.
    liquidity_usd_per_1pct: float = None        # explicit L0; > 0 if set

    # ADV: average daily volume in USD. If None, inferred as adv_frac * initial_circulating_mcap.
    adv_daily_usd: float = None                 # > 0 if set

    # GBM noise: daily volatility (log-return sigma). If None, calibrated from ADV.
    # Typical range: 0.03–0.15 (3%–15% daily vol).
    sigma_daily: float = None                   # > 0 if set; overrides ADV-based calibration

    # Permanent impact: r_perm = impact_alpha_perm * sign(u) * |u|^impact_beta
    # Calibration anchor: u=0.10 (10% of L_t/step), beta=0.6 → r_perm ≈ 2.5% at alpha=0.10
    impact_alpha_perm: float = 0.10            # permanent impact coefficient (a)
    impact_beta: float = 0.6                   # concavity exponent (β), must be in (0, 2]

    # Liquidity dynamics: L_{t+1} = L_t + (L_target - L_t)/tau + L_t * shock
    liquidity_tau_steps: float = 26.0          # mean-reversion speed in steps (weekly: ~6.5 months)
    liquidity_shock_sigma: float = 0.05        # per-step liquidity volatility (~5%/step)

    # Buyback sink mode: what happens to bought-back tokens
    # "none"      — no supply sink (only price impact)
    # "burn"      — tokens permanently removed from circulating
    # "treasury"  — tokens removed from float temporarily
    buyback_sink_mode: str = "none"

    # Execution cap: deterministic support (ongoing flow + manual buyback)
    # executes at most exec_frac * L_t USD per step
    buyback_exec_frac: float = 0.20

    # TGE liquidity boost decay: launch_liquidity_usd acts as a temporary L0 boost
    # that decays to the base L0 over tge_tau_steps steps.
    # L_actual(t) = L0_base + (L0_tge - L0_base) * exp(-t / tge_tau_steps)
    # Default 3.0 steps ≈ 3 weeks at weekly granularity (effect < 5% of boost by step 9).
    launch_liquidity_tge_tau_steps: float = 3.0

    # ── Listing proxy (computed by frontend from exchange tier) ──
    # l0_proxy_usd: USD per 1% move inferred from exchange tier
    # Overrides profile-inferred L0, but loses to explicit liquidity_usd_per_1pct
    l0_proxy_usd: float = None          # > 0 if set

    # listing_units: weighted exchange-tier intensity used for ADV proxy on backend
    listing_units: float = None         # >= 0 if set

    # Внутренние константы (не меняются пользователем)
    project_stage: str = "prelaunch"
    simulation_mode: str = "tokenomics_stress"
    use_regimes: bool = False
    annual_volatility: float = 0.0
    annual_drift: float = 0.0
    price_floor_ratio: float = 1e-6
    random_seed: int = None

    @field_validator("total_supply")
    @classmethod
    def supply_positive(cls, v):
        if v <= 0:
            raise ValueError("total_supply must be > 0")
        return v

    @field_validator("initial_price")
    @classmethod
    def price_positive(cls, v):
        if v <= 0:
            raise ValueError("initial_price must be > 0")
        return v

    @field_validator("horizon_months")
    @classmethod
    def horizon_range(cls, v):
        if v < 1 or v > 120:
            raise ValueError("horizon_months must be between 1 and 120")
        return v

    @field_validator("num_simulations")
    @classmethod
    def sim_range(cls, v):
        if v < 100 or v > 20000:
            raise ValueError("num_simulations must be between 100 and 20000")
        return v

    @field_validator("time_granularity")
    @classmethod
    def valid_granularity(cls, v):
        vv = str(v).lower()
        if vv not in {"daily", "weekly", "monthly"}:
            raise ValueError("time_granularity must be one of: daily, weekly, monthly")
        return vv

    @field_validator("prelaunch_profile")
    @classmethod
    def valid_prelaunch_profile(cls, v):
        vv = str(v).lower()
        if vv not in {"conservative", "balanced", "aggressive"}:
            raise ValueError("prelaunch_profile must be one of: conservative, balanced, aggressive")
        return vv

    @field_validator("buy_pressure_score")
    @classmethod
    def buy_pressure_range(cls, v):
        if v < 0 or v > 10:
            raise ValueError("buy_pressure_score must be between 0 and 10")
        return v

    @field_validator("buy_pressure_model")
    @classmethod
    def valid_buy_pressure_model(cls, v):
        vv = str(v).lower().strip()
        if vv not in {"score_v1", "tge_hype_v1", "hype_v2"}:
            raise ValueError("buy_pressure_model must be one of: score_v1, tge_hype_v1, hype_v2")
        return vv

    @field_validator("exchange_tier")
    @classmethod
    def valid_exchange_tier(cls, v):
        vv = str(v).lower().strip()
        allowed = {
            "tier4",
            "mexc_gate_kucoin",
            "bingx_htx",
            "binance_alpha",
            "okx_bybit",
            "binance",
        }
        if vv not in allowed:
            raise ValueError(
                "exchange_tier must be one of: tier4, mexc_gate_kucoin, bingx_htx, binance_alpha, okx_bybit, binance"
            )
        return vv

    @field_validator("selected_exchanges")
    @classmethod
    def valid_selected_exchanges(cls, v):
        if v is None:
            return []
        allowed = {
            "mexc",
            "gate",
            "kucoin",
            "bingx",
            "htx",
            "binance_alpha",
            "okx",
            "bybit",
            "binance",
        }
        seen = set()
        out = []
        for raw in v:
            vv = str(raw).lower().strip()
            if not vv:
                continue
            if vv not in allowed:
                raise ValueError(
                    "selected_exchanges contains invalid value. Allowed: mexc, gate, kucoin, bingx, htx, "
                    "binance_alpha, okx, bybit, binance"
                )
            if vv not in seen:
                seen.add(vv)
                out.append(vv)
        return out

    @field_validator("tier4_exchange_count")
    @classmethod
    def tier4_count_range(cls, v):
        if v < 0 or v > 200:
            raise ValueError("tier4_exchange_count must be between 0 and 200")
        return v

    @field_validator(
        "exchange_tge_buy_usd",
        "marketing_budget_usd",
        "marketing_coef",
        "tge_buy_pressure_usd",
        "ongoing_buy_pressure_monthly_usd",
    )
    @classmethod
    def optional_non_negative_buy_inputs(cls, v, info):
        if v is None:
            return v
        if v < 0:
            raise ValueError(f"{info.field_name} must be >= 0")
        return v

    @field_validator("exchange_multiplier")
    @classmethod
    def exchange_multiplier_range(cls, v):
        if v < 0 or v > 50:
            raise ValueError("exchange_multiplier must be between 0 and 50")
        return v

    @field_validator("marketing_coef")
    @classmethod
    def marketing_coef_range(cls, v):
        if v is None:
            return v
        if v < 0 or v > 2:
            raise ValueError("marketing_coef must be between 0 and 2")
        return v

    @field_validator("hype_multiplier")
    @classmethod
    def hype_multiplier_range(cls, v):
        if v < 0.5 or v > 10:
            raise ValueError("hype_multiplier must be between 0.5 and 10")
        return v

    @field_validator("hype_level")
    @classmethod
    def hype_level_range(cls, v):
        if v < 1 or v > 10:
            raise ValueError("hype_level must be between 1 and 10")
        return v

    @field_validator("buyback_usd_per_month")
    @classmethod
    def buyback_non_negative(cls, v):
        if v is None:
            return v
        if v < 0:
            raise ValueError("buyback_usd_per_month must be >= 0")
        return v

    @field_validator("impact_eta", "price_floor_ratio")
    @classmethod
    def positive_values(cls, v, info):
        if v <= 0:
            raise ValueError(f"{info.field_name} must be > 0")
        return v

    @field_validator("impact_free_absorption")
    @classmethod
    def absorption_threshold_range(cls, v):
        if v < 0:
            raise ValueError("impact_free_absorption must be >= 0")
        if v > 1:
            raise ValueError("impact_free_absorption must be <= 1")
        return v

    @field_validator("launch_liquidity_usd")
    @classmethod
    def launch_liquidity_non_negative(cls, v):
        if v is None:
            return v
        if v < 0:
            raise ValueError("launch_liquidity_usd must be >= 0")
        return v

    @field_validator("liquidity_usd_per_1pct", "adv_daily_usd", "sigma_daily")
    @classmethod
    def optional_positive_market(cls, v, info):
        if v is None:
            return v
        if v <= 0:
            raise ValueError(f"{info.field_name} must be > 0")
        return v

    @field_validator("impact_beta")
    @classmethod
    def beta_range(cls, v):
        if v <= 0 or v > 2.0:
            raise ValueError("impact_beta must be in (0, 2]")
        return v

    @field_validator("impact_alpha_perm", "liquidity_tau_steps")
    @classmethod
    def positive_market_params(cls, v, info):
        if v <= 0:
            raise ValueError(f"{info.field_name} must be > 0")
        return v

    @field_validator("liquidity_shock_sigma")
    @classmethod
    def shock_sigma_range(cls, v):
        if v < 0 or v > 1.0:
            raise ValueError("liquidity_shock_sigma must be in [0, 1]")
        return v

    @field_validator("buyback_sink_mode")
    @classmethod
    def valid_sink_mode(cls, v):
        vv = str(v).lower()
        if vv not in {"none", "burn", "treasury"}:
            raise ValueError("buyback_sink_mode must be one of: none, burn, treasury")
        return vv

    @field_validator("buyback_exec_frac")
    @classmethod
    def exec_frac_range(cls, v):
        if v < 0 or v > 1:
            raise ValueError("buyback_exec_frac must be in [0, 1]")
        return v

    @field_validator("launch_liquidity_tge_tau_steps")
    @classmethod
    def tge_tau_positive(cls, v):
        if v <= 0:
            raise ValueError("launch_liquidity_tge_tau_steps must be > 0")
        return v

    @field_validator("l0_proxy_usd", "listing_units")
    @classmethod
    def optional_non_negative_proxy(cls, v, info):
        if v is None:
            return v
        if v < 0:
            raise ValueError(f"{info.field_name} must be >= 0")
        return v
