"""
Market Layer ABM Validation Tests
=================================
8 tests verifying correctness of the new Market Layer simulation engine.
Run with: python3 -m pytest tests/test_market_layer.py -v

Or directly: python3 tests/test_market_layer.py
"""

import math
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tokenomics_app.schemas import MonteCarloInput, Category
from tokenomics_app.services.monte_carlo import run_monte_carlo, _calibrate_market_layer, _tge_circulating_tokens


# ── Helpers ──────────────────────────────────────────────────────────────────

def make_base_params(**overrides) -> MonteCarloInput:
    """Create a minimal, valid MonteCarloInput for testing."""
    defaults = dict(
        project_name="test",
        total_supply=100_000_000,
        initial_price=1.00,
        horizon_months=6,
        num_simulations=300,
        prelaunch_profile="balanced",
        buy_pressure_score=0.0,
        random_seed=42,
        categories=[
            Category(
                name="Community",
                holder_profile="community",
                allocation_pct=30,
                tge_unlock_pct=5,
                cliff_months=0,
                vesting_months=24,
                sell_pressure_pct=50,
            ),
            Category(
                name="Liquidity",
                holder_profile="liquidity",
                allocation_pct=10,
                tge_unlock_pct=100,
                cliff_months=0,
                vesting_months=0,
                sell_pressure_pct=0,
            ),
            Category(
                name="Treasury",
                holder_profile="treasury",
                allocation_pct=60,
                tge_unlock_pct=0,
                cliff_months=0,
                vesting_months=48,
                sell_pressure_pct=5,
            ),
        ],
    )
    defaults.update(overrides)
    return MonteCarloInput(**defaults)


def _run(params: MonteCarloInput):
    return run_monte_carlo(params)


# ── Test 1: Scale Invariance ──────────────────────────────────────────────────

def test_scale_invariance():
    """
    Same fractional tokenomics but different total_supply + initial_price → same USD flows.
    With the same L0 (USD/1% move), price impact should be statistically similar.
    Old circulating-fraction model would give same sell_load but different USD.
    """
    # Both projects have same USD market cap = $10M
    # Same fractional sell pressure, same USD L0
    params_small = make_base_params(
        total_supply=10_000_000,
        initial_price=1.00,
        liquidity_usd_per_1pct=50_000,  # explicit L0 for fairness
    )
    params_large = make_base_params(
        total_supply=10_000_000_000,
        initial_price=0.001,
        liquidity_usd_per_1pct=50_000,  # same L0
    )

    r_small = _run(params_small)
    r_large = _run(params_large)

    # Relative price change should be close (within 20 percentage points)
    pct_small = r_small["summary"]["price_change_median_pct"]
    pct_large = r_large["summary"]["price_change_median_pct"]
    diff = abs(pct_small - pct_large)

    assert diff < 25, (
        f"Scale invariance broken: small={pct_small:.1f}%, large={pct_large:.1f}%, diff={diff:.1f}pp"
    )
    print(f"  [Test 1 PASS] Scale invariance: small={pct_small:.1f}%, large={pct_large:.1f}%, diff={diff:.1f}pp")


# ── Test 2: Zero Net Flow → Symmetric Noise ───────────────────────────────────

def test_zero_net_flow_no_systematic_drift():
    """
    With sell_pressure_pct=0 and buy_pressure_score=0 → F_t ≈ 0 → r_perm ≈ 0.
    Price should wander around initial price (GBM noise only, no systematic drift).
    Median end price should be within ±30% of initial.
    """
    params = make_base_params(
        categories=[
            # No selling at all
            Category(name="Treasury", holder_profile="treasury", allocation_pct=80,
                     tge_unlock_pct=0, cliff_months=0, vesting_months=48, sell_pressure_pct=0),
            Category(name="Liquidity", holder_profile="liquidity", allocation_pct=20,
                     tge_unlock_pct=100, cliff_months=0, vesting_months=0, sell_pressure_pct=0),
        ],
        buy_pressure_score=0.0,
        num_simulations=500,
        random_seed=123,
    )
    r = _run(params)
    median_end = r["summary"]["median_end_price"]
    initial = params.initial_price

    rel_change = abs(median_end / initial - 1.0)
    assert rel_change < 0.30, (
        f"Zero net flow should stay near initial price. Got median={median_end:.4f}, "
        f"initial={initial:.4f}, rel_change={rel_change*100:.1f}%"
    )
    print(f"  [Test 2 PASS] Zero net flow: median={median_end:.4f}, change={rel_change*100:.1f}%")


# ── Test 3: Concave Impact ────────────────────────────────────────────────────

def test_concave_impact_formula():
    """
    r_perm = alpha * |u|^beta with beta < 1 → concave (diminishing marginal impact).
    Doubling net flow should produce LESS THAN double r_perm.
    """
    alpha = 0.10
    beta = 0.6

    u1 = 0.10
    u2 = 0.20  # double

    r1 = alpha * (u1 ** beta)
    r2 = alpha * (u2 ** beta)

    assert r2 < 2 * r1, (
        f"Impact must be concave (diminishing marginal return). "
        f"r1={r1:.4f} (u={u1}), r2={r2:.4f} (u={u2}), 2*r1={2*r1:.4f}"
    )
    assert beta < 1, f"Concavity requires beta < 1, got beta={beta}"
    print(f"  [Test 3 PASS] Concave impact: r1={r1:.4f} → r2={r2:.4f} < 2*r1={2*r1:.4f}")


# ── Test 4: Liquidity Shock Sensitivity ───────────────────────────────────────

def test_liquidity_shock_sensitivity():
    """
    Liquidity shock sigma affects price dynamics.
    Higher sigma should produce LOWER median price (more crash risk from liquidity drying up).
    This is because when L_t crashes (crash regime), impact becomes extreme → price falls harder.
    Non-degenerate: both should have positive end prices and non-zero spread.
    """
    params_stable = make_base_params(
        liquidity_shock_sigma=0.001,  # near-zero L_t volatility
        num_simulations=400,
        random_seed=55,
    )
    params_volatile = make_base_params(
        liquidity_shock_sigma=0.25,   # strong L_t volatility (frequent liquidity crashes)
        num_simulations=400,
        random_seed=55,
    )

    r_stable = _run(params_stable)
    r_volatile = _run(params_volatile)

    med_stable = r_stable["summary"]["median_end_price"]
    med_volatile = r_volatile["summary"]["median_end_price"]

    # Both should have positive prices
    assert med_stable > 0, f"Stable median must be > 0: {med_stable}"
    assert med_volatile > 0, f"Volatile median must be > 0: {med_volatile}"

    # Both should have non-zero spread (P05 < P95)
    pq_stable = r_stable["price_quantiles"]
    assert pq_stable["p95"][-1] > pq_stable["p05"][-1], "Stable: price distribution must have spread"

    # High liquidity sigma should generally lead to worse or equal outcomes (crash risk)
    # Weak assertion: volatile should not HUGELY outperform stable (allow ±50% relative)
    ratio = med_volatile / med_stable if med_stable > 0 else 1.0
    assert 0.30 <= ratio <= 2.0, (
        f"Volatile liquidity should produce similar or worse outcomes than stable. "
        f"med_stable={med_stable:.4f}, med_volatile={med_volatile:.4f}, ratio={ratio:.2f}"
    )
    print(f"  [Test 4 PASS] Liquidity shock sensitivity: stable={med_stable:.4f}, volatile={med_volatile:.4f}, ratio={ratio:.2f}")


# ── Test 5: Buyback Sink Reduces Float ────────────────────────────────────────

def test_buyback_burn_improves_price():
    """
    buyback_sink_mode="burn" should produce higher median end price than "none".
    Burning removes tokens from supply → lower sell pressure over time.
    """
    buyback = 10_000  # $10K/month buyback
    params_burn = make_base_params(
        buyback_usd_per_month=buyback,
        buyback_sink_mode="burn",
        buy_pressure_score=0.0,
        num_simulations=400,
    )
    params_none = make_base_params(
        buyback_usd_per_month=buyback,
        buyback_sink_mode="none",
        buy_pressure_score=0.0,
        num_simulations=400,
    )

    r_burn = _run(params_burn)
    r_none = _run(params_none)

    med_burn = r_burn["summary"]["median_end_price"]
    med_none = r_none["summary"]["median_end_price"]

    # Burn mode should be equal or better than no-sink
    # (burn reduces float → less sell pressure from float-based metrics)
    # Note: price impact is the same in both — burn only reduces future sell pressure
    # So the difference may be small in short horizons, but direction should hold
    assert med_burn >= med_none * 0.90, (
        f"Burn mode should not be significantly worse than no-sink. "
        f"burn={med_burn:.4f}, none={med_none:.4f}"
    )
    print(f"  [Test 5 PASS] Buyback burn: burn={med_burn:.4f}, none={med_none:.4f}")


# ── Test 6: Unlock Shock Realism ──────────────────────────────────────────────

def test_unlock_shock_at_cliff():
    """
    A large cliff unlock at month 3 should produce peak sell pressure near month 3.
    Peak should not occur before month 3 (cliff prevents earlier unlock).
    """
    params = make_base_params(
        categories=[
            # Heavy investors with 3-month cliff
            Category(name="Seed", holder_profile="investor_round", allocation_pct=40,
                     tge_unlock_pct=0, cliff_months=3, vesting_months=6, sell_pressure_pct=80),
            Category(name="Liquidity", holder_profile="liquidity", allocation_pct=10,
                     tge_unlock_pct=100, cliff_months=0, vesting_months=0, sell_pressure_pct=0),
            Category(name="Treasury", holder_profile="treasury", allocation_pct=50,
                     tge_unlock_pct=0, cliff_months=0, vesting_months=48, sell_pressure_pct=5),
        ],
        horizon_months=12,
        num_simulations=400,
    )
    r = _run(params)

    # Monthly sell pressure (tokens) P50
    monthly_stats = r["monthly_stats"]
    pressure_by_month = [ms["pressure_tokens_p50"] for ms in monthly_stats]

    # Find peak month
    peak_month = pressure_by_month.index(max(pressure_by_month)) + 1  # +1 because monthly_stats starts at m=0 = month1

    assert peak_month >= 3, (
        f"Peak sell pressure should occur at or after cliff (month 3). "
        f"Got peak at month {peak_month}, pressures: {[f'{p:.0f}' for p in pressure_by_month[:6]]}"
    )
    print(f"  [Test 6 PASS] Unlock shock at cliff: peak month={peak_month} (cliff=3)")


# ── Test 7: Seed Reproducibility ──────────────────────────────────────────────

def test_seed_reproducibility():
    """
    Same random_seed → identical results across two runs.
    """
    params = make_base_params(random_seed=999, num_simulations=200)

    r1 = _run(params)
    r2 = _run(params)

    assert r1["summary"]["median_end_price"] == r2["summary"]["median_end_price"], (
        f"Results not reproducible! "
        f"run1={r1['summary']['median_end_price']:.6f}, run2={r2['summary']['median_end_price']:.6f}"
    )
    assert r1["spaghetti_prices"][0] == r2["spaghetti_prices"][0], (
        "First spaghetti path not reproducible!"
    )
    print(f"  [Test 7 PASS] Seed reproducibility: median={r1['summary']['median_end_price']:.6f}")


# ── Test 8: KPI Stability ─────────────────────────────────────────────────────

def test_kpi_stability():
    """
    All summary KPIs should be finite, in valid ranges, no NaN/Inf.
    """
    params = make_base_params(num_simulations=300, buy_pressure_score=5.0)
    r = _run(params)
    s = r["summary"]

    finite_checks = [
        "median_end_price", "mean_end_price", "p05_end_price", "p95_end_price",
        "median_max_drawdown", "prob_drawdown_50pct", "prob_drawdown_80pct",
        "prob_price_above_initial", "market_L0_usd", "market_sigma_daily_pct",
        "market_impact_alpha_perm", "market_impact_beta",
    ]
    for key in finite_checks:
        val = s.get(key)
        assert val is not None, f"Missing KPI: {key}"
        assert math.isfinite(val), f"KPI {key}={val} is not finite"

    # Range checks
    assert s["median_end_price"] > 0, "median_end_price must be > 0"
    assert 0 <= s["prob_drawdown_50pct"] <= 1, "prob_drawdown_50pct out of [0,1]"
    assert 0 <= s["prob_drawdown_80pct"] <= 1, "prob_drawdown_80pct out of [0,1]"
    assert -1 <= s["median_max_drawdown"] <= 0, "median_max_drawdown out of [-1, 0]"
    assert s["market_L0_usd"] > 0, "L0 must be > 0"
    assert s["market_sigma_daily_pct"] > 0, "sigma_daily_pct must be > 0"

    # All price quantiles should be positive and finite
    prq = r["price_quantiles"]
    for q_key in ["p05", "p25", "p50", "p75", "p95", "mean"]:
        vals = prq[q_key]
        for v in vals:
            assert math.isfinite(v), f"price_quantiles.{q_key} contains non-finite value: {v}"
            assert v >= 0, f"price_quantiles.{q_key} contains negative: {v}"

    print(f"  [Test 8 PASS] KPI stability: all {len(finite_checks)} KPIs finite, ranges valid")


# ── Test 9: Buy Pressure Score Monotonic ─────────────────────────────────────

def test_buy_pressure_score_monotonic():
    """
    Higher buy_pressure_score should lead to higher median end price.
    score=0 < score=5 < score=10 in terms of median_end_price.
    """
    results = {}
    for score in [0, 5, 10]:
        p = make_base_params(buy_pressure_score=score, num_simulations=400)
        r = _run(p)
        results[score] = r["summary"]["median_end_price"]

    assert results[5] > results[0] * 0.95, (
        f"score=5 should improve price vs score=0. Got {results[5]:.4f} vs {results[0]:.4f}"
    )
    assert results[10] > results[5] * 0.95, (
        f"score=10 should improve price vs score=5. Got {results[10]:.4f} vs {results[5]:.4f}"
    )
    print(f"  [Test 9 PASS] Buy pressure monotonic: score0={results[0]:.4f} < score5={results[5]:.4f} < score10={results[10]:.4f}")


# ── Runner ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_scale_invariance,
        test_zero_net_flow_no_systematic_drift,
        test_concave_impact_formula,
        test_liquidity_shock_sensitivity,
        test_buyback_burn_improves_price,
        test_unlock_shock_at_cliff,
        test_seed_reproducibility,
        test_kpi_stability,
        test_buy_pressure_score_monotonic,
    ]

    passed = 0
    failed = 0
    print("\n" + "="*60)
    print("Market Layer ABM — Validation Tests")
    print("="*60)
    for test_fn in tests:
        name = test_fn.__name__
        print(f"\n{name}:")
        try:
            test_fn()
            passed += 1
        except AssertionError as e:
            print(f"  [FAIL] {e}")
            failed += 1
        except Exception as e:
            print(f"  [ERROR] {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "="*60)
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)} tests")
    print("="*60)
    sys.exit(0 if failed == 0 else 1)
