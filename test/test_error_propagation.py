"""Assertion tests for trapz_errors' uncertainty propagation.

Step 1 of solvation_fe_ti/docs/uncertainty_methods_evaluation_plan.md: pin what
the existing code does before any reducer is changed, and test the decomposed
integrator against quantities with known answers. No plotting, no files.

    /home/atb/ATB/.venv/bin/python -m pytest test/test_error_propagation.py
"""
import numpy as np
import pytest

from trapz_errors.calculate_error import (interval_errors, interval_errors_with_uncertainty, point_error_calc,
                                          trapz_integrate_decomposed, trapz_integrate_with_uncertainty)
from trapz_errors.helpers import rss
from trapz_errors.reduce_error import get_updates

# A grid of the shape the TI pipeline produces: 14 initial points, refined
# unevenly where the curve bends.
GRID = np.array([0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.825, 0.85, 0.9, 1.0])


def test_point_weights_are_trapezoid_weights():
    """Each point's weight is half the width of its two neighbouring intervals,
    so the weights sum to the integration range."""
    weights = np.array(point_error_calc(GRID, np.ones_like(GRID)))
    expected = np.empty_like(GRID)
    expected[0] = (GRID[1] - GRID[0]) / 2
    expected[-1] = (GRID[-1] - GRID[-2]) / 2
    expected[1:-1] = (GRID[2:] - GRID[:-2]) / 2
    assert np.allclose(weights, expected)
    assert np.isclose(weights.sum(), GRID[-1] - GRID[0])


def test_point_error_propagation_matches_monte_carlo():
    """RSS of the weighted point errors is the SD of the trapezoid integral when
    the points carry independent normal noise of those SDs, on a non-uniform grid."""
    rng = np.random.default_rng(20260923)
    ys = 50 * np.sin(3 * GRID)
    es = rng.uniform(0.2, 3.0, size=GRID.shape)
    predicted = rss(point_error_calc(GRID, es))
    n = 40000
    samples = np.trapezoid(ys + rng.normal(size=(n, GRID.size)) * es, GRID, axis=1)
    # SD of an SD estimate from n normal samples is sigma / sqrt(2n): 0.35% here
    assert np.std(samples, ddof=1) == pytest.approx(predicted, rel=0.02)
    assert np.mean(samples) == pytest.approx(np.trapezoid(ys, GRID), abs=4 * predicted / np.sqrt(n))


def test_quadratic_truncation_error_is_exact():
    """For a quadratic the three-point second derivative is exact, so the summed
    interval errors equal the trapezoid rule's actual error, sign included
    (positive: the trapezoid overestimates a convex function)."""
    ys = 7 * GRID ** 2 - 3 * GRID + 1
    exact = 7 / 3 - 3 / 2 + 1
    actual = np.trapezoid(ys, GRID) - exact
    d = trapz_integrate_decomposed(GRID, ys, np.zeros_like(GRID))
    assert d["quad_signed"] == pytest.approx(actual, rel=1e-9)
    assert d["quad_abs"] == pytest.approx(actual, rel=1e-9)
    assert d["sigma_points"] == 0.0


def test_opposite_curvatures_cancel_in_the_signed_sum_only():
    """Defect 3 of the plan: signed interval errors cancel. quad_abs does not."""
    xs = np.linspace(0, 1, 11)
    ys = 100 * np.sin(2 * np.pi * xs)
    d = trapz_integrate_decomposed(xs, ys, np.zeros_like(xs))
    # not exactly zero: the end intervals borrow their neighbour's stencil
    assert abs(d["quad_signed"]) < 0.1 * d["quad_abs"]
    assert d["quad_abs"] > 1.0


@pytest.mark.parametrize("forward", [True, False])
def test_curvature_sigma_matches_monte_carlo(forward):
    """The sigma kept for each interval's truncation estimate is the SD of that
    estimate under point noise, for a fixed difference direction.

    Fixed, because the integrators pick forward or backward by whichever has
    the larger |signed sum|, and under noise that choice flips from draw to
    draw -- so the direction is itself chosen by the noise, which is one more
    way the legacy curvature term chases it."""
    rng = np.random.default_rng(7)
    ys = 30 * GRID ** 3
    es = np.full_like(GRID, 1.5)
    _, _, sigmas = interval_errors_with_uncertainty(GRID, ys, es, forward=forward)
    draws = np.array([interval_errors_with_uncertainty(GRID, ys + rng.normal(size=GRID.shape) * es, es,
                                                       forward=forward)[1]
                      for _ in range(3000)])
    assert np.allclose(draws.std(axis=0, ddof=1), sigmas, rtol=0.08)


def test_noise_dominated_flags_flat_noisy_curves():
    """A straight line has no curvature, so every interval estimate on noisy
    points of one is noise. Not every one is flagged at 2 sigma: the direction
    chosen is the one with the larger |sum|, which selects for large errors."""
    rng = np.random.default_rng(3)
    es = np.full_like(GRID, 2.0)
    ys = 10 * GRID + rng.normal(size=GRID.shape) * es
    d = trapz_integrate_decomposed(GRID, ys, es)
    assert np.mean(d["noise_dominated"]) >= 0.7


def test_decomposed_agrees_with_legacy_intervals():
    """The new function's signed interval errors are the legacy ones; it only
    adds their sigmas. And its legacy_total is the legacy total."""
    rng = np.random.default_rng(11)
    ys = -60 * np.cos(4 * GRID) + rng.normal(size=GRID.shape)
    es = rng.uniform(0.3, 2.0, size=GRID.shape)
    d = trapz_integrate_decomposed(GRID, ys, es)
    fwd = interval_errors(GRID, ys, es, forward=True)
    bwd = interval_errors(GRID, ys, es, forward=False)
    legacy_gaps = max([fwd, bwd], key=lambda r: abs(np.sum(r[-1])))[-1]
    assert np.allclose(d["gap_errors"], legacy_gaps)
    assert np.allclose(d["gap_xs"], max([fwd, bwd], key=lambda r: abs(np.sum(r[-1])))[0])
    _, total, *_ = trapz_integrate_with_uncertainty(GRID, ys, es, be_conservative=True)
    assert d["legacy_total"] == total


def test_legacy_total_is_linear_sum_of_sigma_and_bias_bound():
    """Pins the legacy composition (plan §0): RSS of point errors plus, linearly,
    |signed sum without the largest interval| + the largest |interval| over both
    directions. A change here changes every reported error."""
    rng = np.random.default_rng(5)
    ys = 80 * np.sin(5 * GRID) + rng.normal(size=GRID.shape)
    es = rng.uniform(0.3, 2.0, size=GRID.shape)
    _, total, _, _, gaps, point_errors, max_interval = trapz_integrate_with_uncertainty(GRID, ys, es, True)
    rest = sorted(gaps, key=abs)[:-1]
    assert total == pytest.approx(rss(point_errors) + abs(np.sum(rest)) + max_interval)


def test_get_updates_credits_points_with_the_interval_factor():
    """Pins defect 4 of the plan: get_updates credits a re-run point with 75% of
    its error, the midpoint-insertion factor for an interval. One extension
    actually reduces a point's sigma by 1 - 1/sqrt(2), about 29%. When this test
    fails, the scheduling rule has been changed on purpose; update the plan."""
    xs = [0.0, 0.5, 1.0]
    point_errors = [0.1, 1.0, 0.1]
    # residual 1.5 - 1.0 = 0.5: one point credited 0.75 covers it, so exactly
    # one item is selected. Credited at 0.29 it would take two.
    new_pts, update_xs = get_updates(xs, point_errors, [0.25, 0.75], [0.01, 0.01], 1.5, 1.0, 1)
    assert new_pts == [] and len(update_xs) == 1 and update_xs[0][1] == 0.5
