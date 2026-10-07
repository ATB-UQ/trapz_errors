# trapz_errors

Error analysis for the trapezoidal rule applied to uncertain data: the integral's uncertainty from
the y-value errors, plus an estimate of the truncation error from local curvature, and a guide to
where extra points would reduce the error most. Used by `solvation_fe_ti` for thermodynamic
integration (dV/dlambda -> free energy). Status: live-support library, stable.

Author: Martin Stroet (University of Queensland). Method: `method_summary.pdf`. Licence: `LICENSE`.

## Install

Python 3, `numpy`, `scipy`, `matplotlib`, `click` (see `setup.cfg`). In the platform it is installed
editable in `/home/atb/ATB/.venv`; elsewhere `pip install -e .`. Nothing here needs env vars,
hosts or vendor binaries.

## Python API (what the platform uses)

`trapz_errors.calculate_error`
- `trapz_integrate_with_uncertainty(xs, ys, es, be_conservative=True)` - legacy integral and error.
- `trapz_integrate_decomposed(xs, ys, es, noise_sigmas=2.)` - error kept as separate quantities:
  `sigma_points` (1 sigma from point errors), signed/absolute truncation sums, each interval's own
  uncertainty and a `noise_dominated` flag (added 2026-09). The legacy function's numbers are unchanged.
- `point_error_calc`, `interval_errors`, `interval_errors_with_uncertainty`, `plot_error_analysis`.

`trapz_errors.reduce_error`: `get_updates(...)` and `reduce_error_on_residual_error(...)` - which
intervals to refine/extend to reach a target error. `trapz_errors.helpers`: `rss`, `round_sigfigs`,
`parse_user_data`, `second_derivative_with_uncertainty`.

## Command line

No console scripts are installed; run the modules. Input: whitespace-separated `x y [y_error]` lines.

```bash
python -m trapz_errors.calculate_error -d test/eg_data.dat -c -v -p out.png   # integral +/- error
python -m trapz_errors.reduce_error    -d test/eg_data.dat -t 1.0 -c          # where to add points
```
Options: `-p [PLOT]` save a figure, `-s SIGFIGS` (default 3), `-v` breakdown, `-c` conservative
truncation estimate (adds the largest interval error), `-t TARGET_ERROR`, `-r CONVERGENCE_RATE_SCALING`
(< 1: more iterations, fewer points; > 1: the reverse). `example/example.sh` shows both (its
relative paths predate the package layout; use the commands above).

There is also a self-contained `click` CLI, `python -m trapz_errors.trapz_errors`
(`calculate-error`, `reduce-error`, `simple-reduce-error`, `plot`, `run-tests`); it takes `.dat`
files with `--target` in kJ/mol and duplicates the argparse tools above.

## Tests

```bash
cd test && /home/atb/ATB/.venv/bin/python -m pytest test_error_propagation.py   # 10 pass, <1 s
```
`test/basic_tests.py` is a script of plots/comparisons, not an assertion suite. Untracked `test/*.png`
are plot outputs of those scripts.
