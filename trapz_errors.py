#!/usr/bin/env python3

r"""
trapz_errors
============


Self-contained script to calculate, plot, and determine points
to reduce error using the trapezoidal rule.

Example usage

    ./trapz_errors.py calculate-error test/eg_data.dat --conservative --verbose
    Error from y-value uncertainty: ± 0.564
    Estimated total truncation error: 0.310
    Integral: 21.330 ± 1.539

To list other commands you can use:

    ./trapz_errors.py --help
    Usage: trapz_errors.py [OPTIONS] COMMAND [ARGS]...

    Options:
    --help  Show this message and exit.

    Commands:
    calculate-error      Calculate the integral and error from FILENAME.
    plot                 Plot the dhdl over x from FILENAME.
    reduce-error         Determine points to add or extend to meet target...
    run-tests            Run tests
    simple-reduce-error  Determine points to add (first) or extend (second)...

To run tests:

    ./trapz_errors.py run-tests
    🐑 All tests passed! 🐑

"""


import subprocess
import textwrap
from typing import Union, Tuple, List

try:
    import click
except ImportError:
    raise ImportError("Please install click with `conda install -c conda-forge click`")

try:
    import numpy as np
except ImportError:
    raise ImportError("Please install numpy with `conda install -c conda-forge numpy`")


def calculate_point_errors(xs: np.ndarray, es: np.ndarray) -> np.ndarray:
    xs = np.asarray(xs).reshape((-1,))
    es = np.asarray(es).reshape((-1,))

    padded_xs = np.r_[[xs[0]], xs, [xs[-1]]]
    diffs = padded_xs[2:] - padded_xs[:-2]
    return es * diffs / 2


def calculate_y_value_uncertainty(points: np.ndarray) -> float:
    point_errors = calculate_point_errors(points[:, 0], points[:, 2])
    return (point_errors ** 2).sum() ** 0.5


def calculate_y_intercept(
    point1: np.ndarray,
    point2: np.ndarray,
    x: Union[float, np.ndarray],
) -> int:
    x1, y1 = np.asarray(point1)[..., :2].T
    x2, y2 = np.asarray(point2)[..., :2].T

    gradient = (y2 - y1) / (x2 - x1)
    intercept = (x1 * y2 - x2 * y1) / (x1 - x2)
    return gradient * x + intercept


def calculate_trapezoidal_error(
    points: np.ndarray, dx: Union[float, np.ndarray]
) -> np.ndarray:
    h0 = points[1:-1, 0] - points[:-2, 0]
    h1 = points[2:, 0] - points[1:-1, 0]
    h0h1 = h0 * h1
    a = 2 / (h0h1 + (h0 * h0))
    b = -2 / h0h1
    c = 2 / (h0h1 + (h1 * h1))
    coefficients = np.array([a, b, c])
    y = points[:, 1]
    ys = np.array(list(zip(y[:-2], y[1:-1], y[2:]))).T
    product = np.einsum("ij,ij->j", coefficients, ys)
    error = product * (dx ** 3) / 12
    return error


def calculate_interval_points(points: np.ndarray):
    xs = points[:, 0]
    interval_xs = (xs[:-1] + xs[1:]) / 2  # dim: N-1, 3
    interval_ys = calculate_y_intercept(
        points[:-1], points[1:], interval_xs
    )  # dim: N-1, 3
    return np.array([interval_xs, interval_ys]).T


def calculate_integral(points: np.ndarray):
    return np.trapz(points[:, 1], points[:, 0])


def calculate_interval_errors(points: np.ndarray) -> Tuple[np.ndarray, float]:
    xs = points[:, 0]
    diff_xs = xs[1:] - xs[:-1]  # dim: N-1, 3

    initial_error = calculate_trapezoidal_error(
        points[:3], diff_xs[0])  # dim: 1, 3
    # dim: 1, 3
    last_error = calculate_trapezoidal_error(points[-3:], diff_xs[-1])

    mid_x = diff_xs[1:-1]
    forward_middle_errors = calculate_trapezoidal_error(points[:-1], mid_x)
    backward_middle_errors = calculate_trapezoidal_error(points[1:], mid_x)

    back_e = np.abs(np.sum(backward_middle_errors))
    forward_e = np.abs(np.sum(forward_middle_errors))
    if back_e > forward_e:
        middle_errors = backward_middle_errors
    else:
        middle_errors = forward_middle_errors

    interval_es = np.concatenate([initial_error, middle_errors, last_error])

    all_errors = np.concatenate(
        [
            initial_error,
            backward_middle_errors,
            forward_middle_errors,
            last_error,
        ]
    )
    max_error = np.max(np.abs(all_errors))
    return interval_es, max_error


def calculate_total_error(points: np.ndarray, conservative: bool = False):
    y_uncertainty = calculate_y_value_uncertainty(points)
    interval_errors, max_interval_error = calculate_interval_errors(points)
    if conservative:
        max_excluded = sorted(interval_errors)[:-1]
        error_offset = np.abs(np.sum(max_excluded)) + max_interval_error
    else:
        error_offset = np.abs(np.sum(interval_errors))
    return y_uncertainty + error_offset


def print_integral(points: np.ndarray, precision: int = 3, conservative: bool = False):
    integral = calculate_integral(points)
    total_error = calculate_total_error(points, conservative=conservative)
    print(f"Integral: {integral:.{precision}f} ± {total_error:.{precision}f}")


def print_errors(points: np.ndarray, precision: int = 3, conservative: bool = False):
    y_uncertainty = calculate_y_value_uncertainty(points)
    print(f"Error from y-value uncertainty: ± {y_uncertainty:.{precision}f}")
    integral = calculate_integral(points)
    interval_errors, _ = calculate_interval_errors(points)
    total_truncation_error = np.abs(np.sum(interval_errors))
    print(
        f"Estimated total truncation error: {total_truncation_error:.{precision}f}")
    total_error = calculate_total_error(points, conservative=conservative)
    print(f"Integral: {integral:.{precision}f} ± {total_error:.{precision}f}")


def get_largest_error_points(
    points_to_assess: np.ndarray,
    residual_error: float,
    largest_absolute_error: float,
    convergence_rate_scale_factor: float = 1,
    conservative: bool = False,
) -> np.ndarray:
    points = sorted(points_to_assess, reverse=True, key=lambda x: x[-1])
    largest_error_points = []

    scale_factor = 0.75 / convergence_rate_scale_factor
    while residual_error >= 0 and points:
        point = points.pop(0)
        largest_error_points.append(point)
        abspoint = abs(point)
        if conservative and np.allclose(point[-1], largest_absolute_error):
            residual_error -= abspoint[-1]
        else:
            # COMMENT COPIED FROM Martin Stroet, February 5th, 2018 1:36pm ||  92df53f
            # Since addition of a point to an interval reduces error by a factor of 1/4 according
            # to the truncation error estimate method we're using => (e - e/4) = 0.75*e.
            # We simply assume the same is true for reducing the uncertainty on an existing point.
            # The convergence rate scaling factor allows for additional control over the number of points
            # that will be added on each iteration.
            residual_error -= scale_factor * abspoint[-1]

    return np.array(largest_error_points)


def get_points_to_add_and_extend(
    points: np.ndarray,
    residual_error: float,
    conservative: bool = False,
    convergence_rate_scale_factor: float = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    interval_points = calculate_interval_points(points)
    interval_errors, _ = calculate_interval_errors(points)
    interval_points = np.concatenate([interval_points.T, [interval_errors]]).T

    points_with_error = np.array(points)
    points_with_error[:, 2] = calculate_point_errors(
        points[:, 0], points[:, 2])
    points_to_assess = np.concatenate([points_with_error, interval_points])
    # largest_absolute_error = np.max(np.abs(interval_points[:, 2]))
    largest_absolute_error = np.abs(interval_points[0])[2]
    largest_error_points = get_largest_error_points(
        points_to_assess,
        residual_error,
        largest_absolute_error,
        conservative=conservative,
        convergence_rate_scale_factor=convergence_rate_scale_factor,
    )[:, 0]
    is_interval = np.isin(
        largest_error_points, interval_points[:, 0], assume_unique=True
    )
    return largest_error_points[is_interval], largest_error_points[~is_interval]


def get_points_to_reduce_error(
    points: np.ndarray,
    target_error: float,
    conservative: bool = False,
    convergence_rate_scale_factor: float = 1,
) -> Tuple[List[float], List[float]]:
    total_error = abs(calculate_total_error(points, conservative=conservative))
    target_error = abs(target_error)
    if total_error > target_error:
        residue_error = total_error - target_error
        return get_points_to_add_and_extend(
            points,
            residue_error,
            conservative=conservative,
            convergence_rate_scale_factor=convergence_rate_scale_factor,
        )
    return [], []


def print_reduce(
    points: np.ndarray,
    target_error: float,
    conservative: bool = False,
    convergence_rate_scale_factor: float = 1,
    precision: int = 3,
):
    total_error = abs(calculate_total_error(points, conservative=conservative))
    target_error = abs(target_error)
    if total_error > target_error:
        residual_error = abs(total_error) - abs(target_error)
        print(f"Residual error: {residual_error:.{precision}f}")
        new, extend = get_points_to_add_and_extend(
            points,
            residual_error,
            conservative=conservative,
            convergence_rate_scale_factor=convergence_rate_scale_factor,
        )
        new_formatted = [f"{x:.{precision}f}" for x in new]
        extend_formatted = [f"{x:.{precision}f}" for x in extend]
        print("Suggested new points:")
        print(", ".join(new_formatted))
        print("Suggested to reduce uncertainty in existing points:")
        print(", ".join(extend_formatted))
    else:
        print("Target error has been reached.")


def plot_error_analysis(points: np.ndarray, height: float = 6, width: float = 8,
                        xlabel: str = "x", ylabel="y", title="", filename: str = "",
                        show: bool = False
                        ):
    from matplotlib import pyplot as plt
    _, ax = plt.subplots(figsize=(width, height))
    ax.errorbar(*points.T, marker="o")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()

    if filename:
        if not filename.endswith(".png"):
            filename += ".png"
        plt.savefig(filename, format="png", dpi=300)
        print(f"Saved plot to {filename}")
    if show:
        plt.show()


@click.group()
def cli():
    pass


@cli.command()
@click.argument("filename")
@click.option("--conservative", is_flag=True, help="guess error conservatively")
@click.option("--precision", default=3, help="the precision to print results at")
@click.option("--verbose", is_flag=True)
def calculate_error(filename, conservative, precision, verbose):
    """Calculate the integral and error from FILENAME.

    FILENAME is the data file containing lambda, dhdl_average, and error_est.
    """
    points = np.loadtxt(filename)
    if verbose:
        print_errors(points, precision=precision, conservative=conservative)
    else:
        print_integral(points, precision=precision, conservative=conservative)


@cli.command()
@click.argument("filename")
@click.option("--target", default=1.5, help="target error in kJ/mol")
@click.option("--convergence_rate_scale_factor", default=1, help="scale factor for convergence rate in residual error")
@click.option("--conservative", is_flag=True, help="guess error conservatively")
@click.option("--precision", default=3, help="the precision to print results at")
@click.option("--verbose", is_flag=True)
def reduce_error(
    filename,
    conservative,
    precision,
    verbose,
    target,
    convergence_rate_scale_factor,
):
    """Determine points to add or extend to meet target error.

    FILENAME is the data file containing lambda, dhdl_average, and error_est.
    """
    points = np.loadtxt(filename)
    if verbose:
        print_errors(points, precision=precision, conservative=conservative)
    print_reduce(
        points,
        target_error=target,
        conservative=conservative,
        convergence_rate_scale_factor=convergence_rate_scale_factor,
        precision=precision,
    )


@cli.command()
@click.argument("filename")
@click.option("--target", default=1.5, help="target error in kJ/mol")
@click.option("--convergence_rate_scale_factor", default=1, help="scale factor for convergence rate in residual error")
@click.option("--conservative", is_flag=True, help="guess error conservatively")
@click.option("--precision", default=3, help="the precision to print results at")
def simple_reduce_error(
    filename,
    conservative,
    precision,
    target_error,
    convergence_rate_scale_factor,
):
    """Determine points to add (first) or extend (second) to meet target error.

    FILENAME is the data file containing lambda, dhdl_average, and error_est.
    """
    points = np.loadtxt(filename)
    with np.set_printoptions(precision=precision):
        return get_points_to_reduce_error(
            points,
            target_error,
            conservative,
            convergence_rate_scale_factor,
        )


@cli.command()
@click.argument("filename")
@click.option("--png", default="", help="Filename to save plot to. If not given, plot is not saved")
@click.option("--height", default=6, help="height of plot in inches")
@click.option("--width", default=8, help="width of plot in inches")
@click.option("--xlabel", default="x")
@click.option("--ylabel", default="y")
@click.option("--title", default="")
@click.option("--show", is_flag=True)
def plot(filename, png, height, width, xlabel, ylabel, title, show):
    """Plot the dhdl over x from FILENAME.

    FILENAME is the data file containing lambda, dhdl_average, and error_est.
    """
    points = np.loadtxt(filename)
    plot_error_analysis(points, height=height, width=width,
                        xlabel=xlabel, ylabel=ylabel, title=title,
                        filename=png, show=show)


def test_simple_calculate_error(filename):
    proc = subprocess.run(
        [__file__, "calculate-error", filename],
        capture_output=True,
        text=True,
    )
    assert proc.stdout == "Integral: 21.330 ± 0.873\n", proc.stdout


def test_conservative_verbose_calculate_error(filename):
    proc = subprocess.run(
        [__file__, "calculate-error", filename, "--conservative", "--verbose"],
        capture_output=True,
        text=True,
    )
    expected = textwrap.dedent("""\
    Error from y-value uncertainty: ± 0.564
    Estimated total truncation error: 0.310
    Integral: 21.330 ± 1.539
    """)
    assert proc.stdout == expected, proc.stdout


def test_conservative_verbose_calculate_error_precision(filename):
    proc = subprocess.run(
        [__file__, "calculate-error", filename,
            "--conservative", "--verbose", "--precision", "2"],
        capture_output=True,
        text=True,
    )
    expected = textwrap.dedent("""\
    Error from y-value uncertainty: ± 0.56
    Estimated total truncation error: 0.31
    Integral: 21.33 ± 1.54
    """)
    assert proc.stdout == expected, proc.stdout


def test_reduce_error_conservative_verbose(filename):
    proc = subprocess.run(
        [__file__, "reduce-error", filename,
            "--conservative", "--verbose", "--target", "1", "--convergence_rate_scale_factor", "1"],
        capture_output=True,
        text=True,
    )
    expected = textwrap.dedent("""\
    Error from y-value uncertainty: ± 0.564
    Estimated total truncation error: 0.310
    Integral: 21.330 ± 1.539
    Residual error: 0.539
    Suggested new points:
    0.812
    Suggested to reduce uncertainty in existing points:
    0.850
    """)
    assert proc.stdout == expected, proc.stdout


@cli.command()
def run_tests():
    """Run tests"""
    import os
    import tempfile

    try:
        import tqdm
    except ImportError:
        class tqdm:
            def tqdm(self, iterable):
                return iterable
        tqdm = tqdm()

    EXAMPLE_DATA = textwrap.dedent("""\
    # lambda  dvdl_average  error_est
    0.000 99.84849 0.36028
    0.025 155.28053 0.90581
    0.050 186.13209 1.11968
    0.075 188.50786 1.42872
    0.100 177.67623 1.45245
    0.125 149.29522 1.37675
    0.150 125.41653 1.41022
    0.200 83.82993 0.78686
    0.250 63.47751 0.54568
    0.300 50.74429 0.35037
    0.400 32.61911 0.41166
    0.500 17.44440 0.62932
    0.600 -2.15131 1.13114
    0.649 -16.64929 1.29989
    0.700 -32.44893 1.58581
    0.750 -71.44813 2.10821
    0.775 -94.12352 5.47328
    0.850 -113.45883 8.90801
    0.875 -97.05920 1.65601
    0.900 -71.38330 0.95010
    0.950 -30.13460 0.54041
    1.000 3.15286 0.21252
    """)

    with tempfile.TemporaryDirectory() as tmp:
        datafile = os.path.join(tmp, "data.dat")

        with open(datafile, "w") as f:
            f.write(EXAMPLE_DATA)

        functions = [
            test_simple_calculate_error,
            test_conservative_verbose_calculate_error,
            test_conservative_verbose_calculate_error_precision,
            test_reduce_error_conservative_verbose
        ]
        for func in tqdm.tqdm(functions):
            func(datafile)
        print("🐑 All tests passed! 🐑")


if __name__ == "__main__":
    cli()
