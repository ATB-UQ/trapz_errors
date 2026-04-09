import numpy as np
from scipy.integrate import trapezoid
from scipy import interpolate

import matplotlib.pyplot as plt

from trapz_errors.reduce_error import reduce_error_on_residual_error, run as run_reduce_error
from trapz_errors.calculate_error import trapz_integrate_with_uncertainty, point_error_calc
from trapz_errors.helpers import rss, parse_user_data


def integrate_with_point_uncertinaty(xs, ys, es):
    integration_error_per_point = point_error_calc(xs, es)
    return trapezoid(ys, xs), rss(integration_error_per_point)


def run_test(xs, ys, es, x_fine=None, y_fine=None):
    if x_fine is not None:
        integral = trapezoid(y_fine, x=x_fine)
        print("Integral: {0}".format(integral))
        print("True truncation error: {0}".format(trapezoid(ys, x=xs) - integral))

    trapz_integral, total_error, gap_xs, _, gap_errors, _, _ = trapz_integrate_with_uncertainty(xs, ys, es, be_conservative=True)
    #print "Truncation error estimate: {0} +/- {1}".format(trapz_integral, total_error - rss(integration_point_errors))
    print("Combined error estimate: {0} +/- {1}".format(trapz_integral, total_error))

    plot_data(xs, ys, es, x_fine, y_fine,
        figure_name = "example_integration_{0}.png".format(len(xs)),
        )
    return gap_errors, gap_xs, total_error


def plot_data(xs, ys, es, x_fine=None, y_fine=None, figure_name=None):
    fig = plt.figure(figsize=(5,4))
    ax = fig.add_subplot(111)
    f = interpolate.interp1d(xs, ys, kind=1)
    if x_fine is not None and y_fine is not None:
        ax.fill_between(x_fine, y_fine, list(map(f, x_fine)), facecolor='red', alpha=0.8)
        ax.plot(x_fine, y_fine, "r")
    ax.set_title("N={0}".format(len(xs)), fontweight="bold")
    ax.errorbar(xs, ys, es, marker="o", label="Integration Points")
    ax.set_xlabel(r'$\mathbf{\lambda}$')
    ax.set_ylabel(r'<$\mathbf{dV/d\lambda}$> (kJ/mol)')
    #ax.set_title("Measurement Error Propagation", fontweight="bold")

    #plt.legend(loc = 'upper right', prop={'size':11}, numpoints = 1, frameon = False)
    fig.tight_layout()
    if figure_name:
        plt.savefig("{0}".format(figure_name), dpi=300)
    # plt.show()


def get_realistic_function(xs, ys):
    xs, ys = filter_(xs, ys)
    f = interpolate.interp1d(xs, ys, kind=3)
    return f


def filter_(xs, ys):
    initPts = [x/10. for x in range(0,11,2)]
    newxs = []
    newys = []
    for i, x in enumerate(xs):
        if x in initPts:
            newxs.append(x)
            newys.append(ys[i])
    return newxs, newys


def iterative_refinement_demonstration(data_file, target_error):
    N = 5
    a = 0
    b = 1
    xs, ys, _ = parse_user_data(open(data_file).read())
    f = get_realistic_function(xs, ys)

    xs = list(np.linspace(a, b, N))
    es = np.zeros(N)
    ys = list(map(f, xs))

    x_fine = np.linspace(a, b, N*100)
    y_fine = list(map(f, x_fine))

    generated_pts = [xs, ys, es]
    total_error = target_error + 1
    while total_error > target_error:
        gap_errors, gap_xs, total_error = run_test(*(generated_pts + [x_fine, y_fine]))
        gap_error_pts = list(zip(gap_errors, gap_xs, ["gap"]*len(gap_errors)))
        largest_gap_error = reduce_error_on_residual_error(gap_error_pts, total_error-target_error, 0.5, False)
        if largest_gap_error:
            largest_gap_errors_x = list(zip(*largest_gap_error))[1]
        else:
            largest_gap_errors_x = []
        generated_pts = list(zip(*sorted(list(zip(*generated_pts)) + list(zip(largest_gap_errors_x, list(map(f, largest_gap_errors_x)), list(np.zeros(len(largest_gap_errors_x))) )))))


if __name__=="__main__":
    eg_data = "eg_data.dat"

    with open(eg_data) as fh:
        xs_eg, ys_eg, es_eg = parse_user_data(fh.read())

    run_reduce_error(xs_eg, ys_eg, es_eg, 0.5, 1, True, "test.png", 3, True)
    iterative_refinement_demonstration(eg_data, 0.5)