import numpy as np
from scipy.integrate import trapz
from scipy import interpolate
import os
import sys
sys.path.append("../")

if not os.environ.has_key("DISPLAY"):
    import matplotlib
    matplotlib.use("Agg")
    CAN_SHOW_PLOT = False
else:
    CAN_SHOW_PLOT = True

import matplotlib.pyplot as plt

from integration_errors.reduce_error import reduce_error_on_residual_error, run as run_reduce_error
from integration_errors.calculate_error import trapz_integrate_with_uncertainty, point_error_calc
from integration_errors.helpers import rss, parse_user_data

def integrate_with_point_uncertinaty(xs, ys, es):
    integration_error_per_point = point_error_calc(xs, es)
    return np.trapz(ys, xs), rss(integration_error_per_point)

def run_test(xs, ys, es, x_fine=None, y_fine=None):
    if x_fine is not None:
        integral = trapz(y_fine, x_fine)
        print "Integral: {0}".format(integral)
        print "True truncation error: {0}".format(trapz(ys, xs) - integral)

    trapz_integral, total_error, gap_xs, _, gap_errors, _, _ = trapz_integrate_with_uncertainty(xs, ys, es, be_conservative=True)
    #print "Truncation error estimate: {0} +/- {1}".format(trapz_integral, total_error - rss(integration_point_errors))
    print "Combined error estimate: {0} +/- {1}".format(trapz_integral, total_error)

    plot_data(xs, ys, es, x_fine, y_fine,
        figure_name = None if CAN_SHOW_PLOT else "example_integration_{0}.eps".format(len(xs)),
        )
    return gap_errors, gap_xs, total_error

def plot_data(xs, ys, es, x_fine=None, y_fine=None, figure_name=None, title=""):
    fig = plt.figure(figsize=(5,4))
    ax = fig.add_subplot(111)
    fig.hold(True)
    f = interpolate.interp1d(xs, ys, kind=1)
    if x_fine is not None and y_fine is not None:
        ax.fill_between(x_fine, y_fine, map(f, x_fine), facecolor='red', alpha=0.8)
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
    plt.show()

def get_realistic_function(xs, ys):
    xs, ys = filter_(xs, ys)
    f = interpolate.interp1d(xs, ys, kind=3)
    return f

def filter_(xs, ys):
    initPts = [x/10. for x in range(0,11,2)]
    newxs = []; newys = [];
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
    ys = map(f, xs)

    x_fine = np.linspace(a, b, N*100)
    y_fine = map(f, x_fine)

    generated_pts = [xs, ys, es]
    total_error = target_error + 1
    while total_error > target_error:
        gap_errors, gap_xs, total_error = run_test(*(generated_pts + [x_fine, y_fine]))
        gap_error_pts = zip(gap_errors, gap_xs, ["gap"]*len(gap_errors))
        largest_gap_error = reduce_error_on_residual_error(gap_error_pts, total_error-target_error, 0.5, False)
        if largest_gap_error:
            largest_gap_errors_x = zip(*largest_gap_error)[1]
        else:
            largest_gap_errors_x = []
        generated_pts = zip(*sorted(zip(*generated_pts) + zip(largest_gap_errors_x, map(f, largest_gap_errors_x), list(np.zeros(len(largest_gap_errors_x))) )))

if __name__=="__main__":
    eg_data = "eg_data.dat"

    with open(eg_data) as fh:
        xs, ys, es = parse_user_data(fh.read())

    run_reduce_error(xs, ys, es, 0.5, 1, True, "test.png", 3, True)
    iterative_refinement_demonstration(eg_data, 0.5)