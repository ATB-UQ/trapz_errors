import sys
sys.path.append("../")
from scipy.integrate import trapz
import numpy as np
from simulated_iterative_integration import get_realistic_function
from integration_with_point_uncertainty import integrate_with_point_uncertinaty
from monte_carlo_integration import mc_trapz
from integration_error_estimate import trapz_integrate_with_uncertainty, interval_errors, plot_error_analysis
from helpers import rss

def integrate_with_gap_uncertinaty(xs, ys, es):
    _, _, gap_es = interval_errors(xs, ys, es)
    truncation_error, truncation_error_error = np.abs(np.sum(zip(*gap_es)[0])), rss(zip(*gap_es)[1])
    return np.trapz(ys, xs), truncation_error, truncation_error_error

def run_test(xs, ys, es, x_fine=None, y_fine=None):
    if x_fine is not None:
        integral = trapz(y_fine, x_fine)
        print "Integral: {0}".format(integral)
        print "True truncation error: {0}".format(trapz(ys, xs) - integral)
    mc_integral, mc_error = mc_trapz(xs, ys, es)
    print "Truncation error estimate: {0} +/- {1}".format(*integrate_with_gap_uncertinaty(xs, ys, es)[1:])
    print "Monte Carlo integral point uncertainty: {0} +/- {1}".format(mc_integral, mc_error)
    print "Analytical integral point uncertainty: {0} +/- {1}".format(*integrate_with_point_uncertinaty(xs, ys, es))
    trapz_integral, total_error, gap_xs, gap_ys, gap_errors, _ = trapz_integrate_with_uncertainty(xs, ys, es)
    print "Combined error estimate: {0} +/- {1}".format(trapz_integral, total_error)
    plot_error_analysis(xs, ys, es, gap_xs, gap_ys, zip(*gap_errors)[0])

def simple_test_function():
    xs = [0, 1, 2]
    ys = [0, 10, 0]
    es = [0.5, 0.5, 0.5]
    run_test(xs, ys, es)

def periodic_test_function():
    N = 11
    a = 0
    b = np.pi
    f = lambda x: 100*(np.sin(x) + 2*np.cos(3*x-.5) + 2*np.sin(x-.2))
    xs = list(np.linspace(a, b, N))
    es = np.random.uniform(0, 10, N)
    ys = map(f, xs)

    x_fine = np.linspace(a, b, N*100)
    y_fine = map(f, x_fine)
    run_test(xs, ys, es, x_fine, y_fine)

def interpolated_test_function():
    N = 11
    a = 0
    b = 1
    f = get_realistic_function()
    xs = list(np.linspace(a, b, N))
    es = np.random.uniform(0, 10, N)
    ys = map(f, xs)

    x_fine = np.linspace(a, b, N*100)
    y_fine = map(f, x_fine)
    run_test(xs, ys, es, x_fine, y_fine)

if __name__=="__main__":
    simple_test_function()
    periodic_test_function()
    interpolated_test_function()