import numpy as np
import pylab as pl
from scipy.integrate import simps, trapz
from scipy import interpolate
from integrate_with_gap_uncertainty import get_realistic_function
CONVERGENCE_RATE_SCALING = 1
SIGMA = 5
N_INIT = 11
def trapz_errorbased_integration(function, a, b, target_uncertainty, plot=True):
    xs = np.linspace(0, 1, 11)
    es = np.random.uniform(0, SIGMA, N_INIT)
    ys = map(function, xs)
    x_fine = np.linspace(a, b, 1000)
    y_fine = map(function, x_fine)
    fine_integral = simps(y_fine, x_fine)
    trapz_integral, trapz_est_error, gap_xs, gap_ys, gap_es = trapz_with_uncertainty(xs, ys)
    show_results(trapz_integral, trapz_est_error, fine_integral, xs, ys, gap_xs, gap_ys, gap_es, x_fine, y_fine, plot)
    while trapz_est_error > target_uncertainty:
        new_xs = additional_points(gap_xs, gap_es, trapz_est_error, target_uncertainty)
        print "gaps which received new points: {0:.1f}%".format(len(new_xs)/float(len(gap_xs))*100)
        xs = sorted(list(xs) + new_xs)
        ys = map(function, xs)
        trapz_integral, trapz_est_error, gap_xs, gap_ys, gap_es = trapz_with_uncertainty(xs, ys)
        show_results(trapz_integral, trapz_est_error, fine_integral, xs, ys, gap_xs, gap_ys, gap_es, x_fine, y_fine, plot)

def test_trapz_integration():
    target_uncertainty = 1
    f = get_realistic_function()

    a, b = 0, 1
    trapz_errorbased_integration(f, a, b, target_uncertainty)

if __name__=="__main__":
    #compair_integration()
    test_trapz_integration()