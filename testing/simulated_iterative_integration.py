import numpy as np
from integration_error_estimate import trapz_integrate_with_uncertainty,\
    plot_error_analysis
from scipy import interpolate
from reduce_integration_uncertainty import reduce_error_on_residual_error
from helpers import parse_user_data, rss
from config import CONVERGENCE_RATE_SCALING
from testing.integration_with_point_uncertainty import integrate_with_point_uncertinaty

SIGMA = 5
N_INIT = 11
EXAMPLE_DVDL_DATA = '''
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
0.750 -71.44813 3.10821
0.775 -94.12352 3.47328
0.800 -110.41922 3.80674
0.825 -127.66568 2.59015
0.850 -113.45883 1.90801
0.875 -97.05920 1.65601
0.900 -71.38330 0.95010
0.950 -30.13460 0.54041
1.000 3.15286 0.21252'''

def simulate_updates(xs, es, integration_point_errors, gap_xs, gap_errors, trapz_est_error, target_uncertainty):
    n_gaps = len(gap_xs)
    gap_error_pts = zip(gap_errors, gap_xs, np.random.uniform(0, SIGMA, n_gaps), ["gap"]*n_gaps)
    pts_errors = zip(integration_point_errors, xs, es)
    combined_pts_errors = sorted( gap_error_pts + pts_errors )[::-1]
    residule_error = trapz_est_error - target_uncertainty
    largest_error_pts = reduce_error_on_residual_error(combined_pts_errors, residule_error, CONVERGENCE_RATE_SCALING)

    # (xs, es)
    new_pts = [(pt[1], pt[2]) for pt in largest_error_pts if pt[-1] == "gap"]
    updated_xs = [pt[2] for pt in largest_error_pts if pt[-1] != "gap"]
    updated_pts = [(pt[1], 0) if pt[2] in updated_xs else (pt[1], pt[2]) for pt in pts_errors]

    sorted_pts = sorted(updated_pts + new_pts)
    print "Intervals which received new points: {0:.1f}%".format(len(new_pts)/float(len(gap_xs))*100)
    print "Updated points: {0:.1f}%".format(len(updated_xs)/float(len(xs))*100)
    xs, es = zip(*sorted_pts)

    return xs, es


def show_results(xs, es, ys, fine_integral, trapz_integral, trapz_est_error, gap_xs, gap_ys, gap_errors):
    #print "Monte Carlo integral point uncertainty: {0} +/- {1}".format(mc_integral, mc_error)
    _, point_uncertainty = integrate_with_point_uncertinaty(xs, ys, es)
    print "Analytical integral point uncertainty: +/-{0}".format(point_uncertainty)
    print "Truncation error estimate: {0} +/- {1}".format(np.sum(zip(*gap_errors)[0]), rss(zip(*gap_errors)[1]))
    print "True truncation error: {0}".format(trapz_integral - fine_integral)
    print
    print "Estimated integral: {0} +/- {1}".format(trapz_integral, trapz_est_error)
    print "Fine scale integral: {0} true error {1}".format(fine_integral, abs(fine_integral - trapz_integral) + point_uncertainty)
    plot_error_analysis(xs, ys, es, gap_xs, gap_ys, np.abs(zip(*gap_errors)[0]))


def do_iteration(function, target_uncertainty, xs, es, ys, fine_integral):
    ys = map(function, xs)
    trapz_integral, trapz_est_error, gap_xs, gap_ys, gap_errors, integration_point_errors = trapz_integrate_with_uncertainty(xs, ys, es)
    #mc_integral, mc_error = mc_trapz(xs, ys, es)
    show_results(xs, es, ys, fine_integral, trapz_integral, trapz_est_error, gap_xs, gap_ys, gap_errors)
    return integration_point_errors, gap_xs, gap_errors, trapz_est_error

def trapz_errorbased_integration(function, a, b, target_uncertainty, plot=True):
    xs = np.linspace(0, 1, 11)
    es = np.random.uniform(0, SIGMA, N_INIT)
    ys = map(function, xs)
    x_fine = np.linspace(a, b, 1000)
    y_fine = map(function, x_fine)
    fine_integral = np.trapz(y_fine, x_fine)
    integration_point_errors, gap_xs, gap_errors, trapz_est_error = do_iteration(function, target_uncertainty, xs, es, ys, fine_integral)
    while trapz_est_error > target_uncertainty:
        xs, es = simulate_updates(xs, es, integration_point_errors, gap_xs, np.abs(zip(*gap_errors)[0]), trapz_est_error, target_uncertainty)
        integration_point_errors, gap_xs, gap_errors, trapz_est_error = do_iteration(function, target_uncertainty, xs, es, ys, fine_integral)

def get_realistic_function():
    xs, ys, _ = parse_user_data(EXAMPLE_DVDL_DATA)
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

def simulated_trapz_integration():
    target_uncertainty = 0.5
    f = get_realistic_function()

    a, b = 0, 1
    trapz_errorbased_integration(f, a, b, target_uncertainty)

if __name__=="__main__":
    simulated_trapz_integration()