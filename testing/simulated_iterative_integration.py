import numpy as np
from scipy.integrate import simps
from integration_error_estimate import trapz_integrate_with_uncertainty
from monte_carlo_integration import mc_trapz
from scipy import interpolate
from reduce_integration_uncertainty import reduce_error_on_average_error_tolerance, reduce_error_on_residule_error

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

def simulate_updates(xs, es, integration_point_errors, gap_xs, gap_errors, trapz_est_error, target_uncertainty, residule_method=True):
    n_gaps = len(gap_xs)
    gap_error_pts = zip(gap_errors, gap_xs, np.random.uniform(0, SIGMA, n_gaps), ["gap"]*n_gaps)
    pts_errors = zip(integration_point_errors, xs, es)
    combined_pts_errors = sorted( gap_error_pts + pts_errors )[::-1]
    residule_error = trapz_est_error - target_uncertainty
    if residule_method:
        largest_error_pts = reduce_error_on_residule_error(combined_pts_errors, residule_error)
    else:
        largest_error_pts = reduce_error_on_average_error_tolerance(combined_pts_errors, target_uncertainty)

    # (xs, es)
    new_pts = [(pt[1], pt[2]) for pt in largest_error_pts if pt[-1] == "gap"]
    updated_xs = [pt[2] for pt in largest_error_pts if pt[-1] != "gap"]
    updated_pts = [(pt[1], 0) if pt[2] in updated_xs else (pt[1], pt[2]) for pt in pts_errors]

    sorted_pts = sorted(updated_pts + new_pts)
    print "Intervals which received new points: {0:.1f}%".format(len(new_pts)/float(len(gap_xs))*100)
    print "Updated points: {0:.1f}%".format(len(updated_xs)/float(len(xs))*100)
    xs, es = zip(*sorted_pts)

    return xs, es

def trapz_errorbased_integration(function, a, b, target_uncertainty, plot=True):
    xs = np.linspace(0, 1, 11)
    es = np.random.uniform(0, SIGMA, N_INIT)
    ys = map(function, xs)
    x_fine = np.linspace(a, b, 1000)
    y_fine = map(function, x_fine)
    fine_integral = simps(y_fine, x_fine)
    _, mc_error = mc_trapz(xs, ys, es)
    trapz_integral, trapz_est_error, gap_xs, gap_errors, integration_point_errors = trapz_integrate_with_uncertainty(xs, ys, es, plot=True)
    print "Estimated integral: {0} +/- {1}".format(trapz_integral, trapz_est_error)
    print "Fine scale integral: {0} true error {1}".format(fine_integral, np.sqrt(abs(fine_integral-trapz_integral)**2 + mc_error**2))
    while trapz_est_error > target_uncertainty:
        xs, es = simulate_updates(xs, es, integration_point_errors, gap_xs, gap_errors, trapz_est_error, target_uncertainty)
        ys = map(function, xs)
        trapz_integral, trapz_est_error, gap_xs, gap_errors, integration_point_errors = trapz_integrate_with_uncertainty(xs, ys, es, plot=True)
        _, mc_error = mc_trapz(xs, ys, es)
        print "Estimated integral: {0} +/- {1}".format(trapz_integral, trapz_est_error)
        print "Fine scale integral: {0} true error {1}".format(fine_integral, np.sqrt(abs(fine_integral-trapz_integral)**2 + mc_error**2))

def get_realistic_function():
    xs, ys, _ = getXYE(extract_fe_data(EXAMPLE_DVDL_DATA))
    xs, ys = filter_(xs, ys)
    ys = [y for y in ys]
    f = interpolate.interp1d(xs, ys, kind=2)
    return f

def extract_fe_data(rawData):
    dvdlDict = {}
    for row in rawData.splitlines():
        if row.startswith("#") or not row:
            continue
        cols = row.split()
        lam, dvdl, err = tuple(cols[:3])
        lam = float(lam)
        if not lam in dvdlDict.keys():
            dvdlDict[lam] = {}
        dvdlDict[lam].setdefault("dvdl", []).append(float(dvdl))
        dvdlDict[lam].setdefault("err", []).append(float(err))
    avData = {}
    for lam, vals in dvdlDict.items():
        if not lam in avData.keys():
            avData[lam] = {}
        # average value
        avData[lam]["dvdl"] = np.mean(vals["dvdl"])
        avData[lam]["err"] = vals["err"][0]
    return avData

def getXYE(avWatData):
    # prepair return data
    pts = [(lam, val["dvdl"], val["err"]) for lam, val in sorted(avWatData.items(), key=lambda x:x[0])]
    xs, ys, es = zip(*pts)
    return np.array(xs), np.array(ys), np.array(es)

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
    trapz_errorbased_integration(lambda x:10*f(x), a, b, target_uncertainty)

if __name__=="__main__":
    simulated_trapz_integration()