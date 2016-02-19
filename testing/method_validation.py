import pickle
import json
import os
import glob
from itertools import groupby
from helpers import parse_user_data, rss, round_sigfigs
import numpy as np
import matplotlib.pyplot as plt
from integration_error_estimate import trapz_integrate_with_uncertainty
from scipy import interpolate, integrate
from reduce_integration_uncertainty import reduce_error_on_residual_error
#import sys
#sys.path.append("../../")
#from fit_dihedral_potential.fit_dihedral import find_optimal_fit
DATA_FILE = "av_dvdl.pick"
RESULTS_FILE = "previous_results.pick"
DATA_DIR = "av_dvdl_data"
RESULTS_DIR = "final_results"
INTEGRATION_RESULTS = "integration_results.pick"

CACHE_RESULTS = False

ERROR_CUTOFF = 200
BOUNDS_PADDING = 0.2
TARGET_ERROR = 1
INIT_PTS = list(np.linspace(0, 1, num=5))
X_FINE = list(np.linspace(0, 1, num=100))
def parse_data_dir():
    data = {}
    sort_func = lambda x:os.path.basename(x).split("_")[0]
    for g, files in groupby(sorted(glob.glob("{0}/*".format(DATA_DIR)), key=sort_func), sort_func):

        raw_data = []
        for f in sorted(files):
            with open(f) as fh:
                raw_data.append(fh.read())
        data[g] = map(list, parse_dvdl(*raw_data))

    return data

def parse_results_dir():
    results = {}
    for f in sorted(glob.glob("{0}/*".format(RESULTS_DIR))):
        with open(f) as fh:
            results[os.path.basename(f).split("_")[0]] = json.load(fh)
    return results

def parse_dvdl(av_dvdl_vac, av_dvdl_wat):
    vac = remove_duplicates(parse_user_data(av_dvdl_vac))
    wat = remove_duplicates(parse_user_data(av_dvdl_wat))

    assert all([vx==wx for vx, wx in zip(vac[0], wat[0])])
    xs = vac[0]
    ys = np.array(vac[1]) - np.array(wat[1])
    es = [rss([e1, e2]) for e1, e2 in zip(vac[2], wat[2])]
    return xs, ys, es

def remove_duplicates(data):
    data_filtered = []
    for d in zip(*data):
        if data_filtered and d[0] in zip(*data_filtered)[0]:
            continue
        data_filtered.append(d)
    return zip(*data_filtered)

def get_data(data_file, generation_method, args=[], method=pickle, use_cache=True):
    if os.path.exists(data_file) and use_cache:
        data = method.load(open(data_file))
    else:
        data = generation_method(*args)
        method.dump(data, open(data_file, "w"))
    return data

def plot_data(xs, ys, es, show=True, ax=None, marker="o"):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        fig.hold(True)
    ax.errorbar(xs, ys, es, marker=marker, label="Integration Points")
    if show:
        plt.show()
    return ax

def get_realistic_function(xs, ys, n_divisions=2):

    def get_pt_x_division(pts):
        return [0.5*(pts[i][0] + pts[i+1][0]) for i in range(len(pts) - 1)]

    pts = zip(xs, ys)
    sorted_y = sorted(pts, key=lambda x:x[1])
    min_y_pt, max_y_pt = sorted_y[0], sorted_y[-1]
    min_x_pt, max_x_pt = pts[0], pts[-1]
    mid_pt = get_pts_closest_to(pts, [0.5])[0]
    initial_pts = sorted([mid_pt, min_y_pt, max_y_pt, min_x_pt, max_x_pt])
    filtered_pts = initial_pts
    for _ in range(n_divisions):
        filtered_pts.extend(get_pts_closest_to(pts, get_pt_x_division(sorted(filtered_pts))))

    filtered_pts = sorted(list(set(filtered_pts)))
    filtered_x, filtered_y = zip(*filtered_pts)
    
    
    f = interpolate.interp1d(filtered_x, filtered_y, kind=3)
    if False:
        ax = plot_data(filtered_x, filtered_y, np.zeros(len(filtered_y)), show=False)
        plot_data(X_FINE, map(f,X_FINE), np.zeros(len(X_FINE)), ax=ax, marker="")
    #f = fit_periodic(filtered_x, filtered_y)
    #f = interpolate.InterpolatedUnivariateSpline(filtered_x, filtered_y)
    return lambda x:float(f(x))

def get_pts_closest_to(pts, xs):
    sort_x_dist = lambda x_0: lambda x:abs(x[0] - x_0)
    return [sorted(pts, key=sort_x_dist(x))[0] for x in xs]

def get_results(data, num_iterations):
    results = {}
    round_sf = lambda x:round_sigfigs(x, 3)
    for molid, dvdl in data.items():
        if any([e > ERROR_CUTOFF for e in dvdl[2]]):
            continue
        f = get_realistic_function(dvdl[0], dvdl[1])
        iteration_results = []
        true_integral = integrate.quad(f, 0, 1, epsabs=1e-03, epsrel=1e-3,)[0]
        generated_pts = [INIT_PTS, map(f, INIT_PTS), list(np.zeros(len(INIT_PTS)))]
        for _ in range(num_iterations):
            calc_integral, total_error, gap_xs, gap_ys, gap_errors, integration_point_errors, error_safety_policy = trapz_integrate_with_uncertainty(*generated_pts)
            iteration_results.append(map(round_sf, [calc_integral, total_error]) + [gap_xs, gap_ys, gap_errors, integration_point_errors] + [map(f, X_FINE)] + generated_pts + [true_integral])
            gap_error_pts = zip(gap_errors, gap_xs, ["gap"]*len(gap_errors))
            largest_gap_error = reduce_error_on_residual_error(gap_error_pts, total_error-TARGET_ERROR, 1, be_conservative=True)
            if largest_gap_error:
                largest_gap_errors_x = zip(*largest_gap_error)[1]
            else:
                largest_gap_errors_x = []
            generated_pts = zip(*sorted(zip(*generated_pts) + zip(largest_gap_errors_x, map(f, largest_gap_errors_x), list(np.zeros(len(largest_gap_errors_x))) )))
        results[molid] = iteration_results
    return results

def re_run_integration():
    data = get_data(DATA_FILE, parse_data_dir)
    previous_results = get_data(RESULTS_FILE, parse_results_dir)
    results = get_data(INTEGRATION_RESULTS, get_results, [data, 2])
    round_sf = lambda x:round_sigfigs(x, 3)
    print "integral difference"
    n = 0
    for molid, result in sorted(results.items(), key=lambda x:x[1][1]):
        if abs(result[0] - round_sf(previous_results[molid]["DG"])) > 0.5:
            n += 1
            print "old {0:5}: {1:>5g} +/- {2:g}".format(molid, round_sf(previous_results[molid]["DG"]), round_sf(previous_results[molid]["Err"]))
            print "new {0:5}: {1:>5g} +/- {2:g}".format(molid, result[0], result[1])
            print "dif {0:5}: {1:>5g} ({2})".format(molid, previous_results[molid]["DG"] - result[0], result[1])
            print "interpolated {0:5}: {1:>5g} ({2})".format(molid, result[-1], abs(result[0] - result[-1]))
            print
    print n

    print "sorted error"
    n = 0
    #for molid, result in sorted(results.items(), key=lambda x:previous_results[x[0]]["Err"]):
    for molid, result in sorted(results.items(), key=lambda x:x[1][1]):
        #if previous_results[molid]["Err"] < result[1]:
        print "old {0:5}: {1:>5g} +/- {2:g}".format(molid, round_sf(previous_results[molid]["DG"]), round_sf(previous_results[molid]["Err"]))
        print "new {0:5}: {1:>5g} +/- {2:g}".format(molid, result[0], result[1])
        print "dif {0:5}: {1:>5g} ({2})".format(molid, previous_results[molid]["DG"] - result[0], result[1])
        gap_errors = result[4]
        print "error breakdown: point error={0:g}, truncation error={1:g} +/- {2:g}".format(rss(result[5]), np.abs(np.sum(zip(*gap_errors)[0])), rss(zip(*gap_errors)[1]))
        print
        n += 1
    print n

def plot_to_axis(ax, actual_values, estimated_values):
    estimated_values, N = zip(*estimated_values)
    all_value = np.abs(actual_values + list(estimated_values))
    bounds = (min(all_value)-BOUNDS_PADDING, max(all_value)+BOUNDS_PADDING)
    if False and max(all_value) < 15:
        bounds=(0,15)
    markerSize = 10
    lineWidth = 1.25
    width = 1.25 # table borders and ticks
    tickWidth = 0.75
    fontSize = 12
    #mew = 1

    symbols = (u'o', u'v', u'^', u'<', u'>', u's', u'p', u'h', u'H', u'D', u'd')
    ax.scatter(actual_values, estimated_values, marker="o", alpha=0.5, label = "true vs predicted values", s = (np.array(N)/5.)**2)
    #ax.scatter(actual_values, estimated_values, marker="o", alpha=0.5, label = "true vs predicted values")

    
    #ax11.plot(x,y, '^',color=color,linestyle='None', label = 'ATB Version 2.2',markersize = markerSize, zorder=3)
    #ax11.plot(x,y, 'x',linestyle='None',label = 'Test Set molecules',markersize = markerSize-2, mew=mew)

    ax.plot(bounds, bounds, linestyle='--', color ='k', linewidth=lineWidth)

    plt.ylabel('Predicted truncation error', fontsize = fontSize, fontweight="bold")
    plt.xlabel('Actual truncation error', fontsize = fontSize, fontweight="bold")
    plt.title('Mean N={0:.0f}'.format(np.mean(N)+1), fontsize = fontSize, fontweight="bold")
    plt.xlim(bounds)
    plt.ylim(bounds)

    plt.xticks(fontsize = fontSize, fontweight="bold")
    plt.yticks(fontsize = fontSize, fontweight="bold")

    [i.set_linewidth(width) for i in ax.spines.itervalues()]
    plt.tick_params(which='major', length = 4, color ='k', width =tickWidth)
    plt.tight_layout()
    plt.savefig("close_up_error_prediction_{0:.0f}.eps".format(np.mean(N)+1), dpi=300)
    #plt.legend(loc = 'lower right', prop={'size':11}, numpoints = 1, frameon = False)
    return

def run_error_stats():
    PLOT=True
    data = get_data(DATA_FILE, parse_data_dir)
    results = get_data(INTEGRATION_RESULTS, get_results, [data, 8], use_cache=CACHE_RESULTS)
    round_sf = lambda x:round_sigfigs(x, 3)
    x_fine = np.linspace(0, 1, 100)

    print len(results)
    n = 0
    for i in range(len(results.values()[0])):
        truncation_errors = []
        estimated_truncation_error = []
        n += 1
        for molid, iteration_results in results.items():
            result = iteration_results[i]
            gap_errors = result[4]
            total_error = result[1]
            print molid
            truncation_errors.append(abs(result[0] - result[-1]))
            estimated_truncation_error.append( (total_error, len(gap_errors)))
            #estimated_truncation_error.append(np.abs(rss(zip(*gap_errors)[0])))
            print "True truncation error: {0:>5g} ({1})".format(result[-1], abs(result[0] - result[-1]))
            print "error breakdown: point error={0:g}, truncation error={1:g}".format(rss(result[5]), np.abs(np.sum(gap_errors)))
            if PLOT and n > 6 and abs(result[0] - result[-1]) - np.abs(np.sum(gap_errors)) > 1:
                print molid
                ax = plot_data(result[-4], result[-3], np.zeros(len(result[-4])), show=False)
                plot_data(X_FINE, result[-5], np.zeros(len(x_fine)), ax=ax, marker="")
        fig = plt.figure(figsize=(5,4))
        ax = fig.add_subplot(111)
        
        fig.hold(True)
        plot_to_axis(ax, truncation_errors, estimated_truncation_error)
        print "Percentage under estimated: {0}%".format(100*float(len([1 for te, ete in zip(truncation_errors, estimated_truncation_error) if te > ete[0]]))/float(len(truncation_errors)))
        plt.show()
if __name__=="__main__":
    #re_run_integration()
    run_error_stats()