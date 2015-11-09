import json
import os
import glob
from itertools import groupby
from helpers import parse_user_data, rss, round_sigfigs
import numpy as np
import matplotlib.pyplot as plt
from integration_error_estimate import trapz_integrate_with_uncertainty
from scipy import interpolate, integrate
DATA_FILE = "av_dvdl.json"
RESULTS_FILE = "previous_results.json"
DATA_DIR = "av_dvdl_data"
RESULTS_DIR = "final_results"
INTEGRATION_RESULTS = "integration_results.json"
ERROR_CUTOFF = 5
BOUNDS_PADDING = 0.2
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

def get_data(data_file, generation_method, args=[], method=json):
    if os.path.exists(data_file):
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

def get_realistic_function(xs, ys):
    f = interpolate.interp1d(xs, ys, kind=3)
    return f

def get_results(data):
    results = {}
    round_sf = lambda x:round_sigfigs(x, 3)
    for molid, dvdl in data.items():
        if any([e > ERROR_CUTOFF for e in dvdl[2]]):
            continue
        res = trapz_integrate_with_uncertainty(*dvdl)
        f = get_realistic_function(dvdl[0], dvdl[1])
        results[molid] = map(round_sf, res[:2]) + list(res[2:]) + dvdl + [integrate.quad(f, 0, 1)[0]]
    return results

def re_run_integration():
    data = get_data(DATA_FILE, parse_data_dir)
    previous_results = get_data(RESULTS_FILE, parse_results_dir)
    results = get_data(INTEGRATION_RESULTS, get_results, [data])
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

    all_value = np.abs(actual_values + estimated_values)
    bounds = (min(all_value)-BOUNDS_PADDING, max(all_value)+BOUNDS_PADDING)

    markerSize = 10
    lineWidth = 1.25
    width = 1.25 # table borders and ticks
    tickWidth = 0.75
    fontSize = 12
    #mew = 1

    symbols = (u'o', u'v', u'^', u'<', u'>', u's', u'p', u'h', u'H', u'D', u'd')
    ax.plot(actual_values, estimated_values, "^", color="k", linestyle='None', label = "true vs predicted values", markersize = markerSize)


    #ax11.plot(x,y, '^',color=color,linestyle='None', label = 'ATB Version 2.2',markersize = markerSize, zorder=3)
    #ax11.plot(x,y, 'x',linestyle='None',label = 'Test Set molecules',markersize = markerSize-2, mew=mew)

    ax.plot(bounds, bounds, linestyle='--', color ='k', linewidth=lineWidth)

    plt.ylabel('Predicted truncation error', fontsize = fontSize, fontweight="bold")

    plt.xlabel('Actual truncation error', fontsize = fontSize, fontweight="bold")
    plt.xlim(bounds)
    plt.ylim(bounds)

    plt.xticks(fontsize = fontSize, fontweight="bold")
    plt.yticks(fontsize = fontSize, fontweight="bold")

    [i.set_linewidth(width) for i in ax.spines.itervalues()]
    plt.tick_params(which='major', length = 4, color ='k', width =tickWidth)

    #plt.legend(loc = 'lower right', prop={'size':11}, numpoints = 1, frameon = False)
    return

def run_error_stats():
    PLOT=False
    data = get_data(DATA_FILE, parse_data_dir)
    results = get_data(INTEGRATION_RESULTS, get_results, [data])
    round_sf = lambda x:round_sigfigs(x, 3)
    x_fine = np.linspace(0, 1, 100)
    truncation_errors = []
    estimated_truncation_error = []
    for molid, result in sorted(results.items(), key=lambda x:x[1][1]):
        if PLOT:
            f = get_realistic_function(result[-4], result[-3])
            ax = plot_data(result[-4], result[-3], np.zeros(len(result[-4])), show=False)
            plot_data(x_fine, map(f, x_fine), np.zeros(len(x_fine)), ax=ax, marker="")
        gap_errors = result[4]
        print molid
        truncation_errors.append(abs(result[0] - result[-1]))
        estimated_truncation_error.append(np.abs(np.sum(zip(*gap_errors)[0])) + rss(zip(*gap_errors)[1]))
        #estimated_truncation_error.append(np.abs(rss(zip(*gap_errors)[0])))
        print "True truncation error: {0:>5g} ({1})".format(result[-1], abs(result[0] - result[-1]))
        print "error breakdown: point error={0:g}, truncation error={1:g} +/- {2:g}".format(rss(result[5]), np.abs(np.sum(zip(*gap_errors)[0])), rss(zip(*gap_errors)[1]))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    fig.hold(True)
    plot_to_axis(ax, truncation_errors, estimated_truncation_error)
    print "Percentage under estimated: {0}%".format(100*float(len([1 for te, ete in zip(truncation_errors, estimated_truncation_error) if te > ete]))/float(len(truncation_errors)))
    plt.show()
if __name__=="__main__":
    #re_run_integration()
    run_error_stats()