import json
import pickle
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
ERROR_CUTOFF = 3
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

def plot_data(xs, ys, es):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    fig.hold(True)
    ax.errorbar(xs, ys, es, marker="o", label="Integration Points")
    plt.show()

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

    print "larger error"
    n = 0
    #for molid, result in sorted(results.items(), key=lambda x:previous_results[x[0]]["Err"]):
    for molid, result in sorted(results.items(), key=lambda x:x[1][1]):
        if True or previous_results[molid]["Err"] < result[1]:
            x_fine = np.linspace(0, 1, 100)
            f = get_realistic_function(result[-4], result[-3])
            plot_data(result[-4], result[-3], np.zeros(len(result[-4])))
            plot_data(x_fine, map(f, x_fine), np.zeros(len(x_fine)))
            print "old {0:5}: {1:>5g} +/- {2:g}".format(molid, round_sf(previous_results[molid]["DG"]), round_sf(previous_results[molid]["Err"]))
            print "new {0:5}: {1:>5g} +/- {2:g}".format(molid, result[0], result[1])
            print "dif {0:5}: {1:>5g} ({2})".format(molid, previous_results[molid]["DG"] - result[0], result[1])
            gap_errors = result[4]
            print "error breakdown: point error={0:g}, truncation error={1:g} +/- {2:g}".format(rss(result[4]), np.abs(np.sum(zip(*gap_errors)[0])), rss(zip(*gap_errors)[1]))
            print
            n += 1
    print n
if __name__=="__main__":
    run()