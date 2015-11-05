import json
import os
import glob
from itertools import groupby
from helpers import parse_user_data, rss, round_sigfigs
import numpy as np
import matplotlib.pyplot as plt
from integration_error_estimate import trapz_integrate_with_uncertainty
DATA_FILE = "av_dvdl.json"
RESULTS_FILE = "previous_results.json"
DATA_DIR = "av_dvdl_data"
RESULTS_DIR = "final_results"
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

def get_data(data_file, generation_method):
    if os.path.exists(data_file):
        data = json.load(open(data_file))
    else:
        data = generation_method()
        json.dump(data, open(data_file, "w"))
    return data

def plot_data(xs, ys, es):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    fig.hold(True)
    ax.errorbar(xs, ys, es, marker="o", label="Integration Points")
    plt.show()

def run():
    data = get_data(DATA_FILE, parse_data_dir)
    previous_results = get_data(RESULTS_FILE, parse_results_dir)
    n = len(data)
    print n
    results = {}
    round_sf = lambda x:round_sigfigs(x, 3)
    for molid, dvdl in data.items():
        res = trapz_integrate_with_uncertainty(*dvdl)
        results[molid] = map(round_sf, res[:2]) + list(res[2:]) + dvdl

    print "integral difference"
    n = 0
    for molid, result in sorted(results.items(), key=lambda x:x[1][1]):
        if abs(result[0] - round_sf(previous_results[molid]["DG"])) > 0.5:
            n += 1
            print "old {0:5}: {1:>5g} +/- {2:g}".format(molid, round_sf(previous_results[molid]["DG"]), round_sf(previous_results[molid]["Err"]))
            print "new {0:5}: {1:>5g} +/- {2:g}".format(molid, result[0], result[1])
            print "dif {0:5}: {1:>5g} ({2})".format(molid, previous_results[molid]["DG"] - result[0], result[1])
            print
    print n

    print "larger error"
    n = 0
    #for molid, result in sorted(results.items(), key=lambda x:previous_results[x[0]]["Err"]):
    for molid, result in sorted(results.items(), key=lambda x:x[1][1]):
        print "old {0:5}: {1:>5g} +/- {2:g}".format(molid, round_sf(previous_results[molid]["DG"]), round_sf(previous_results[molid]["Err"]))
        print "new {0:5}: {1:>5g} +/- {2:g}".format(molid, result[0], result[1])
        print "dif {0:5}: {1:>5g} ({2})".format(molid, previous_results[molid]["DG"] - result[0], result[1])
        print
if __name__=="__main__":
    run()