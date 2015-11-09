import sys
from numpy import array, mean, std, random, ones, trapz
import pylab as pl

CONV_TOT = 0.01
MAX_ITER = 50000
MIN_ITER = 20

def mc_trapz(xs, ys, es):
    mc_integral, errorEstTrap, conv = est_error_mc(xs, ys, es, trapz)
    if not conv:
        sys.stderr.write("WARNING: MC error estimate did not fully converge.\n")
    return mc_integral, errorEstTrap

def est_error_mc(xs, ys, es, method, plot=False):
    expctValue = method(ys, xs)
    y_trials = samplY(ys, es, MAX_ITER)
    trialResults = [method(y_trial, xs) for y_trial in y_trials]
    if plot:
        fig = pl.figure()
        ax = fig.add_subplot(111)
        std_conv, mean_conv = zip(*[(std(trialResults[:i+1]), mean(trialResults[:i+1])) for i in range(len(trialResults)-1)])
        ax.plot(range(len(trialResults)-1), std_conv, "r")
        ax1 = ax.twinx()
        ax1.plot(range(len(trialResults)-1), mean_conv, "b")
        ax1.plot([0, len(trialResults) - 1], [expctValue]*2, "g")
        pl.show()
    return mean(trialResults), std(trialResults), converged(trialResults, expctValue)

def samplY(ys, es, N):
    yTrial = [random.normal(mu, sig, N) if sig != 0 else mu*ones(N) for mu, sig in zip(ys, es)]
    return array(yTrial).transpose()

def converged(trialResults, expctValue):
    if abs(mean(trialResults) - expctValue) > CONV_TOT:
        return False
    if abs(std(trialResults) - std(trialResults[:-MIN_ITER])) > CONV_TOT:
        return False
    return True
