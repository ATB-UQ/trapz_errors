
from scipy.integrate import simps, trapz
from numpy import array, mean, std, random
import pylab as pl

CONV_TOT = 0.01
MAX_ITER = 10000
MIN_ITER = 20

def mc_simps(xs, ys, es):
    simpIntegral = simps(ys, xs, even='avg')
    errorEstSimps, conv = estErrorMC(xs, ys, es, simps)
    if not conv:
        print "WARNING: MC error estimate did not fully converge."
    return simpIntegral, errorEstSimps

def mc_trapz(xs, ys, es):
    trapIntegral = trapz(ys, xs)
    errorEstTrap, conv = estErrorMC(xs, ys, es, trapz)
    if not conv:
        print "WARNING: MC error estimate did not fully converge."
    return trapIntegral, errorEstTrap

def estErrorMC(xs, ys, es, method, plot=True):
    expctValue = method(ys, xs)
    y_trials = samplY(ys, es, MAX_ITER)
    trialResults = [method(y_trial, xs) for y_trial in y_trials]
    if plot:
        fig = pl.figure()
        ax = fig.add_subplot(111)
        ax.plot(range(len(trialResults)-1), [std(trialResults[:i+1]) for i in range(len(trialResults)-1)])
        pl.show()
    return std(trialResults), converged(trialResults, expctValue)

def samplY(ys, es, N):
    yTrial = [random.normal(mu, sig, N) for mu, sig in zip(ys, es)]
    return array(yTrial).transpose()

def converged(trialResults, expctValue):
    if abs(mean(trialResults) - expctValue) > CONV_TOT:
        return False
    if abs(std(trialResults) - std(trialResults[:-MIN_ITER])) > CONV_TOT:
        return False
    return True
