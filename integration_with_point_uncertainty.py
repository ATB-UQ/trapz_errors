from monte_carlo_integration import mc_trapz
from error_estimated_integration import integrate_with_point_uncertinaty
import numpy as np

import pylab as pl

#np.random.seed(9)
N = 50
SIGMA = .5

def test_piecewise_function():
    xs = range(0,20,2)
    ys = np.random.uniform(0, SIGMA, len(xs))
    f = piecewise_function_generator(zip(xs, ys))
    X = np.linspace(0, 15.999, 200)
    ys = map(f, X)
    fig = pl.figure()
    ax = fig.add_subplot(111)
    ax.plot(X, ys)
    pl.show()

def piecewise_function_generator(discontinuous_points):
    sorted_points = sorted(discontinuous_points)
    p1_p2 = lambda x_1: [ [sorted_points[i-1], (x,y)] for (i, (x, y)) in enumerate(sorted_points) if x >= x_1 ][0]
    return lambda x: linear_function_evaluator(*(p1_p2(x) + [x]))

def linear_function_evaluator(pt1, pt2, xi):
    x1, y1 = pt1
    x2, y2 = pt2

    # gradient of line 1
    m = (y2 - y1)/float(x2 - x1)
    # y-intercept of line1
    c = (x1*y2 - x2*y1)/float(x1 - x2)

    return m*xi+c

def f(x):
    return np.sin(x) + 2*np.cos(3*x-.5) + 2*np.sin(x-.2)

def get_xs_es(start, end, number_pts):
    #initial_pts = np.linspace(start, end, number_pts*2)[1:-1]
    xs = list(np.linspace(start, end, number_pts))
    return xs, np.random.uniform(0, SIGMA, number_pts)

def compair_integration():
    xs, es = get_xs_es(0, 1*np.pi, N)
    ys = map(f, xs)
    fig = pl.figure()
    ax = fig.add_subplot(111)
    ax.errorbar(xs, ys, es)
    pl.show()
    trap_integral, trap_error = mc_trapz(xs, ys, es)
    print "mc trap: {0} +/- {1}".format(trap_integral, trap_error)
    #simps_integral, simps_error = mc_simps(xs, ys, es)
    #print "mc simps: {0} +/- {1}".format(simps_integral, simps_error)
    trap_est_integral_p, trap_est_error_p = integrate_with_point_uncertinaty(xs, ys, es)
    print "estimated trap error (propagated): {0} +/- {1}".format(trap_est_integral_p, trap_est_error_p)
    #simps_est_integral, simps_est_error = integrate_with_uncertinaty(xs, ys, es, method=simps)
    #print "estimated simps error: {0} +/- {1}".format(simps_est_integral, simps_est_error)



if __name__=="__main__":
    compair_integration()
    #test_piecewise_function()