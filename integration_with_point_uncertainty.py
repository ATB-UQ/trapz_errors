from monte_carlo_integration import mc_trapz, mc_simps
from error_estimated_integration import integrate_with_uncertinaty
import numpy as np
from scipy.integrate import trapz

import pylab as pl

#np.random.seed(9)
N = 15
SIGMA = 1

def test_piecewise_function():
    xs = range(0,20,2)
    ys = np.random.uniform(0, SIGMA, len(xs))
    f = f_random_piecewise_generator(zip(xs, ys))
    X = np.linspace(0, 15.999, 200)
    ys = map(f, X)
    fig = pl.figure()
    ax = fig.add_subplot(111)
    ax.plot(X, ys)
    pl.show()

def f_random_piecewise_generator(discontinuous_points):
    sorted_points = sorted(discontinuous_points)
    p1_p2 = lambda x_1: [ [sorted_points[i-1], (x,y)] for (i, (x, y)) in enumerate(sorted_points) if x >= x_1 ][0]
    return lambda x: linear_function_evaluator(*(p1_p2(x) + [x]))

def linear_function_evaluator(pt1, pt2, xi):
    '''
    Return the point at the intersection of the linear function defined by pt1,pt2 and the line x = x_c
    ''' 
    
    x1, y1 = pt1
    x2, y2 = pt2
    
    # gradient of line 1
    m = (y2 - y1)/float(x2 - x1)
    # y-intercept of line1
    c = (x1*y2 - x2*y1)/float(x1 - x2)
    
    #if c == float("nan") or c == float("inf"):
    #print "c = {0}; x1 and x2: {1}, {2}".format(c, x1, x2)
    # if y2 == y1

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
    trap_est_integral, trap_est_error = integrate_with_uncertinaty(xs, ys, es, method=trapz)
    print "estimated trap error: {0} +/- {1}".format(trap_est_integral, trap_est_error)
    #simps_est_integral, simps_est_error = integrate_with_uncertinaty(xs, ys, es, method=simps)
    #print "estimated simps error: {0} +/- {1}".format(simps_est_integral, simps_est_error)



if __name__=="__main__":
    compair_integration()
    #test_piecewise_function()