from scipy.integrate import trapz
import numpy as np
from integrate_with_gap_uncertainty import get_realistic_function
from monte_carlo_integration import mc_trapz

def calc_y_intersection_pt(pt1, pt2, x_c):
    '''
    Return the point at the intersection of the linear function defined by pt1,pt2 and the line x = x_c
    '''
    x1, y1 = pt1
    x2, y2 = pt2
    # gradient of line 1
    m = (y2 - y1)/float(x2 - x1)
    # y-intercept of line1
    c = (x1*y2 - x2*y1)/float(x1 - x2)
    # intersection with x_c
    yi = m*x_c + c
    return yi

def second_derivative(pts):
    (x1, y1), (x2, y2), (x3, y3) = pts
    return abs( ((y3-y2)/(x3-x2) - (y2-y1)/(x2-x1))/((x3-x1)/2.) )

def trapz_interval_error(pts, dx):
    return (dx**3)/12.*second_derivative(pts)

def four_pt_trapz_interval_error(pts):
    dx = pts[2][0] - pts[1][0]
    return np.max([trapz_interval_error(pts[1:], dx), trapz_interval_error(pts[:-1], dx)])

def point_error_calc(xs, es):
    '''
    This method is based on propagation of point uncertainty for Trapezoidal algorithm on 
    non-uniformly distributed points: https://en.wikipedia.org/wiki/Trapezoidal_rule#Non-uniform_grid
    '''
    # left boundary point
    errors = [ 0.5*(xs[1]-xs[0])*es[0] ]
    for i in range(len(xs)-2):
        # intermediate point error
        errors.append( 0.5*(xs[i+2]-xs[i])*es[i+1] )
    # right boundary point
    errors.append( 0.5*(xs[-1]-xs[-2])*es[-1] )
    # return half the RSS of individual errors (factor of 2 is due to double counting of domain).
    return errors

def interval_errors(xs, ys):
    '''
    Based on analytical Trapezoidal error function with 2nd derivative estimated numerically:
    https://en.wikipedia.org/wiki/Trapezoidal_rule#Error_analysis
    '''
    pts = zip(xs, ys)

    gap_xs = [ (xs[0] + xs[1])/2. ]
    gap_ys = [ calc_y_intersection_pt(pts[0], pts[1], gap_xs[0]) ]
    gap_es = [ trapz_interval_error(pts[:3], (xs[1] - xs[0])) ]

    for i in range(len(xs)-3):
        gap_xs.append( (xs[i+1] + xs[i+2])/2. )
        gap_ys.append( calc_y_intersection_pt(pts[i+1], pts[i+2], gap_xs[i+1]) )
        gap_es.append( four_pt_trapz_interval_error(pts[i:i+4]) )
        print gap_es[-1]

    gap_xs.append( (xs[-1] + xs[-2])/2. )
    gap_ys.append( calc_y_intersection_pt(pts[-2], pts[-1], gap_xs[-1]) )
    gap_es.append( trapz_interval_error(pts[-3:], (xs[-1] - xs[-2])) )

    return gap_xs, gap_ys, gap_es

def integrate_with_point_uncertinaty(xs, ys, es):
    integration_error_per_point = point_error_calc(xs, es)
    return trapz(ys, xs), np.sqrt(np.sum(np.array(integration_error_per_point)**2))

def integrate_with_gap_uncertinaty(xs, ys, rss=True):
    _, _, gap_es = interval_errors(xs, ys)
    total_error_est = np.sqrt(np.sum(np.square(gap_es))) if rss else np.sum(gap_es)
    return trapz(ys, xs), total_error_est

def trapz_integrate_with_uncertainty(xs, ys, es, rss=True, plot=False):
    integration_point_errors = point_error_calc(xs, es)
    gap_xs, gap_ys, gap_errors = interval_errors(xs, ys)
    if rss:
        total_error = np.sqrt(np.sum(np.square(integration_point_errors + gap_errors)))
    else:
        total_gap_errors = np.sum(gap_errors)
        total_error = np.sqrt(np.sum(np.square(integration_point_errors + total_gap_errors)))
    if plot:
        import pylab as pl
        fig = pl.figure()
        ax = fig.add_subplot(111)
        fig.hold(True)
        ax.errorbar(xs, ys, es, marker="o")
        ax.errorbar(gap_xs, gap_ys, 12.*np.array(gap_errors), linestyle="")
        pl.show()
    return trapz(ys, xs), total_error

def simple_test_function():
    xs = [0, 1, 2]
    ys = [0, 10, 0]
    es = [0.5, 0.5, 0.5]
    print integrate_with_point_uncertinaty(xs, ys, es)
    print mc_trapz(xs, ys, es)
    print integrate_with_gap_uncertinaty(xs, ys)
    print trapz_integrate_with_uncertainty(xs, ys, es, plot=True)

def periodic_test_function():
    N = 11
    a = 0
    b = np.pi
    f = lambda x: 100*(np.sin(x) + 2*np.cos(3*x-.5) + 2*np.sin(x-.2))
    xs = list(np.linspace(a, b, N))
    es = np.random.uniform(0, 10, N)
    ys = map(f, xs)

    x_fine = np.linspace(a, b, N*100)
    y_fine = map(f, x_fine)
    print trapz(y_fine, x_fine) - trapz(ys, xs)

    print integrate_with_point_uncertinaty(xs, ys, es)
    print mc_trapz(xs, ys, es)
    print integrate_with_gap_uncertinaty(xs, ys, rss=True)
    print trapz_integrate_with_uncertainty(xs, ys, es, plot=True)

def interpolated_test_function():
    N = 11
    a = 0
    b = 1
    f = get_realistic_function()
    xs = list(np.linspace(a, b, N))
    es = np.random.uniform(0, 10, N)
    ys = map(f, xs)

    x_fine = np.linspace(a, b, N*100)
    y_fine = map(f, x_fine)
    print trapz(y_fine, x_fine) - trapz(ys, xs)

    print integrate_with_point_uncertinaty(xs, ys, es)
    print mc_trapz(xs, ys, es)
    print integrate_with_gap_uncertinaty(xs, ys, rss=True)
    print trapz_integrate_with_uncertainty(xs, ys, es, plot=True)

if __name__=="__main__":
    #simple_test_function()
    periodic_test_function()
    interpolated_test_function()