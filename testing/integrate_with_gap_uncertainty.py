import numpy as np
import pylab as pl
from scipy.integrate import simps, trapz
from simulated_iterative_integration import get_realistic_function

CONVERGENCE_RATE_SCALING = 1

def second_derivative(pts):
    (x1, y1), (x2, y2), (x3, y3) = pts
    return abs((((y3-y2)/(x3-x2)) - ((y2-y1)/(x2-x1)) )/((x3-x1)/2.))

def trapz_interval_error(pts, dx):
    return (dx**3)/12.*second_derivative(pts)

def four_pt_trapz_interval_error(pts):
    dx = pts[2][0] - pts[1][0]
    return np.max([trapz_interval_error(pts[1:], dx), trapz_interval_error(pts[:-1], dx)])

def trapz_with_uncertainty(xs, ys):
    gap_xs, gap_ys, gap_es = get_gap_errors(xs, ys)
    #total_error_est = np.sum(gap_es)
    total_error_est = np.sqrt(np.sum(np.square(gap_es)))
    return trapz(ys, xs), total_error_est, gap_xs, gap_ys, gap_es

def get_gap_errors(xs, ys):
    '''
    Add new points to the center of each 4-point region with uncertainty estimated from 
    linear extrapolation of points 1->2 and 4->3. 
    '''
    gap_xs = []
    gap_ys = []
    gap_es = []
    n_regions = len(xs)-3
    for i in range(n_regions):
        # treat boundary regions as 3 pts
        if i == 0:
            gapXs = xs[i:i+3]
            gapYs = ys[i:i+3]

            # calculate the x-value of the gap center
            x_gapCenter = (gapXs[0] + gapXs[1])/2.

            yPt, err = calc_point_with_uncertainty_second_derivative(gapXs, gapYs, x_gapCenter, boundary_left=True)
            gap_xs.append(x_gapCenter)
            gap_ys.append(yPt)
            gap_es.append(err)

        # a set of 4 consecutive points
        gapXs = xs[i:i+4]
        gapYs = ys[i:i+4]
        # calculate the x-value of the gap center
        x_gapCenter = (gapXs[1] + gapXs[2])/2.

        # calculate the y-value at the gap center x-value by linearly interpolation between points 2 and 3.
        # Also estimate the uncertainty in the y-value as the maximum difference between the interpolated y-value
        # and a linearly extrapolated y-value using points 1 -> 2 (forward), and 4 -> 3 (reverse).   
        #yPt, err = calc_point_with_uncertainty_interpolation(gapXs, gapYs, x_gapCenter)
        yPt, err = calc_point_with_uncertainty_second_derivative(gapXs, gapYs, x_gapCenter)
        print err
        gap_xs.append(x_gapCenter)
        gap_ys.append(yPt)
        gap_es.append(err)

        if i == n_regions - 1:
            gapXs = xs[-3:]
            gapYs = ys[-3:]

            # calculate the x-value of the gap center
            x_gapCenter = (gapXs[1] + gapXs[2])/2.

            yPt, err = calc_point_with_uncertainty_second_derivative(gapXs, gapYs, x_gapCenter, boundary_right=True)
            gap_xs.append(x_gapCenter)
            gap_ys.append(yPt)
            gap_es.append(err)

    return gap_xs, gap_ys, gap_es

def calc_point_with_uncertainty_second_derivative(x_pts, y_pts, x_center, boundary_left=False, boundary_right=False):
    pts = zip(x_pts, y_pts)

    # calculate y-value by interpolation
    if boundary_left:
        y = calc_y_intersection_pt(pts[0], pts[1], x_center)
        return y, trapz_interval_error(pts, dx=(pts[1][0] - pts[0][0]))
    if boundary_right:
        y = calc_y_intersection_pt(pts[1], pts[2], x_center)
        return y, trapz_interval_error(pts, dx=(pts[-1][0] - pts[-2][0]))
    else:
        y = calc_y_intersection_pt(pts[1], pts[2], x_center)
        return y, four_pt_trapz_interval_error(pts)

def calc_point_with_uncertainty_interpolation(gapXs, gapYs, x_c):
    '''
    New y-point will be at the intersection of the a linear interpolation from point 2 -> 3 and x_c.   
    The y-point uncertainty is the maximum of the linearly extrapolation from points:
        1 -> 2 to the line x = x_c
    and from 
        4 -> 3 to the line x = x_c

    This is calculated by finding the intersection point of the line formed by pt1, pt2 (paramaterized by m1 and c1) 
    and the center of the middle gap (x_c); and then similarly for pt3, pt4
    '''

    pts = zip(gapXs, gapYs)

    # calculate y-value by interpolation
    y = calc_y_intersection_pt(pts[1], pts[2], x_c)

    # calculate y-values by extrapolation
    yf = calc_y_intersection_pt(pts[0], pts[1], x_c)
    yr = calc_y_intersection_pt(pts[2], pts[3], x_c)

    forwardErr = abs(yf-y)
    reverseErr = abs(yr-y)
    return y, np.mean([forwardErr, reverseErr])

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

    #if c == float("nan") or c == float("inf"):
    #print "c = {0}; x1 and x2: {1}, {2}".format(c, x1, x2)
    # if y2 == y1
    if m == 0.0:
        return y1

    # intersection with x_c
    yi = m*x_c + c
    return yi

def select_points_on_residule_error(gap_error_pts, residule_error):
    largest_error_xs = []
    for e, x in gap_error_pts:
        largest_error_xs.append(x)
        residule_error -= (1./CONVERGENCE_RATE_SCALING)*0.75*e # since addition of a point to an interval reduces error by a factor of 1/4 => (e - e/4)
        if residule_error < 0:
            break
    return largest_error_xs

def select_points_on_average_error_tolerance(gap_error_pts, target_error):
    average_error_tolerance = target_error/np.sqrt(len(gap_error_pts))
    return [x for e, x in gap_error_pts if e > (1./CONVERGENCE_RATE_SCALING)*average_error_tolerance]

def additional_points(gap_xs, gap_es, current_error, target_error, residule_method=True):
    gap_error_pts = sorted(zip(gap_es, gap_xs))[::-1]
    residule_error = current_error - target_error
    if residule_method:
        largest_error_xs = select_points_on_residule_error(gap_error_pts, residule_error)
    else:
        largest_error_xs = select_points_on_average_error_tolerance(gap_error_pts, target_error)
    return largest_error_xs


def show_results(trapz_integral, trapz_est_error, fine_integral, xs, ys, gap_xs, gap_ys, gap_es, x_fine, y_fine, plot):
    print "estimated trap error: {0} +/- {1}".format(trapz_integral, trapz_est_error)
    print "fine integral: {0} (true error={1})".format(fine_integral, abs(fine_integral-trapz_integral))
    if plot:
        fig = pl.figure()
        ax = fig.add_subplot(111)
        fig.hold(True)
        ax.plot(xs, ys, marker="o")
        ax.errorbar(gap_xs, gap_ys, 16*np.array(gap_es), linestyle="")
        ax.plot(x_fine, y_fine, "r")
        pl.show()

def trapz_errorbased_integration(function, a, b, target_uncertainty, plot=True):
    xs = np.linspace(0, 1, 11)
    ys = map(function, xs)
    x_fine = np.linspace(a, b, 1000)
    y_fine = map(function, x_fine)
    fine_integral = simps(y_fine, x_fine)
    trapz_integral, trapz_est_error, gap_xs, gap_ys, gap_es = trapz_with_uncertainty(xs, ys)
    show_results(trapz_integral, trapz_est_error, fine_integral, xs, ys, gap_xs, gap_ys, gap_es, x_fine, y_fine, plot)
    while trapz_est_error > target_uncertainty:
        new_xs = additional_points(gap_xs, gap_es, trapz_est_error, target_uncertainty)
        print "gaps which received new points: {0:.1f}%".format(len(new_xs)/float(len(gap_xs))*100)
        xs = sorted(list(xs) + new_xs)
        ys = map(function, xs)
        trapz_integral, trapz_est_error, gap_xs, gap_ys, gap_es = trapz_with_uncertainty(xs, ys)
        show_results(trapz_integral, trapz_est_error, fine_integral, xs, ys, gap_xs, gap_ys, gap_es, x_fine, y_fine, plot)

def general_trapz_integration():
    target_uncertainty = 1
    f = get_realistic_function()

    a, b = 0, 1
    trapz_errorbased_integration(f, a, b, target_uncertainty)

def compair_integration():
    N = 10
    f = get_realistic_function()
    #f = lambda x:x**2

    start, end = 0, 1
    xs = list(np.linspace(start, end, N))
    ys = map(f, xs)

    x_fine = np.linspace(start, end, N*100)
    y_fine = map(f, x_fine)

    fig = pl.figure()
    ax = fig.add_subplot(111)
    fig.hold(True)
    ax.plot(xs, ys)
    est_integral, est_error, xs, ys, es = trapz_with_uncertainty(xs, ys)
    print "estimated trap error: {0} +/- {1}".format(est_integral, est_error)
    fine_integral = simps(y_fine, x_fine)
    print "fine integral: {0} (true error={1})".format(fine_integral, abs(fine_integral-est_integral))
    ax.errorbar(xs, ys, 16*np.array(es), linestyle="")

    ax.plot(x_fine, y_fine, "r")
    pl.show()

if __name__=="__main__":
    #compair_integration()
    general_trapz_integration()