import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from integration_errors.config import CONVERGENCE_RATE_SCALING
from integration_errors.calculate_error import config_argparse, process_plot_argument, parse_user_data, \
    plot_error_analysis, trapz_integrate_with_uncertainty
from integration_errors.helpers import round_sigfigs, rss

def reduce_error_on_residual_error(error_pts, residule_error, convergence_rate_scaling, be_conservative):

    if be_conservative:
        # since pts are already sorted we can simply filter out all non-gap points and take the first which remains
        largest_gap_error_pt = [pt for pt in error_pts if pt[-1] == "gap"][0]
    largest_error_pts = []
    for pt in sorted(error_pts, key=lambda x:abs(x[0]))[::-1]:
        if residule_error < 0:
            break
        largest_error_pts.append(pt)
        # Add special case for largest gap error if we're being conservative
        if be_conservative and pt == largest_gap_error_pt:
            residule_error -= abs(pt[0])
        # Since addition of a point to an interval reduces error by a factor of 1/4 according 
        # to the truncation error estimate method we're using => (e - e/4) = 0.75*e.
        # We simply assume the same is true for reducing the uncertainty on an existing point.
        # The convergence rate scaling factor allows for additional control over the number of points
        # that will be added on each iteration.
        residule_error -= (1./convergence_rate_scaling)*0.75*abs(pt[0])

    return largest_error_pts

def get_updates(xs, integration_point_errors, gap_xs, gap_errors, trapz_est_error, target_uncertainty, convergence_rate_scaling, be_conservative=True):
    n_gaps = len(gap_xs)
    gap_error_pts = zip(gap_errors, gap_xs, ["gap"]*n_gaps)
    pts_errors = zip(integration_point_errors, xs)

    combined_pts_errors = gap_error_pts + pts_errors
    residule_error = abs(trapz_est_error) - target_uncertainty

    largest_error_pts = reduce_error_on_residual_error(combined_pts_errors, residule_error, convergence_rate_scaling, be_conservative)

    is_gap = lambda x:x[-1] == "gap"

    update_xs = [map(float, e) for e in largest_error_pts if not is_gap(e)]
    new_pts = [map(float, e[:-1]) for e in largest_error_pts if is_gap(e)]
    return new_pts, update_xs

def parse_args():
    argparser = config_argparse()
    argparser.add_argument('-t', '--target_error', type=float, required=True, nargs=1,
                        help="Target error of integration, used to determine how to reduce integration uncertainty.")
    argparser.add_argument('-r', '--convergence_rate_scaling', type=float, default=CONVERGENCE_RATE_SCALING,
                        help="Determines the rate of convergence to the target error i.e. how many iterations will be required to reach target error. "\
                        "A value < 1 will result in more iterations but fewer overall points as the points will be concentrated in regions of high "\
                        "uncertainty; conversely values > 1 will result it more points but fewer iterations required to reach a given target error. Default=1.")
    args = argparser.parse_args()
    figure_name = process_plot_argument(args)
    with open(args.data) as fh:
        data = parse_user_data(fh.read())
    return data, figure_name, args.conservative, args.sigfigs, args.verbose, args.target_error, args.convergence_rate_scaling

def run(xs, ys, es, target_error, convergence_rate_scaling, be_conservative, figure_name, sigfigs, verbose):
    integral, total_error, gap_xs, gap_ys, gap_errors, integration_point_errors, conservative_error_adjustment = \
        trapz_integrate_with_uncertainty(xs, ys, es, be_conservative=be_conservative)
    round_sf = lambda x:round_sigfigs(x, sigfigs)
    result_string = "{0:g} +/- {1:g}".format(round_sf(integral), round_sf(total_error))
    if verbose:
        value_error = round_sf(rss(integration_point_errors))
        print "Error from y-value uncertainty: +/- {0:g}".format(value_error)
        truncation_error = round_sf(np.sum(gap_errors) if be_conservative else np.sum(gap_errors))
        print "Estimated truncation error: {0:g}".format(truncation_error)
        if be_conservative:
            print "Additional error component: {0:g}".format(round_sf(conservative_error_adjustment))
        print "Integral: {0}".format(result_string)

        print "Truncation errors: interval midpoint -> error"
        print "\n".join(["{1:<6.4f} -> {0:>7.4f}".format(*map(round_sf, d)) for d in sorted(zip(gap_errors, gap_xs), key=lambda x:abs(x[0]))[::-1]])
        print "Point errors: point -> +/- error"
        print "\n".join(["{1:<6.4f} -> +/- {0:6.4f}".format(*map(round_sf, d)) for d in sorted(zip(integration_point_errors, xs))[::-1]])

    else:
        print result_string

    if total_error > target_error:
        new_pts, update_xs = get_updates(xs, integration_point_errors, gap_xs, gap_errors, total_error, target_error, convergence_rate_scaling, be_conservative=be_conservative)
        if new_pts:
            print "Suggested new points:"
            print ",".join(["{0:g}".format(round_sf(p[1])) for p in new_pts])
        if update_xs:
            print "Suggested to reduce uncertainty in existing points:"
            print ",".join(["{0:g}".format(round_sf(p[1])) for p in update_xs])
    else:
        print "Target error has been reached."

    if figure_name:
        plot_error_analysis(xs, ys, es, gap_xs, gap_ys, gap_errors, figure_name, title="Integral: {0}".format(result_string))

def main():
    data, figure_name, be_conservative, sigfigs, verbose, target_error, convergence_rate_scaling = parse_args()
    xs, ys, es = np.array(data)
    run(xs, ys, es, target_error, convergence_rate_scaling, be_conservative, figure_name, sigfigs, verbose)

if __name__=="__main__":
    main()
