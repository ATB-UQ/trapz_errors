import numpy as np
from itertools import groupby
from config import CONVERGENCE_RATE_SCALING, PARALLELIZATION_METHODS
from integration_error_estimate import config_argparse, process_plot_argument, parse_user_data, \
    plot_error_analysis, trapz_integrate_with_uncertainty
from helpers import round_sigfigs, rss

def reduce_error_on_residual_error(error_pts, residule_error, convergence_rate_scaling):
    largest_error_pts = []
    for pt in error_pts:
        if residule_error < 0:
            break
        largest_error_pts.append(pt)
        residule_error -= (1./convergence_rate_scaling)*0.75*pt[0] # since addition of a point to an interval reduces error by a factor of 1/4 => (e - e/4)

    return largest_error_pts

def reduce_error_on_average_error_tolerance(error_pts, target_error, convergence_rate_scaling):
    average_error_tolerance = target_error/np.sqrt(len(error_pts))
    return [pt for pt in error_pts if pt[0] > (1./convergence_rate_scaling)*average_error_tolerance]

def get_updates(xs, integration_point_errors, gap_xs, gap_errors, trapz_est_error, target_uncertainty, convergence_rate_scaling, parallelization_method="residual"):
    n_gaps = len(gap_xs)
    gap_error_pts = zip(gap_errors, gap_xs, ["gap"]*n_gaps)
    pts_errors = zip(integration_point_errors, xs)

    combined_pts_errors = sorted( gap_error_pts + pts_errors )[::-1]
    residule_error = trapz_est_error - target_uncertainty
    if parallelization_method == "residual":
        largest_error_pts = reduce_error_on_residual_error(combined_pts_errors, residule_error, convergence_rate_scaling)
    else:
        largest_error_pts = reduce_error_on_average_error_tolerance(combined_pts_errors, target_uncertainty, convergence_rate_scaling)

    is_gap = lambda x:x[-1] == "gap"
    largest_error_pts = sorted(largest_error_pts, key=is_gap)
    grouped_errors = []
    for _, values in groupby(largest_error_pts, is_gap):
        grouped_errors.append(list(values))

    update_xs, new_pts = grouped_errors
    return new_pts, update_xs

def parse_args():
    argparser = config_argparse()
    argparser.add_argument('-t', '--target_error', type=float, required=True, nargs=1,
                        help="Target error of integration, used to determine how to reduce integration uncertainty.")
    argparser.add_argument('-c', '--convergence_rate_scaling', type=float, nargs=1, default=CONVERGENCE_RATE_SCALING,
                        help="Scaling factor for rate of convergence to target error i.e. determines iterations required to reach target error. "\
                        "e.g. A value < 1 will result in more iterations but fewer overall points as the points will be concentrated in regions of high "\
                        "uncertainty; conversely values > 1 will result it more points but fewer iterations required to reach a given target error. Default=1.")
    argparser.add_argument('-m', '--parallelization_method', type=str, choices=PARALLELIZATION_METHODS, default="residual",
                        help="Method for determining how error reduction actions are chosen (e.g. addition of points or reducing uncertainty in existing points). "\
                        "Both methods choose points/intervals with highest error: 'residual' method adds until (1/convergence_rate_scaling)*(0.75)*sum(errors) > current_error - target_error; "\
                        "'average' method adds all error sources greater than (1./convergence_rate_scaling)*average_error_tolerance where average_error_tolerance = target_error/sqrt(n_error_sources)")
    args = argparser.parse_args()
    figure_name = process_plot_argument(args)
    data = parse_user_data(args.data.read())
    return data, figure_name, args.rss, args.sigfigs, args.verbose, args.target_error, args.convergence_rate_scaling, args.parallelization_method

def main():
    data, figure_name, use_rss, sigfigs, verbose, target_error, convergence_rate_scaling, parallelization_method = parse_args()
    xs, ys, es = np.array(data)
    integral, total_error, gap_xs, gap_ys, gap_errors, integration_point_errors = trapz_integrate_with_uncertainty(xs, ys, es, use_rss)

    round_sf = lambda x:round_sigfigs(x, sigfigs)
    result_string = "{0:g} +/- {1:g}".format(round_sf(integral), round_sf(total_error))
    if verbose:
        value_error = round_sf(rss(integration_point_errors))
        print "Error from y-value uncertainty: +/- {0:g}".format(value_error)
        truncation_error = round_sf(rss(gap_errors) if use_rss else np.sum(gap_errors))
        print "Estimated truncation error: +/- {0:g}".format(truncation_error)
        print "Total error = sqrt(y-value_error^2 + truncation_error^2): +/- {0:g}".format(round_sf(rss([value_error, truncation_error])))
        print "Integral: {0}".format(result_string)

        print "Truncation errors: interval midpoint (+/- error)"
        print "\n".join(["{1:<6g}(+/- {0:g})".format(*map(round_sf, d)) for d in sorted(zip(gap_errors, gap_xs))[::-1]])
        print "Point errors: point (+/- error)"
        print "\n".join(["{1:<6g}(+/- {0:g})".format(*map(round_sf, d)) for d in sorted(zip(integration_point_errors, xs))[::-1]])
    else:
        print result_string

    if total_error > target_error:
        new_pts, update_xs = get_updates(xs, integration_point_errors, gap_xs, gap_errors, total_error, target_error, convergence_rate_scaling, parallelization_method)
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

if __name__=="__main__":
    LOCAL_DEBUG = False
    if LOCAL_DEBUG:
        import sys
        TE = 0.3
        sys.argv.extend(["-m", "average", "-v","-t", str(TE), "-d", "/mddata/uqmstroe/amine_refinement/united_atom/TI_data/TISolv_15_9402_TI_H2O/avWater.dvdl"])
        main()
    else:
        main()