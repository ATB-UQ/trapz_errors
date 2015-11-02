import numpy as np
from itertools import groupby
from config import CONVERGENCE_RATE_SCALING
from integration_error_estimate import config_argparse, process_plot_argument, parse_user_data

def reduce_error_on_residule_error(error_pts, residule_error):
    largest_error_pts = []
    for pt in error_pts:
        largest_error_pts.append(pt)
        residule_error -= (1./CONVERGENCE_RATE_SCALING)*0.75*pt[0] # since addition of a point to an interval reduces error by a factor of 1/4 => (e - e/4)
        if residule_error < 0:
            break
    return largest_error_pts

def reduce_error_on_average_error_tolerance(error_pts, target_error):
    average_error_tolerance = target_error/np.sqrt(len(error_pts))
    return [pt for pt in error_pts if pt[0] > (1./CONVERGENCE_RATE_SCALING)*average_error_tolerance]

def get_updates(xs, integration_point_errors, gap_xs, gap_errors, trapz_est_error, target_uncertainty, residule_method=True):
    n_gaps = len(gap_xs)
    gap_error_pts = zip(gap_errors, gap_xs ["gap"]*n_gaps)
    pts_errors = zip(integration_point_errors, xs)

    combined_pts_errors = sorted( gap_error_pts + pts_errors )[::-1]
    residule_error = trapz_est_error - target_uncertainty
    if residule_method:
        largest_error_pts = reduce_error_on_residule_error(combined_pts_errors, residule_error)
    else:
        largest_error_pts = reduce_error_on_average_error_tolerance(combined_pts_errors, target_uncertainty)

    is_gap = lambda x:x[-1] == "gap"
    largest_error_pts = sorted(largest_error_pts, key=is_gap)
    grouped_errors = []
    for _, values in groupby(largest_error_pts, is_gap):
        grouped_errors.append(list(values))

    new_pts, update_xs = grouped_errors
    return new_pts, update_xs

def parse_args():
    argparser = config_argparse()
    ###argparser.add_argument()
    args = argparser.parse_args()
    figure_name = process_plot_argument(args)
    data = parse_user_data(args.data.read())
    return data, figure_name, args.rss, args.sigfigs, args.verbose

def reduce_integration_uncertainty():
    parse_args()
    xs, integration_point_errors, gap_xs, gap_errors, trapz_est_error = integration_error_estimate.main()

if __name__=="__main__":
    reduce_integration_uncertainty()