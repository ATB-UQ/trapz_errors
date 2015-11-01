import numpy as np
CONVERGENCE_RATE_SCALING = 1

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

    # (xs, es)
    new_pts = [pt[1] for pt in largest_error_pts if pt[-1] == "gap"]
    update_xs = [pt[1] for pt in largest_error_pts if pt[-1] != "gap"]

    print "Intervals to recieve new points: {0:.1f}%".format(len(new_pts)/float(len(gap_xs))*100)
    print "Update points: {0:.1f}%".format(len(update_xs)/float(len(xs))*100)

    return new_pts, update_xs

def reduce_integration_uncertainty():
    pass