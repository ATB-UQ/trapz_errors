import sys
import argparse
import numpy as np
from helpers import round_sigfigs, rss, calc_y_intersection_pt, second_derivative_with_uncertainty, parse_user_data
from config import DEFAULT_FIGURE_NAME


DO_NOT_PLOT = "DO_NOT_PLOT"

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

def interval_errors(xs, ys, es, forward=True):
    '''
    Based on analytical Trapezoidal error function with 2nd derivative estimated numerically:
    https://en.wikipedia.org/wiki/Trapezoidal_rule#Error_analysis
    '''
    pts = zip(xs, ys, es)

    gap_xs = [ (xs[0] + xs[1])/2. ]
    gap_ys = [ calc_y_intersection_pt(pts[0], pts[1], gap_xs[0]) ]
    # if there are only 2 points, the interval error will be zero
    if len(pts) == 2:
        return gap_xs, gap_ys, [0]
    gap_es = [ trapz_interval_error(pts[:3], (xs[1] - xs[0])) ]

    for i in range(len(xs)-3):
        gap_xs.append( (xs[i+1] + xs[i+2])/2. )
        gap_ys.append( calc_y_intersection_pt(pts[i+1], pts[i+2], gap_xs[i+1]) )
        dx = xs[i+2] - xs[i+1]
        if forward:
            gap_es.append( trapz_interval_error(pts[i:i+3], dx) )
        else:
            # reverse
            gap_es.append( trapz_interval_error(pts[i+1:i+4], dx) )
        #print gap_es[-1]

    gap_xs.append( (xs[-1] + xs[-2])/2. )
    gap_ys.append( calc_y_intersection_pt(pts[-2], pts[-1], gap_xs[-1]) )
    gap_es.append( trapz_interval_error(pts[-3:], (xs[-1] - xs[-2])) )

    return gap_xs, gap_ys, gap_es

def trapz_interval_error(pts, dx):
    #second_der, error = second_derivative_with_uncertainty(pts)
    second_der, _ = second_derivative_with_uncertainty(pts)
    return (dx**3)/12.*np.array(second_der)

def plot_error_analysis(xs, ys, es, gap_xs, gap_ys, gap_errors, figure_name=None, title="", show=True, x_label="x", y_label="y"):
    import os
    if not os.environ.has_key("DISPLAY"):
        import matplotlib
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    fig.hold(True)
    ax.errorbar(xs, ys, es, marker="o", label="Integration Points")
    ax.errorbar(gap_xs, gap_ys, 12.*np.array(gap_errors), linestyle="", label="Relative Interval Errors")
    plt.ylabel(y_label, fontweight="bold")
    plt.xlabel(x_label, fontweight="bold")
    plt.title(title, fontweight="bold")
    plt.legend(loc = 'upper right', prop={'size':11}, numpoints = 1, frameon = False)
    fig.tight_layout()
    if figure_name:
        plt.savefig(figure_name, format="svg")
    if show:
        plt.show()

def trapz_integrate_with_uncertainty(xs, ys, es, be_conservative=True):
    integration_point_errors = point_error_calc(xs, es)
    forward_results = interval_errors(xs, ys, es, forward=True)
    backwards_results = interval_errors(xs, ys, es, forward=False)
    gap_xs, gap_ys, gap_errors = sorted([forward_results, backwards_results], key=lambda x:np.abs(np.sum(x[-1])))[-1]
    point_uncertainty_error = rss(list(integration_point_errors))
    sum_gap_errors = np.abs(np.sum(gap_errors))

    max_interval_error = np.max(np.abs(np.concatenate((forward_results[-1], backwards_results[-1]))))
    forward_reverse_diff = np.abs(np.abs(np.sum(forward_results[-1])) - np.abs(np.sum(backwards_results[-1])))

    error_safety_policy = max(max_interval_error, forward_reverse_diff)

    total_error = point_uncertainty_error + sum_gap_errors + (error_safety_policy if be_conservative else 0.0)

    return np.trapz(ys, xs), total_error, gap_xs, gap_ys, gap_errors, integration_point_errors, error_safety_policy

def config_argparse():
    argparser = argparse.ArgumentParser(description='Integration Error Estimate')
    argparser.add_argument('-d', '--data', nargs='?', type=argparse.FileType('r'), default=sys.stdin,
                        help="File containing data to integrate. Line format:<x> <y> [<error>]. Data can also be provided via stdin within argument.")
    argparser.add_argument('-p', '--plot', nargs='?', type=str, default=DO_NOT_PLOT,
                        help="Show plot of integration errors, requires matplotlib. Optional argument will determine where and in what format the figure will be saved in.")
    argparser.add_argument('-s', '--sigfigs', type=int, default=3,
                        help="Number of significant figures in output. Default=3")
    argparser.add_argument('-v', '--verbose', action="store_true",
                        help="Output error breakdown.")
    argparser.add_argument('-b', '--be_conservative', action="store_true",
                        help="Make a conservative estimate of the uncertainty.")
    return argparser

def process_plot_argument(args):
    figure_name = DEFAULT_FIGURE_NAME if args.plot is None else args.plot
    figure_name = False if figure_name == DO_NOT_PLOT else figure_name
    figure_name = "{0}.png".format(figure_name) if (figure_name and "." not in figure_name) else figure_name
    return figure_name

def parse_args():
    argparser = config_argparse()
    args = argparser.parse_args()
    figure_name = process_plot_argument(args)
    data = parse_user_data(args.data.read())
    return data, figure_name, args.sigfigs, args.verbose, args.be_conservative

def main():
    data, figure_name, sigfigs, verbose, be_conservative = parse_args()
    xs, ys, es = np.array(data)

    integral, total_error, gap_xs, gap_ys, gap_errors, integration_point_errors, conservative_error_adjustment = trapz_integrate_with_uncertainty(xs, ys, es, be_conservative=be_conservative)

    round_sf = lambda x:round_sigfigs(x, sigfigs)
    result_string = "{0:g} +/- {1:g}".format(round_sf(integral), round_sf(total_error))
    if verbose:
        value_error = round_sf(rss(integration_point_errors))
        print "Error from y-value uncertainty: +/- {0:g}".format(value_error)
        truncation_error = round_sf(np.sum(gap_errors) + conservative_error_adjustment if be_conservative else np.sum(gap_errors))
        print "Estimated truncation error: {1:g}".format(truncation_error)
        if be_conservative:
            print "Additional error component: {0:g}".format(round_sf(conservative_error_adjustment))
        print "Integral: {0}".format(result_string)
    else:
        print result_string

    if figure_name:
        plot_error_analysis(xs, ys, es, gap_xs, gap_ys, gap_errors, figure_name, title="Integral: {0}".format(result_string), show=True)

if __name__=="__main__":
    main()
