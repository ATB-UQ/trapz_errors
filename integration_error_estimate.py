import sys
import argparse
import numpy as np
from helpers import round_sigfigs, rss, calc_y_intersection_pt, second_derivative, \
    is_2nd_derivative_sign_balanced, parse_user_data
from config import DEFAULT_FIGURE_NAME, IMBALANCED_2ND_DERIVATIVE_TOL

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
        #print gap_es[-1]

    gap_xs.append( (xs[-1] + xs[-2])/2. )
    gap_ys.append( calc_y_intersection_pt(pts[-2], pts[-1], gap_xs[-1]) )
    gap_es.append( trapz_interval_error(pts[-3:], (xs[-1] - xs[-2])) )

    return gap_xs, gap_ys, gap_es

def trapz_interval_error(pts, dx):
    return (dx**3)/12.*abs(second_derivative(pts))

def four_pt_trapz_interval_error(pts):
    dx = pts[2][0] - pts[1][0]
    return np.max([trapz_interval_error(pts[1:], dx), trapz_interval_error(pts[:-1], dx)])

def plot_error_analysis(xs, ys, es, gap_xs, gap_ys, gap_errors, figure_name=None, title=""):
    import os
    if not os.environ.has_key("DISPLAY"):
        import matplotlib
        matplotlib.use("Agg")
    import pylab as pl
    fig = pl.figure()
    ax = fig.add_subplot(111)
    fig.hold(True)
    ax.errorbar(xs, ys, es, marker="o", label="Integration Points")
    ax.errorbar(gap_xs, gap_ys, 12.*np.array(gap_errors), linestyle="", label="Relative Interval Errors")
    pl.ylabel('y', fontweight="bold")
    pl.xlabel('x', fontweight="bold")
    pl.title(title, fontweight="bold")
    pl.legend(loc = 'upper right', prop={'size':11}, numpoints = 1, frameon = False)
    fig.tight_layout()
    if figure_name:
        pl.savefig("{0}".format(figure_name))
    pl.show()

def trapz_integrate_with_uncertainty(xs, ys, es, use_rss=True):
    integration_point_errors = point_error_calc(xs, es)
    gap_xs, gap_ys, gap_errors = interval_errors(xs, ys)
    if use_rss:
        total_error = rss(integration_point_errors + gap_errors)
        if not is_2nd_derivative_sign_balanced(xs, ys):
            sys.stderr.write("WARNING: The 2nd derivative sign for the data provided is unbalanced, i.e. 100*abs( (n_positive) - (n_negative) )/(n_total) > {0}%. " \
            "This can cause underestimation of total truncation error, re-running with argument '--RSS False' will remove this risk.\n".format(IMBALANCED_2ND_DERIVATIVE_TOL))
    else:
        total_gap_errors = np.sum(gap_errors)
        total_error = rss(integration_point_errors + total_gap_errors)
    return np.trapz(ys, xs), total_error, gap_xs, gap_ys, gap_errors, integration_point_errors

def config_argparse():
    argparser = argparse.ArgumentParser(description='Integration Error Estimate')
    argparser.add_argument('-d', '--data', nargs='?', type=argparse.FileType('r'), default=sys.stdin,
                        help="File containing data to integrate. Line format:<x> <y> [<error>]. Data can also be provided via stdin within argument.")
    argparser.add_argument('--rss',type=bool, default=True,
                        help="Combine interval errors as the Root Sum of Squares. Should be set to false if 2nd derivative is predominantly +ve or -ve.")
    argparser.add_argument('-p', '--plot', nargs='?', type=str, default=DO_NOT_PLOT,
                        help="Show plot of integration errors. Optional argument will determine where and in what format the figure will be saved in.")
    argparser.add_argument('-s', '--sigfigs', type=int, default=3,
                        help="Number of significant figures in output.")
    argparser.add_argument('-v', '--verbose', action="store_true",
                        help="Output error breakdown.")
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
    return data, figure_name, args.rss, args.sigfigs, args.verbose

def run(xs, ys, es, use_rss, verbose, sigfigs):
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
    else:
        print result_string

def main():
    data, figure_name, use_rss, sigfigs, verbose = parse_args()
    xs, ys, es = data
    run()

    if figure_name:
        plot_error_analysis(xs, ys, es, gap_xs, gap_ys, gap_errors, figure_name, title="Integral: {0}".format(result_string))
    return 

if __name__=="__main__":
    main()