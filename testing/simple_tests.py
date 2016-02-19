from scipy.integrate import trapz
import numpy as np
from scipy import interpolate
import sys
sys.path.append("../")
from reduce_integration_uncertainty import reduce_error_on_residual_error
from method_validation import TARGET_ERROR
import simulated_iterative_integration
from integration_error_estimate import trapz_integrate_with_uncertainty, interval_errors, point_error_calc
from helpers import rss
from method_validation import DATA_FILE, get_data, parse_data_dir, get_realistic_function

def integrate_with_point_uncertinaty(xs, ys, es):
    integration_error_per_point = point_error_calc(xs, es)
    return np.trapz(ys, xs), rss(integration_error_per_point)

def integrate_with_gap_uncertinaty(xs, ys, es):
    _, _, gap_es = interval_errors(xs, ys, es)
    _, _, gap_es_reverse = interval_errors(xs, ys, es, forward=False)
    gap_errors = sorted([gap_es, gap_es_reverse], key=lambda x:np.abs(np.sum(x)))[-1]
    truncation_error = np.abs(np.sum(gap_errors)) + np.abs(np.abs(np.sum(gap_es)) - np.abs(np.sum(gap_es_reverse)))
    return np.trapz(ys, xs), truncation_error

def run_test(xs, ys, es, x_fine=None, y_fine=None):
    if x_fine is not None:
        integral = trapz(y_fine, x_fine)
        print "Integral: {0}".format(integral)
        print "True truncation error: {0}".format(trapz(ys, xs) - integral)

    print "Truncation error estimate: {0} +/- {1}".format(*integrate_with_gap_uncertinaty(xs, ys, es))
    print "Analytical integral point uncertainty: {0} +/- {1}".format(*integrate_with_point_uncertinaty(xs, ys, es))
    trapz_integral, total_error, gap_xs, _, gap_errors, _, _ = trapz_integrate_with_uncertainty(xs, ys, es)
    print "Combined error estimate: {0} +/- {1}".format(trapz_integral, total_error)
    #plot_error_analysis(xs, ys, es, gap_xs, gap_ys, zip(*gap_errors)[0])
    plot_data(xs, ys, es, x_fine, y_fine, figure_name="example_integration_{0}.eps".format(len(xs)))
    #plot_data_error_propagation(xs, ys, es, x_fine, figure_name="error_prop.png")
    return gap_errors, gap_xs, total_error

def plot_data(xs, ys, es, x_fine=None, y_fine=None, figure_name=None, title=""):
    import os
    if not os.environ.has_key("DISPLAY"):
        import matplotlib
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(5,4))
    ax = fig.add_subplot(111)
    fig.hold(True)
    f = interpolate.interp1d(xs, ys, kind=1)
    if x_fine is not None and y_fine is not None:
        ax.fill_between(x_fine, y_fine, map(f, x_fine), facecolor='red', alpha=0.8)
        ax.plot(x_fine, y_fine, "r")
    ax.set_title("N={0}".format(len(xs)), fontweight="bold")
    ax.errorbar(xs, ys, es, marker="o", label="Integration Points")
    ax.set_xlabel(r'$\mathbf{\lambda}$')
    ax.set_ylabel(r'<$\mathbf{dV/d\lambda}$> (kJ/mol)')
    #ax.set_title("Measurement Error Propagation", fontweight="bold")

    #plt.legend(loc = 'upper right', prop={'size':11}, numpoints = 1, frameon = False)
    fig.tight_layout()
    if figure_name:
        plt.savefig("{0}".format(figure_name), dpi=300)
    plt.show()

def plot_data_error_propagation(xs, ys, es, x_fine, figure_name=None, title=""):

    import os
    if not os.environ.has_key("DISPLAY"):
        import matplotlib
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(5,4))
    ax = fig.add_subplot(111)
    fig.hold(True)
    ax.errorbar(xs, ys, es, marker="o", label="Integration Points")
    ax.set_xlabel(r'$\mathbf{\lambda}$')
    ax.set_ylabel(r'<$\mathbf{dV/d\lambda}$> (kJ/mol)')
    #ax.set_title("Measurement Error Propagation", fontweight="bold")
    #ax.fill_between(xs, np.zeros(len(xs)), ys, facecolor='blue', alpha=0.2)
    for i in range(1,len(xs)-1):
        ax.fill_between(xs[i-1:i+2], ys[i-1:i+2], [ys[i-1], ys[i]+es[i], ys[i+1]], facecolor='red', alpha=0.2)

    #plt.legend(loc = 'upper right', prop={'size':11}, numpoints = 1, frameon = False)
    fig.tight_layout()
    if figure_name:
        plt.savefig("{0}".format(figure_name), dpi=300)
    plt.show()

def plot_data_tapz_simps(xs, ys, es, x_fine, figure_name=None, title=""):
    #mc_integral, mc_error = mc_simps(xs, ys, es)
    #print "Monte Carlo Simps integral point uncertainty: {0} +/- {1}".format(mc_integral, mc_error)
    import os
    if not os.environ.has_key("DISPLAY"):
        import matplotlib
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(9,4))
    ax = fig.add_subplot(121)
    fig.hold(True)
    ax.errorbar(xs, ys, es, marker="o", label="Integration Points")
    ax.set_xlabel(r'$\mathbf{\lambda}$')
    ax.set_ylabel(r'<$\mathbf{dV/d\lambda}$> (kJ/mol)')
    ax.set_title("Linear interpolation", fontweight="bold")
    ax.fill_between(xs, np.zeros(len(xs)), ys, facecolor='blue', alpha=0.2)
    ax3 = fig.add_subplot(122, sharey=ax)
    ax3.set_xlim((0,1))
    ax3.errorbar(xs, ys, es, marker="o", linestyle="", label="Integration Points")
    f = interpolate.interp1d(xs, ys, kind=3)
    ax3.plot(x_fine, map(f, x_fine), "b", label="Integration Points")
    ax3.fill_between(x_fine, np.zeros(len(x_fine)), map(f, x_fine), facecolor='blue', alpha=0.2)
    ax3.get_yaxis().set_visible(False)
    ax3.set_title("Cubic interpolation", fontweight="bold")
    
    ax3.set_xlabel(r'$\mathbf{\lambda}$')
    #plt.legend(loc = 'upper right', prop={'size':11}, numpoints = 1, frameon = False)
    fig.tight_layout()
    if figure_name:
        plt.savefig("{0}".format(figure_name), dpi=300)
    plt.show()
    
def simple_test_function():
    xs = [0, 1, 2]
    ys = [0, 10, 0]
    es = [0.5, 0.5, 0.5]
    run_test(xs, ys, es)

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
    run_test(xs, ys, es, x_fine, y_fine)

def interpolated_test_function():
    N = 11
    a = 0
    b = 1
    f = simulated_iterative_integration.get_realistic_function()
    xs = list(np.linspace(a, b, N))
    es = np.random.uniform(0, 10, N)
    ys = map(f, xs)

    x_fine = np.linspace(a, b, N*100)
    y_fine = map(f, x_fine)
    
    run_test(xs, ys, es, x_fine, y_fine)

def real_function():
    N = 5
    a = 0
    b = 1
    data = get_data(DATA_FILE, parse_data_dir)
    for _, dvdl in data.items():
        f = get_realistic_function(dvdl[0], dvdl[1])
        break
    xs = list(np.linspace(a, b, N))
    es = np.random.uniform(0, 5, N)
    ys = map(f, xs)

    x_fine = np.linspace(a, b, N*100)
    y_fine = map(f, x_fine)

    #generated_pts = [xs, ys, list(np.zeros(len(xs)))]
    generated_pts = [xs, ys, es]
    for _ in range(6):
        gap_errors, gap_xs, total_error = run_test(*(generated_pts + [x_fine, y_fine]))
        gap_error_pts = zip(gap_errors, gap_xs, ["gap"]*len(gap_errors))
        largest_gap_error = reduce_error_on_residual_error(gap_error_pts, total_error-TARGET_ERROR, 0.5, False)
        if largest_gap_error:
            largest_gap_errors_x = zip(*largest_gap_error)[1]
        else:
            largest_gap_errors_x = []
        generated_pts = zip(*sorted(zip(*generated_pts) + zip(largest_gap_errors_x, map(f, largest_gap_errors_x), list(np.zeros(len(largest_gap_errors_x))) )))

if __name__=="__main__":
    #simple_test_function()
    #periodic_test_function()
    #interpolated_test_function()
    real_function()