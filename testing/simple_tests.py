from scipy.integrate import trapz
import numpy as np
from simulated_iterative_integration import get_realistic_function
from integration_with_point_uncertainty import integrate_with_point_uncertinaty
from monte_carlo_integration import mc_trapz
from integration_error_estimate import integrate_with_gap_uncertinaty, trapz_integrate_with_uncertainty

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
    simple_test_function()
    periodic_test_function()
    interpolated_test_function()