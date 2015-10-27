from scipy.integrate import trapz
import numpy as np
from monte_carlo_integration import mc_trapz

def integrate_with_uncertinaty(xs, ys, es, method=trapz):
    # Calculate error estimate for each interval
    integration_means = []
    integration_errors = []
    for i in range(len(xs) - 1):
        localXs = xs[i:i + 2]; localYs = ys[i:i + 2]; localEs = es[i:i + 2]
        mu, error = subinterval_integration(localXs, localYs, localEs, method)
        integration_means.append(mu)
        integration_errors.append(error)

    total_error_est = np.sqrt(np.sum(np.square(integration_errors)))
    #total_integral = np.sum(integration_means)/2.

    return method(ys, xs), total_error_est

def subinterval_integration(localXs, localYs, localEs, method):

    mean_integration = method(localYs, localXs)
    left_bound_integral = method([localYs[0] + localEs[0], localYs[1]             ], localXs)
    right_bound_integral = method([localYs[0]            , localYs[1] + localEs[1]], localXs)
    error_integral = np.sqrt(np.sum(np.square([abs(left_bound_integral - mean_integration), abs(right_bound_integral - mean_integration)])))

    return mean_integration, error_integral

def unfactorized_error_calc(xs, es):
    errors = []
    for i in range(len(xs)-1):
        errors.append(np.sqrt(np.sum([(xs[i+1]*es[i+1])**2, (xs[i+1]*es[i])**2, (xs[i]*es[i+1])**2, (xs[i]*es[i])**2])))
    return 0.5*np.sqrt(np.sum(np.array(errors)**2))

def factorized_error_calc(xs, es):
    errors = []
    for i in range(len(xs)-1):
        errors.append((xs[i+1]-xs[i])*np.sqrt(es[i]**2 + es[i+1]**2))
    return 0.5*np.sqrt(np.sum(np.array(errors)**2))

if __name__=="__main__":
    xs = [0, 1]
    ys = [0, 1]
    es = [0.1, 0.1]
    print trapz(ys, xs)
    print unfactorized_error_calc(xs, es)
    print factorized_error_calc(xs, es)
    print mc_trapz(xs, ys, es)
    print integrate_with_uncertinaty(xs, ys, es)