from scipy.integrate import trapz
import numpy as np

def integrate_with_uncertinaty(xs, ys, es, method=trapz):
    xs, ys, es = add_pseudo_pts(xs, ys, es)
    # Calculate error estimate for each interval
    integration_means = []
    integration_errors = []
    for i in range(len(xs) - 2):
        localXs = xs[i:i + 3]; localYs = ys[i:i + 3]; localEs = es[i:i + 3]
        mu, error = subinterval_integration(localXs, localYs, localEs, method)
        integration_means.append(mu)
        integration_errors.append(error)

    total_error_est = np.sqrt(np.sum(np.square(integration_errors)))
    total_integral = np.sum(integration_means)/2.

    return total_integral, total_error_est

def subinterval_integration(localXs, localYs, localEs, method):

    mean_integration = method(localYs, localXs)
    upper_bound_integral = method([localYs[0], localYs[1] + localEs[1], localYs[2]], localXs)

    return mean_integration, abs(upper_bound_integral - mean_integration)

def add_pseudo_pts(xs, ys, es):
    '''
    Due to the fact that boundary values are never in the middle of an interval
    we add "pseudo" points to the beginning and end of the arrays to force the algorithm to
    calculate a reasonable error for the boundary regions. Pseudo points will have the same y
    value as the boundary and shifted in x such that the boundary is in the center of the 3 point interval.
    '''
    leftPseudoPt = (xs[0] - (xs[1] - xs[0]),
                    ys[0],
                    es[0]
                    )
    rightPseudoPt = (xs[-1] + (xs[-1] - xs[-2]),
                     ys[-1],
                     es[-1]
                     ) 
    xs = [leftPseudoPt[0]] + list(xs) + [rightPseudoPt[0]]
    ys = [leftPseudoPt[1]] + list(ys) + [rightPseudoPt[1]]
    es = [leftPseudoPt[2]] + list(es) + [rightPseudoPt[2]]
    return xs, ys, es