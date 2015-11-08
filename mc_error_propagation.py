import numpy as np
CONV_TOL = 0.01
MIN_ITER = 20

def mc_error_estimate(xs, ys, es, f, n):
    expectation_value = f(xs, ys, es)[1]
    es_sampled = sample_y(es, n)
    trial_results = [f(xs, np.array(ys) + es_s, es_s)[1] for es_s in es_sampled]
    return np.mean(trial_results), np.std(trial_results), converged(trial_results, expectation_value)

def converged(trial_results, expectation_value, conv_tol=CONV_TOL, min_iteration=MIN_ITER):
    if abs(np.mean(trial_results) - expectation_value) > conv_tol:
        return False
    if abs(np.std(trial_results) - np.std(trial_results[:-min_iteration])) > conv_tol:
        return False
    return True

def sample_y(es, n):
    y_samples = [np.random.normal(0, sig, n) for sig in es]
    return np.array(y_samples).transpose()