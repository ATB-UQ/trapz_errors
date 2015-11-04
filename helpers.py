import numpy as np
from itertools import groupby
from config import IMBALANCED_2ND_DERIVATIVE_TOL

def round_sigfigs(num, sig_figs):
    """Round to specified number of sigfigs.

    >>> round_sigfigs(0, sig_figs=4)
    0
    >>> int(round_sigfigs(12345, sig_figs=2))
    12000
    >>> int(round_sigfigs(-12345, sig_figs=2))
    -12000
    >>> int(round_sigfigs(1, sig_figs=2))
    1
    >>> '{0:.3}'.format(round_sigfigs(3.1415, sig_figs=2))
    '3.1'
    >>> '{0:.3}'.format(round_sigfigs(-3.1415, sig_figs=2))
    '-3.1'
    >>> '{0:.5}'.format(round_sigfigs(0.00098765, sig_figs=2))
    '0.00099'
    >>> '{0:.6}'.format(round_sigfigs(0.00098765, sig_figs=3))
    '0.000988'
    """
    if num != 0:
        return round(num, -int(np.floor(np.log10(abs(num))) - (sig_figs - 1)))
    else:
        return 0  # Can't take the log of 0

def rss(values):
    return np.sqrt(np.sum(np.square(values)))

def calc_y_intersection_pt(pt1, pt2, x_c):
    '''
    Return the point at the intersection of the linear function defined by pt1,pt2 and the line x = x_c
    '''
    x1, y1 = pt1
    x2, y2 = pt2
    # gradient of line 1
    m = (y2 - y1)/float(x2 - x1)
    # y-intercept of line1
    c = (x1*y2 - x2*y1)/float(x1 - x2)
    # intersection with x_c
    yi = m*x_c + c
    return yi

def second_derivative(pts):
    (x1, y1), (x2, y2), (x3, y3) = pts
    return ((y3-y2)/(x3-x2) - (y2-y1)/(x2-x1))/((x3-x1)/2.)

def is_2nd_derivative_sign_balanced(sign_imbalance):
    return sign_imbalance <= IMBALANCED_2ND_DERIVATIVE_TOL

def get_2nd_derivative_sign_imbalanced(xs, ys):
    pts = zip(xs, ys)
    second_derivatives = sorted([second_derivative(pts[i:i+3]) for i in range(len(pts)-3)])
    return 100.*abs(np.sum(second_derivatives)/float(np.sum(np.abs(second_derivatives))))

def parse_user_data(data):
    # first attempt to get xs, ys and errors
    try:
        xs, ys, es = zip(*[map(float, l.split()[:3]) for l in data.splitlines() if l and not l.startswith("#")])
    except:
        # now try just xs and ys with errors set to 0.0
        xs, ys, es = zip(*[map(l.split()[:2]) + [0.0] for l in data.splitlines() if l and not l.startswith("#")])
    return xs, ys, es