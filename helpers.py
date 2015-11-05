import numpy as np

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
    x1, y1, _ = pt1
    x2, y2, _ = pt2
    # gradient of line 1
    m = (y2 - y1)/float(x2 - x1)
    # y-intercept of line1
    c = (x1*y2 - x2*y1)/float(x1 - x2)
    # intersection with x_c
    yi = m*x_c + c
    return yi

def second_derivative(pts):
    (x1, y1, e1), (x2, y2, e2), (x3, y3, e3) = pts
    d32 = float(x3-x2)
    d21 = float(x2-x1)
    d31_over_2 = float(x3-x1)/2.

    second_der = ((y3-y2)/d32 - (y2-y1)/d21)/d31_over_2
    error = np.sqrt(np.sum(np.square([e3/d32, e2/d32, e2/d21, e1/d21])))/d31_over_2
    return second_der, error

def parse_user_data(data):
    # first attempt to get xs, ys and errors
    try:
        xs, ys, es = zip(*[map(float, l.split()[:3]) for l in data.splitlines() if l and not l.startswith("#")])
    except:
        # now try just xs and ys with errors set to 0.0
        xs, ys, es = zip(*[map(l.split()[:2]) + [0.0] for l in data.splitlines() if l and not l.startswith("#")])
    return xs, ys, es

if __name__=="__main__":
    xs = [0, 1, 2]
    ys = [0, 10, 0]
    es = [0.5, 0.5, 0.5]
    pts = zip(xs, ys, es)
    print second_derivative(pts)
    samples = []
    for i in range(10000):
        es = np.random.normal(0, 0.5, 3).transpose()
        pts = zip(xs, np.array(ys) + es, es)
        samples.append(second_derivative(pts)[0])
    print len(samples)
    print np.mean(samples)
    print np.std(samples)