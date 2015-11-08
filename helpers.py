import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

EXAMPLE_DVDL_DATA = '''
# lambda  dvdl_average  error_est
0.000 99.84849 0.36028
0.025 155.28053 0.90581
0.050 186.13209 1.11968
0.075 188.50786 1.42872
0.100 177.67623 1.45245
0.125 149.29522 1.37675
0.150 125.41653 1.41022
0.200 83.82993 0.78686
0.250 63.47751 0.54568
0.300 50.74429 0.35037
0.400 32.61911 0.41166
0.500 17.44440 0.62932
0.600 -2.15131 1.13114
0.649 -16.64929 1.29989
0.700 -32.44893 1.58581
0.750 -71.44813 3.10821
0.775 -94.12352 3.47328
0.800 -110.41922 3.80674
0.825 -127.66568 2.59015
0.850 -113.45883 1.90801
0.875 -97.05920 1.65601
0.900 -71.38330 0.95010
0.950 -30.13460 0.54041
1.000 3.15286 0.21252'''

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
    h0 = x2-x1
    h1 = x3-x2
    a = 2./(h0*(h0+h1))
    b = -2./(h0*h1)
    c = 2./(h1*(h0+h1))
    return a*y1 + b*y2 + c*y3, rss([a*e1, b*e2, c*e3])

def parse_user_data(data):
    # first attempt to get xs, ys and errors
    try:
        xs, ys, es = zip(*[map(float, l.split()[:3]) for l in data.splitlines() if l and not l.startswith("#")])
    except:
        # now try just xs and ys with errors set to 0.0
        xs, ys, es = zip(*[map(l.split()[:2]) + [0.0] for l in data.splitlines() if l and not l.startswith("#")])
    return xs, ys, es

def sum_2nd_derivertives(pts):
    ders = []
    for i in range(len(pts)-2):
        ders.append(second_derivative(pts[i:i+3]))
    ders = np.array(ders).transpose()
    return np.sum(ders[0]), rss(ders[1])

def test_correlation_dependence(n_max, f):
    sigma = 0.1
    #xs = [0., 1., 2., 3., 4, 5, 6]
    #ys = [0., 1., 4., 9, 16, 9, 4]
    analytical_err = []
    calc_err = []
    test_range = range(3, n_max+1)
    for i in test_range:
        xs = np.linspace(0, 1, i)
        #ys = 10*np.exp(-(xs-i/2.)**2/(i/4.)**2)
        ys = [f(x) for x in xs]
        es = [sigma]* len(xs)
        pts = zip(xs, ys, es)
        analytical_err.append( sum_2nd_derivertives(pts))
        samples = []
        for _ in range(2000):
            es = np.random.normal(0, sigma, len(xs)).transpose()
            pts = zip(xs, np.array(ys) + es, es)
            samples.append(sum_2nd_derivertives(pts))
        samples = np.array(samples).transpose()
        calc_err.append([np.mean(samples[0]), np.std(samples[0])])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    fig.hold(True)
    ax.errorbar(xs, ys, es, marker="o", label="analytical value - calculated value")
    plt.show()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    fig.hold(True)
    ax.plot(test_range, np.array(zip(*analytical_err)[0]) - np.array(zip(*calc_err)[0]), marker="o", label="analytical value - calculated value")
    ax.plot(test_range, np.array(zip(*analytical_err)[1])/np.array(zip(*calc_err)[1]), marker="o", label="analytical error - calculated error")
    #ax.plot(test_range, , marker="o", label="calculated error")
    plt.legend()
    plt.show()

def filter_(xs, ys):
    initPts = [x/10. for x in range(0,11,2)]
    newxs = []; newys = [];
    for i, x in enumerate(xs):
        if x in initPts:
            newxs.append(x)
            newys.append(ys[i])
    return newxs, newys

def get_realistic_function():
    xs, ys, _ = parse_user_data(EXAMPLE_DVDL_DATA)
    xs, ys = filter_(xs, ys)
    ys = [y for y in ys]
    f = interpolate.interp1d(xs, ys, kind=2)
    return f

def single_test(f):
    sigma = 0.5
    N = 4
    #xs = [0., 1., 2., 3., 4, 5, 6]
    #ys = [0., 1., 4., 9, 16, 9, 4]
    xs = np.linspace(0,1, N)
    ys = [f(x) for x in xs]
    es = [sigma]* len(xs)
    pts = zip(xs, ys, es)
    print sum_2nd_derivertives(pts)
    samples = []
    for i in range(20000):
        es = np.random.normal(0, sigma, len(xs)).transpose()
        pts = zip(xs, np.array(ys) + es, es)
        samples.append(sum_2nd_derivertives(pts))
    samples = np.array(samples).transpose()
    print np.mean(samples[0])
    print np.std(samples[0])
if __name__=="__main__":
    f = get_realistic_function()
    single_test(f)
    test_correlation_dependence(30, f)
    