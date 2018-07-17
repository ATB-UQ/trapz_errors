-----------------------------------
Error Analysis for Trapezoidal Rule Applied to Uncertain Data
-----------------------------------

A tool to analyse errors associated with applying the Trapezoidal rule to uncertain data.

Author: Martin Stroet (University of Queensland)

---------------------
Requirements
----------------------

    Python 2.7
    numpy

Optional

    matplotlib (show plots)

-------------------------------------------------------------------------------
Usage: Calculate integration error considering y-value uncertainty and truncation error.
-------------------------------------------------------------------------------
    python calculate_error.py [-h] -d DATA [-p [PLOT]] [-s SIGFIGS] [-v] [-c]

    arguments:
      -h, --help            show this help message and exit
      -d DATA, --data DATA  File containing data to integrate. Lines are read as
                            whitespace separated values of the form: <x> <y>
                            [<y_error>].
      -p [PLOT], --plot [PLOT]
                            Show plot of integration errors, requires matplotlib.
                            Optional argument will determine where and in what
                            format the figure will be saved in.
      -s SIGFIGS, --sigfigs SIGFIGS
                            Number of significant figures in output. Default=3
      -v, --verbose         Output error breakdown.
      -c, --conservative    Make a conservative estimate of the total truncation
                            error; add the maximum interval error to the sum of
                            all interval errors.


-------------------------------------------------------------------------------
Usage: Identify largest sources of uncertainty.
-------------------------------------------------------------------------------

    python reduce_error.py [-h] -d DATA [-p [PLOT]] [-s SIGFIGS] [-v] [-c] -t
                           TARGET_ERROR [-r CONVERGENCE_RATE_SCALING]

    arguments:
      -h, --help            show this help message and exit
      -d DATA, --data DATA  File containing data to integrate. Lines are read as
                            whitespace separated values of the form: <x> <y>
                            [<y_error>].
      -p [PLOT], --plot [PLOT]
                            Show plot of integration errors, requires matplotlib.
                            Optional argument will determine where and in what
                            format the figure will be saved in.
      -s SIGFIGS, --sigfigs SIGFIGS
                            Number of significant figures in output. Default=3
      -v, --verbose         Output error breakdown.
      -c, --conservative    Make a conservative estimate of the total truncation
                            error; add the maximum interval error to the sum of
                            all interval errors.
      -t TARGET_ERROR, --target_error TARGET_ERROR
                            Target error of integration, used to determine how to
                            reduce integration uncertainty.
      -r CONVERGENCE_RATE_SCALING, --convergence_rate_scaling CONVERGENCE_RATE_SCALING
                            Determines the rate of convergence to the target error
                            i.e. how many iterations will be required to reach
                            target error. A value < 1 will result in more
                            iterations but fewer overall points as the points will
                            be concentrated in regions of high uncertainty;
                            conversely values > 1 will result it more points but
                            fewer iterations required to reach a given target
                            error. Default=1.

----------------------------
Example
----------------------------

For an example of how to use this tool see example/example.sh