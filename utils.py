import numpy as np
from functools import reduce

def get_csv(filename):
    try:
        parameters = np.genfromtxt(
            filename, skip_header=1, delimiter=',')
    except OSError:
        parameters = [0., 0.]
    return parameters

def put_csv(filename, theta0, theta1):
    f = open(filename, 'w')
    f.write("theta0,theta1\n{},{}".format(theta0, theta1))
    f.close()

def get_prediction_fun(filename):
    parameters = get_csv(filename)
    theta0 = parameters[0]
    theta1 = parameters[1]
    return lambda x: theta0 + theta1 * x
