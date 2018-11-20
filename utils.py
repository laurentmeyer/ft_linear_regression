import numpy as np
from functools import reduce

def get_csv(filename):
    try:
        parameters = np.genfromtxt(
            filename, skip_header=1, delimiter=',')
    except OSError:
        parameters = [0., 0.]
    return parameters

def get_prediction_fun(filename):
    parameters = get_csv(filename)
    theta0 = parameters[0]
    theta1 = parameters[1]
    return lambda x: theta0 + theta1 * x

def cost(data, theta0, theta1):
    m = data[0].size
    h = lambda x: theta0 + np.multiply(theta1, x)
    return sum(np.power(h(data[0]) - data[1], 2)) / (2 * m)
    # return sum([(h(data[0][x]) - data[1][x]) ** 2 for x in range(m)]) / (2 * m)
