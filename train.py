import numpy as np
import utils as utils
from plot_line import plot_line

def cost(X, y, theta0, theta1):
    return sum((theta0 + theta1 * X - y) ** 2) / (2 * y.size)

def deriv_cost_theta0(X, y, theta0, theta1):
    return sum(theta0 + theta1 * X - y) / y.size

def deriv_cost_theta1(X, y, theta0, theta1):
    return sum((theta0 + theta1 * X - y) * X) / y.size

def scale(A):
    minA = min(A)
    maxA = max(A)
    meanA = sum(A) / A.size
    return (A - meanA) / (maxA - minA)

def unscale(X, y, theta0, theta1):
    ax = 1. / (max(X) - min (X))
    ay = 1. / (max(y) - min (y))
    bx = -sum(X) / (X.size * (max(X) - min (X)))
    by = -sum(y) / (y.size * (max(y) - min (y)))
    t0 = (theta0 + bx * theta1 - by) / ay
    t1 = (ax * theta1) / ay
    return t0, t1

def train_array(X, y, alpha, iterations):
    scaled_X = scale(X)
    scaled_y = scale(y)
    theta = np.zeros([2, iterations + 1])
    theta[0][0] = 0.
    theta[1][0] = 0.
    i = 1
    while i < iterations + 1:
        prev0 = theta[0][i - 1]
        prev1 = theta[1][i - 1]
        theta[0][i] = prev0 - alpha * deriv_cost_theta0(scaled_X, scaled_y, prev0, prev1)
        theta[1][i] = prev1 - alpha * deriv_cost_theta1(scaled_X, scaled_y, prev0, prev1)
        i += 1
    t0, t1 = unscale(X, y, theta[0], theta[1])
    return ([t0, t1])

def train(X, y, alpha=1e-3, iterations=10000):
    theta = train_array(X, y, alpha, iterations)
    return theta[0][iterations], theta[1][iterations]

def main():
    data = np.transpose(utils.get_csv('data.csv'))
    X = data[0]
    y = data[1]
    theta0, theta1 = train(X, y)
    plot_line(data, theta0, theta1)
    utils.put_csv('parameters.csv', theta0, theta1)

if (__name__ == "__main__"):
	main()
