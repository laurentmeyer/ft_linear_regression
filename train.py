import numpy as np
import utils as utils
from plot_line import plot_line

def cost(data, h):
    X = data[0]
    y = data[1]
    N = X.size
    return sum((h(X) - y) ** 2) / (2 * N)

def deriv_cost_theta0(data, h):
    X = data[0]
    y = data[1]
    N = X.size
    return sum(h(X) - y) / N

def deriv_cost_theta1(data, h):
    X = data[0]
    y = data[1]
    N = X.size
    return sum((h(X) - y) * X) / N

def train(data, alpha=1e-7, iterations=10000):
    theta = np.empty([2, iterations + 1])
    theta[0][0] = 0.
    theta[1][0] = 0.
    i = 1
    while i < iterations + 1:
        prev0 = theta[0][i - 1]
        prev1 = theta[1][i - 1]
        h = lambda x: prev0 + x * prev1
        theta[0][i] = prev0 - alpha * deriv_cost_theta0(data, h)
        theta[1][i] = prev1 - alpha * deriv_cost_theta1(data, h)
        i += 1
    utils.put_csv('parameters.csv', theta[0][iterations], theta[1][iterations])
    return theta

# def loss_per_iteration(data, t):
#     [cost(data, )]

def main():
    data = np.transpose(utils.get_csv('data.csv'))
    print("data:\n{}".format(data))
    parameters = utils.get_csv('parameters.csv')
    theta0 = parameters[0]
    theta1 = parameters[1]
    print("cost:\n{}".format(cost(data, lambda x: theta0 + x * theta1)))
    print("train:\n{}".format(np.transpose(train(data))))
    # print("train:\n{}".format(train(data)))
    plot_line(data, theta0, theta1)

if (__name__ == "__main__"):
	main()
