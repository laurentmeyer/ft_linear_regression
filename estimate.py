#!/usr/local/bin/python3
import numpy as np
import utils as utils

def deriv_cost_theta0(h, data):
    m = data[0].size
    return sum([h(data[0][i]) - data[1][i] for i in range(m)]) / m

def deriv_cost_theta1(h, data):
    m = data[0].size
    return sum([(h(data[0][i]) - data[1][i]) * (data[0][i]) for i in range(m)]) / m

def train(data, alpha=0.1, iterations=10):
    # print(data)
    theta0 = 0.
    theta1 = 0.
    i = 0
    # while i < 1:
    h = lambda x: theta0 + x * theta1
    while i < iterations:
        # print(utils.cost(data, theta0, theta1))
        theta0 = theta0 - alpha * deriv_cost_theta0(h, data)
        # print(theta0)
        theta1 = theta1 - alpha * deriv_cost_theta1(h, data)
        # print(theta1)
        i += 1
    # print (theta0)
    # print (theta1)
        

def main():
    data = utils.get_csv('data.csv')
    data = np.transpose(data)
    parameters = utils.get_csv('parameters.csv')
    theta0 = parameters[0]
    theta1 = parameters[1]
    print (utils.cost(data, theta0, theta1))

    train(data)


if (__name__ == "__main__"):
	main()
