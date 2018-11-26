import numpy as np
import utils as utils
import functools as ft
import train as t
from plot_line import plot_line

data = np.transpose(utils.get_csv('data.csv'))
X = data[0]
y = data[1]

sample_size = .1
mask = np.random.sample(y.size)
while (((mask < sample_size) == True).size == 0):
    mask = np.random.sample(y.size)
train = mask >= sample_size
test = mask < sample_size

theta0, theta1 = t.train(X[train], y[train])
estimation = theta0 + theta1 * X[test]
error = abs((estimation - y[test]) / y[test])
error = sum(error) / error.size
print("sample size = {}".format(y[train].size))
print("test size = {}".format(y[test].size))
print("error = {:2f}%".format(error * 100))