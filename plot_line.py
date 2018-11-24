import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import utils as utils
from functools import partial

def plot_line(data, theta0, theta1):
    km = data[0]
    price = data[1]
    fig, ax = plt.subplots()
    fig.suptitle('Scatter plot')
    plt.scatter(km, price, s=10)
    x = np.linspace(min(km), max(km), 100)
    f = lambda x : theta0 + x * theta1
    plt.plot(x, f(x), 'r')
    plt.xlabel('km')
    plt.ylabel('price')
    plt.show()
