import matplotlib as mpl
mpl.use('TkAgg')
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import utils as utils
from train import cost
import numpy as np
from functools import partial

fig = plt.figure()
ax = fig.gca(projection='3d')

data = utils.get_csv('data.csv')
data = np.transpose(data)
km = data[0]
price = data[1]

m = 40
theta0 = np.linspace(7000, 9000, num=m)
theta1 = np.linspace(-0.05, 0.0, num=m)
theta0, theta1 = np.meshgrid(theta0, theta1)
def partial_cost(theta0, theta1):
    return cost(km, price, theta0, theta1)
cost_fun = np.vectorize(partial_cost)
Z = cost_fun(theta0, theta1)
surf = ax.plot_surface(theta0, theta1, Z, rstride=1, cstride=1, cmap=mpl.cm.coolwarm,
                       linewidth=0, antialiased=False)
fig.colorbar(surf, shrink=0.5, aspect=10)
plt.show()