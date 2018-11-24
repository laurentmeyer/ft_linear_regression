import matplotlib as mpl
mpl.use('TkAgg')
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import utils as utils
import numpy as np
from functools import partial

fig = plt.figure()
ax = fig.gca(projection='3d')

data = utils.get_csv('data.csv')
data = np.transpose(data)
km = data[0]
price = data[1]

m = 40
X = np.linspace(5000, 20000, num=m)
Y = np.linspace(-0.1, 0.0, num=m)
X, Y = np.meshgrid(X, Y)
cost_fun = np.vectorize(partial(utils.cost, data))
Z = cost_fun(X, Y)
# print(data)
# print(Z)
surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=mpl.cm.coolwarm,
                       linewidth=0, antialiased=False)
fig.colorbar(surf, shrink=0.5, aspect=10)


plt.show()
