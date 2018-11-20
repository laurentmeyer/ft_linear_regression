import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import utils as utils
from functools import partial

data = utils.get_csv('data.csv')
data = np.transpose(data)
km = data[0]
price = data[1]

fig = plt.figure()
fig.suptitle('Scatter plot')
ax = plt.subplot(121)
ax.set_aspect(1)
plt.scatter(km, price, s=10)

min_km = min(km)
max_km = max(km)
x = np.linspace(min_km, max_km, 100)
f = utils.get_prediction_fun('parameters.csv')
plt.plot(x, f(x), 'r')
plt.xlabel('km')
plt.ylabel('price')


ax = fig.add_subplot(1, 2, 2, projection='3d')
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
