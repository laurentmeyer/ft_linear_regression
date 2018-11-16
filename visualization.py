import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

try:
    parameters = np.genfromtxt('parameters.csv', skip_header=1, delimiter=',')
except OSError:
    parameters = [0., 0.]
theta0 = parameters[0]
theta1 = parameters[1]

try:
    data = np.genfromtxt('data.csv', skip_header=1, delimiter=',')
except OSError:
    data = [[0., 0.]]
data = np.transpose(data)
km = data[0]
price = data[1]

fig = plt.figure()
fig.suptitle('Scatter plot')
plt.scatter(km, price, s=10)

min_km = min(km)
max_km = max(km)
x = np.linspace(min_km, max_km, 100)
plt.plot(x, theta0 + theta1 * x, 'r')
plt.xlabel('km')
plt.ylabel('price')
plt.show()