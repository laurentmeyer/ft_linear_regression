import numpy as np
import utils as utils

try:
    m = int(input("mileage: "))
    parameters = utils.get_csv('parameters.csv')
    theta0 = parameters[0]
    theta1 = parameters[1]
    prediction = theta0 + m * theta1
    print("estimate: {:.2f}".format(prediction))
except ValueError:
    print("Please enter valid int input")