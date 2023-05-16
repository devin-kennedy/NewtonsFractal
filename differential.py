import numpy as np
from matplotlib import pyplot as plt
import math
from functools import cache
from time import thread_time
from scipy.optimize import root
dt = 0.1
x0_pred = 1000
x0_prey = 1000


def main():
    n = 50000
    # valsx = list(range(n))
    # valsy = [x(i) for i in range(n)]
    #
    # plt.plot(valsx, valsy)
    # plt.show()
    preys, preds = x_backwards(n)
    preys = list(preys)
    preds = list(preds)

    valsx = list(range(len(preys)))

    plt.plot(valsx, preys)
    plt.plot(valsx, preds)
    plt.show()


def f(xi):
    a = -0.25
    b = 0.25
    ga = 0.2
    d = 0.8
    return np.array([
        a*xi[0] - b*xi[0]*xi[1],
        d*xi[0]*xi[1] - ga*xi[1]
    ])


# @cache
# def x(n):
#
#     if n == 0:
#         return x0
#
#     return (dt * f(x(n - 1))) + x(n - 1)


# What a backwards euler method
def x_backwards(n):
    preds = np.zeros(n)
    preds[0] = x0_pred
    preys = np.zeros(n)
    preys[0] = x0_prey

    for i in range(1, n):
        x0 = np.array(
            (preys[i-1], preds[i-1])
        )
        sol = root(
            lambda x: x0 + (dt * f(x)) - x,
            x0=x0,
        )
        preys[i] = sol.x[0]
        preds[i] = sol.x[1]

    return preys, preds


if __name__ == "__main__":
    main()
