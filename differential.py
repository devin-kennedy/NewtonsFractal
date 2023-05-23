import numpy as np
from matplotlib import pyplot as plt
import math
from functools import cache
from time import thread_time
from scipy.optimize import root

dt = 10 ** -3
x0_pred = 100
x0_prey = 100


def main():
    # valsx = list(range(n))
    # valsy = [x(i) for i in range(n)]
    #
    # plt.plot(valsx, valsy)
    # plt.show()


    # preys, preds = x_backwards(n)
    # preys = list(preys)
    # preds = list(preds)
    #
    # valsx = list(range(len(preys)))
    #
    # plt.plot(valsx, preys)
    # plt.plot(valsx, preds)
    # plt.show()

    # i_s = 100
    # r = 3.59
    # x0 = 0.5
    # valsx = list(range(i_s))
    # plt.plot(valsx, logisticMap(x0, r, i_s))
    # plt.xlabel("x")
    # plt.ylabel("y")
    # plt.title(f"r = {r}   x0 = {x0}")
    # plt.show()

    n = 15
    prec = 0.01
    maps = {}
    for r in list(np.arange(2.0, 4.0, 0.01)):
        map = logisticMap(0.3, r, 150)
        maps[r] = map[-n:]

    for r, map in list(maps.items()):
        plt.scatter([r for _ in range(len(map))], map, c="blue", s=0.1)

    plt.title(f"n = {n}, ")
    plt.xlabel("r value")
    plt.ylabel("")
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


def logisticMap(x0, r, iters):
    xs = [x0]

    for i in range(1, iters):
        xs.append(f_logmap(xs[i-1], r))

    return xs


def f_logmap(x, r):
    return r * x * (1 - x)


if __name__ == "__main__":
    main()
