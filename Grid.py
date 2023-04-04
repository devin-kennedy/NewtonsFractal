import math
from math import e, pi, cos, sin, sqrt

import numpy as np
import PIL.Image as img
import random
from functools import cache
from utils import *
import threading


class Grid:
    def __init__(self, xmin: int, xmax: int, ymin: int, ymax: int, res: tuple):
        self.__check_args_grid(xmin, xmax, ymin, ymax, res)

        xs = np.linspace(xmin, xmax, res[0])
        ys = np.linspace(ymin, ymax, res[1])

        self.grid = xs[:, None] + 1j * ys
        self.threadData = {}

        # CUBIC
        self.z0 = 1
        self.z1 = -0.5+np.sqrt(3) * 1j/2
        self.z2 = -0.5-np.sqrt(3) * 1j/2
        self.roots = [self.z0, self.z1, self.z2]

        # QUARTIC
        # self.z0 = 1
        # self.z1 = -1
        # self.z2 = -1j
        # self.z3 = 1j
        # self.roots = [self.z0, self.z1, self.z2, self.z3]

        # QUINTIC
        # self.z0 = 1
        # self.z1 = (-1 / 4) - (math.sqrt(5) / 4) - (1j * math.sqrt((5 / 8) - (math.sqrt(5) / 8)))
        # self.z2 = (-1 / 4) + (math.sqrt(5) / 4) + (1j * math.sqrt((5 / 8) + (math.sqrt(5) / 8)))
        # self.z3 = (-1 / 4) + (math.sqrt(5) / 4) - (1j * math.sqrt((5 / 8) + (math.sqrt(5) / 8)))
        # self.z4 = (-1 / 4) - (math.sqrt(5) / 4) + (1j * math.sqrt((5 / 8) - (math.sqrt(5) / 8)))
        # self.roots = [self.z0, self.z1, self.z2, self.z3, self.z4]

        # P(z) = z^8 + 15z^4 - 16
        # self.z0 = 1
        # self.z1 = -1
        # self.z2 = 1j
        # self.z3 = -1j
        # self.z4 = (-1-1j)*math.sqrt(2)
        # self.z5 = (1+1j)*math.sqrt(2)
        # self.z6 = (1-1j)*math.sqrt(2)
        # self.roots = [self.z0, self.z1, self.z2, self.z3, self.z4, self.z5, self.z6]

        # P(z) = z^(4+3i)-1
        # self.z0 = 1
        # self.z1 = (e**(-(18 * pi) / 25)) * (e**(-(24j * pi) / 25))
        # self.z2 = (e**(-(12 * pi) / 25)) * (e**(-(16j * pi) / 25))
        # self.z3 = (e**(-(6 * pi) / 25)) * (e**(-(8j * pi) / 25))
        # self.z4 = (e**((6 * pi) / 25)) * (e**((8j * pi) / 25))
        # self.z5 = (e**((12 * pi) / 25)) * (e**((16j * pi) / 25))
        # self.roots = [self.z0, self.z1, self.z2, self.z3, self.z4, self.z5]

        # P(z) = z^10 - 1
        # self.z0 = -1
        # self.z1 = 1
        # self.z2 = -(1 / 4) - (sqrt(5) / 4) - (1j * sqrt((5 / 8) - (sqrt(5) / 8)))
        # self.z3 = (1 / 4) + (sqrt(5) / 4) + (1j * sqrt((5 / 8) - (sqrt(5) / 8)))
        # self.z4 = (1 / 4) - (sqrt(5) / 4) - (1j * sqrt((5 / 8) + (sqrt(5) / 8)))
        # self.z5 = -(1 / 4) + (sqrt(5) / 4) + (1j * sqrt((5 / 8) + (sqrt(5) / 8)))
        # self.z6 = -(1 / 4) + (sqrt(5) / 4) - (1j * sqrt((5 / 8) + (sqrt(5) / 8)))
        # self.roots = [self.z0, self.z1, self.z2, self.z3, self.z4, self.z5, self.z6]

        self.res = res
        self.newton_state = False

    def __check_args_grid(self, xmin, xmax, ymin, ymax, res):
        if type(res[0]) != int or type(res[1]) != int:
            raise ValueError("Resolution must be a tuple of integers")
        if res[0] <= 0 or res[1] <= 0:
            raise ValueError("Resolution must be greater than 0")
        if type(xmin) != int or type(xmax) != int or type(ymin) != int or type(ymax) != int:
            raise ValueError("All args must be integers")

    def f_cubic(self, x):
        return power(x, (3-1))

    def fprime_cubic(self, x):
        return 3 * power(x, 2)

    def f_quartic(self, x):
        return x**4-1

    def fprime_quartic(self, x):
        return 4*x**3

    def f_quintic(self, x):
        return x**5-1

    def fprime_quintic(self, x):
        return 5*x**4

    def f_other(self, x):
        return x**10 - 1

    def fprime_other(self, x):
        return 10 * x**9

    def newton(self, x, a=1):
        return x - (a * (self.f_cubic(x) / self.fprime_cubic(x)))

    def newton_iter(self):
        unchanged = 0
        for i in range(200):
            lastGrid = self.grid.copy()
            self.grid = self.newton(self.grid)

            if np.isclose(lastGrid.all(), self.grid.all()):
                unchanged += 1
            elif unchanged != 0:
                unchanged = 0
            if unchanged == 5:
                break

        self.newton_state = True

    def darken(self, color_a, t):
        return tuple(int(a + (b - a) * t) for a, b in zip(color_a, (0, 0, 0)))

    def gen_image(self):
        im = img.new("RGB", self.res)
        px = im.load()

        if not self.newton_state:
            print("\nBeginning Newton iter")
            self.newton_iter()

        print("Beginning color classification")

        numThreads = 3
        threads = []
        splitRangesI = list(split_range(range(self.res[0]), numThreads))
        splitRangesJ = list(split_range(range(self.res[1]), numThreads))

        for i in range(numThreads):
            self.color_classify_batch(i, splitRangesI[i], splitRangesJ[i])

        for batch_ind in range(numThreads):
            toWrite = self.threadData[batch_ind]
            for i in splitRangesI[batch_ind]:
                for j in splitRangesJ[batch_ind]:
                    px[i, j] = tuple(toWrite[i, j])

        # for i in range(self.res[0]):
        #     for j in range(self.res[1]):
        #         px[i, j] = self.color_classification(i, j)

        #im.save("out.png")
        return im

    def color_classification(self, i, j):
        colors = [[252, 231, 98], [209, 73, 91], [27, 231, 255], [187, 182, 223], [83, 134, 228]]

        if np.isclose(self.grid[i, j], self.z0):
            out = colors[0]
        elif np.isclose(self.grid[i, j], self.z1):
            out = colors[1]
        elif np.isclose(self.grid[i, j], self.z2):
            out = colors[2]
        # elif np.isclose(self.grid[i, j], self.z3):
        #     out = colors[3]
        # elif np.isclose(self.grid[i, j], self.z4):
        #     out = colors[4]
        # elif np.isclose(self.grid[i, j], self.z5):
        #     out = (200, 200, 200)
        # elif np.isclose(self.grid[i, j], self.z6):
        #     out = (100, 0, 255)
        else:
            out = [0, 0, 0]
        return out

    def color_classify_batch(self, ind, rngeI, rngeJ):
        results = np.zeros(self.res, dtype=list)
        for i in rngeI:
            for j in rngeJ:
                print(i, j)
                results[i][j] = self.color_classification(i, j)
        self.threadData[ind] = results
