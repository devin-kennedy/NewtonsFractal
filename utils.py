from functools import cache

import numpy as np


def power(x, y):
    if y == 0:
        return 1

    thisPow = power(x, y//2)

    if y & 1:
        return x * thisPow * thisPow
    return thisPow * thisPow


def split_range(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))
