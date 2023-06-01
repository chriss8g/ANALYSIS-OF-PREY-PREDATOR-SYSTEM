import numpy as np
import scipy as sc

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from collections import namedtuple
from types import FunctionType

Params = namedtuple("Params", ['r', 'beta', 'alpha', 'alpha_1', 'n', 'n_1', 'K', 'ro', 'm', 'miu'])

def runge_kutta3( f: FunctionType, g: FunctionType, h: FunctionType, x0: float, y0:float,
                 z0: float, par: Params, a: float, b: float,
               step: float, keep_positive: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.arange(a, b, step)
    
    n = len(rng)

    X = np.zeros(n);X[0] = x0
    Y = np.zeros(n);Y[0] = y0
    Z = np.zeros(n);Z[0] = z0

    for i in range(1, n):
        xn = X[i - 1]
        yn = Y[i - 1]
        zn = Z[i - 1]

        t = rng[i - 1]

        k1 = f(t, xn, yn, zn, par)
        l1 = g(t, xn, yn, zn, par)
        m1 = h(t, xn, yn, zn, par)

        k2 = f(t + step / 2, xn + k1 * step / 2, yn + l1 * step / 2, zn + m1 * step / 2, par)
        l2 = g(t + step / 2, xn + k1 * step / 2, yn + l1 * step / 2, zn + m1 * step / 2, par)
        m2 = h(t + step / 2, xn + k1 * step / 2, yn + l1 * step / 2, zn + m1 * step / 2, par)

        k3 = f(t + step / 2, xn + k2 * step / 2, yn + l2 * step / 2, zn + m2 * step / 2, par)
        l3 = g(t + step / 2, xn + k2 * step / 2, yn + l2 * step / 2, zn + m2 * step / 2, par)
        m3 = h(t + step / 2, xn + k2 * step / 2, yn + l2 * step / 2, zn + m2 * step / 2, par)

        k4 = f(t + step, xn + k3 * step, yn + l3 * step, zn + m3 * step, par)
        l4 = g(t + step, xn + k3 * step, yn + l3 * step, zn + m3 * step, par)
        m4 = h(t + step, xn + k3 * step, yn + l3 * step, zn + m3 * step, par)

        X[i] += xn + (k1 + .5 * k2 + .5 * k3 + k4) * step / 6
        Y[i] += yn + (l1 + .5 * l2 + .5 * l3 + l4) * step / 6
        Z[i] += zn + (m1 + .5 * m2 + .5 * m3 + m4) * step / 6

        if keep_positive:
            X[i] = max(X[i], 0)
            Y[i] = max(Y[i], 0)
            Z[i] = max(Z[i], 0)

    return (rng, X, Y, Z)
