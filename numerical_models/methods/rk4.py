import numpy as np
import scipy.integrate as it

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from collections import namedtuple
from types import FunctionType as func

Params = namedtuple("Params", ['r', 'beta', 'alpha', 'alpha_1', 'n', 'n_1', 'K', 'ro', 'm', 'miu'])

def runge_kutta3( f: func, g: func, h: func, x0: float, y0:float,
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

        X[i] += xn + (k1 + 2 * k2 + 2 * k3 + k4) * step / 6
        Y[i] += yn + (l1 + 2 * l2 + 2 * l3 + l4) * step / 6
        Z[i] += zn + (m1 + 2 * m2 + 2 * m3 + m4) * step / 6

        if keep_positive:
            X[i] = max(X[i], 0)
            Y[i] = max(Y[i], 0)
            Z[i] = max(Z[i], 0)

    return (rng, X, Y, Z)

def runge_kutta_default(f: func, x0: np.ndarray, a: float, b: float, h: float):
    t = np.arange(a, b, h)
    y = it.odeint(f, x0, t)

    return (t, y[:,0], y[:,1], y[:,2])
