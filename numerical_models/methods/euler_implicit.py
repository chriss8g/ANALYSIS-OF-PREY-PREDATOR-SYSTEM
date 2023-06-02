import numpy as np
from types import FunctionType as func
from collections import namedtuple

from .system_solvers import newton_solve3

Params = namedtuple("Params", ['r', 'beta', 'alpha', 'alpha_1', 'n', 'n_1', 'K', 'ro', 'm', 'miu'])

def euler_3i(f: func, g: func, h: func,
            x0: float, y0: float, z0: float, a: float, b: float,
            step: float, par: Params, EvalJacobian: func) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    
    r = np.arange(a, b, step)
    n = len(r)

    X = np.zeros(n);X[0] = x0
    Y = np.zeros(n);Y[0] = y0
    Z = np.zeros(n);Z[0] = z0

    for i in range(1, n):
        xi = float(X[i - 1])
        yi = float(Y[i - 1])
        zi = float(Z[i - 1])

        fn = lambda xn, yn, zn: xn - X[i - 1] - step * f(r[i], xn, yn, zn, par)
        gn = lambda xn, yn, zn: yn - Y[i - 1] - step * g(r[i], xn, yn, zn, par)
        hn = lambda xn, yn, zn: zn - Z[i - 1] - step * h(r[i], xn, yn, zn, par)

        sol = newton_solve3(fn, gn, hn, np.array([xi, yi, zi]), EvalJacobian)

        X[i], Y[i], Z[i] = sol
    
    return (r, X, Y, Z)