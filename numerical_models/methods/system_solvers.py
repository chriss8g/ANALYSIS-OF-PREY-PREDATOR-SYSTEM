import numpy as np
from types import FunctionType as func

def newton_solve3(f: func, g: func, h: func, x0: np.ndarray, EvalJacobian: func) -> np.ndarray:
    X = np.array(x0, copy=True)

    for _ in range(8): #10 steps of newton's method is good enought
        J = EvalJacobian(X)
        x_n = np.array([
            f(X[0], X[1], X[2]),
            g(X[0], X[1], X[2]),
            h(X[0], X[1], X[2]),
        ]) * -1
        try:
            sol = np.linalg.solve(J, x_n)
        except:
            print(x0)
            raise Exception('papdkapdkpakd')
        X += sol
    
    return sol