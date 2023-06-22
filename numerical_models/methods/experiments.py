import numpy as np
import matplotlib.pyplot as plt

from types import FunctionType as func

from collections import namedtuple

from methods.rk4 import runge_kutta_default, Params

STEP = 2**-10

def get_f(par: Params) -> func:
    
    def f(v: np.ndarray, t: float):
        x, y, z = v

        a = par.r * x * (1 - x / par.K) - par.beta*x - par.alpha * x * z
        b = par.beta * x - (par.n * y * z / (y + par.m)) - par.miu * y
        c = par.alpha_1 * x * z + (par.ro * z * z) - (par.n_1 * z * z / (y + par.m))

        return np.array([a, b, c])
    
    return f


def first_experiment(par: Params):
    r1, x1, y1, z1 = runge_kutta_default(get_f(par), np.array([3.01, 5.05, 4.28]), 1, 60,STEP) 
    r2, x2, y2, z2 = runge_kutta_default(get_f(par), np.array([4.6, 5.9, 3.1]), 1, 60,STEP)

    fig = plt.figure()

    ax = fig.add_subplot(1,3,1)
    ax.plot(r1, x1, label='$x(t)$')
    ax.plot(r1, y1, label='$y(t)$')
    ax.plot(r1, z1, label='$z(t)$')
    ax.set_title('1.1')
    ax.legend(loc='best')
    ax.grid()
    ax.set_xlim([1,4])

    ax = fig.add_subplot(1,3,2)
    ax.plot(r2, x2, label='$x(t)$')
    ax.plot(r2, y2, label='$y(t)$')
    ax.plot(r2, z2, label='$z(t)$')
    ax.set_title('1.2')
    ax.grid()
    ax.set_xlim([1,4])

    ax = fig.add_subplot(1,3,3, projection='3d')
    ax.plot(x1,y1,z1, label='1.1')
    ax.plot(x2,y2,z2, label='1.2')
    ax.set_title('1.3')
    ax.legend(loc='best')

    plt.legend(loc='best')
    plt.show()


def second_experiment(par: Params):
    r1, x1, y1, z1 = runge_kutta_default(get_f(par), np.array([0.3, 2.4, 3.9]), 1, 50,STEP) 
    r2, x2, y2, z2 = runge_kutta_default(get_f(par), np.array([0.6, 2.4, 4.1]), 1, 50,STEP)
    r3, x3, y3, z3 = runge_kutta_default(get_f(par), np.array([2.1, 1.2, 1.1]), 1, 50,STEP)

    fig = plt.figure()

    ax = fig.add_subplot(2,2,1)
    ax.plot(r1, x1, label='$x(t)$')
    ax.plot(r1, y1, label='$y(t)$')
    ax.plot(r1, z1, label='$z(t)$')
    ax.set_title('2.1')
    ax.grid()
    ax.set_xlim([1,4])

    ax = fig.add_subplot(2,2,2)
    ax.plot(r2, x2, label='$x(t)$')
    ax.plot(r2, y2, label='$y(t)$')
    ax.plot(r2, z2, label='$z(t)$')
    ax.legend(loc='best')
    ax.set_title('2.2')
    ax.grid()
    ax.set_xlim([1,4])

    ax = fig.add_subplot(2,2,3)
    ax.plot(r3, x3, label='$x(t)$')
    ax.plot(r3, y3, label='$y(t)$')
    ax.plot(r3, z3, label='$z(t)$')
    ax.set_title('2.3')
    ax.grid()
    ax.set_xlim([1,4])

    ax = fig.add_subplot(2,2,4, projection='3d')
    ax.plot(x1,y1,z1, label='2.1')
    ax.plot(x2,y2,z2, label='2.2')
    ax.plot(x3,y3,z3, label='2.3')
    ax.set_title('2.4')
    ax.legend(loc='best')

    plt.show()

def third_experiment(par: Params):
    r1, x1, y1, z1 = runge_kutta_default(get_f(par), np.array([0.3, 2.4, 3.9]), 1, 50,STEP) 

    fig = plt.figure()

    ax = fig.add_subplot(1,2,1)
    ax.plot(r1, x1, label='$x(t)$')
    ax.plot(r1, y1, label='$y(t)$')
    ax.plot(r1, z1, label='$z(t)$')
    ax.set_title('3.1')
    ax.legend(loc='best')
    ax.grid()
    ax.set_xlim([1,4])

    ax = fig.add_subplot(1,2,2, projection='3d')
    ax.plot(x1,y1,z1)
    ax.set_title('3.2')

    plt.show()

def fourth_experiment(par: Params):
    r1, x1, y1, z1 = runge_kutta_default(get_f(par), np.array([1.2, 2.1, 2.4]), 1, 50,STEP) 

    fig = plt.figure()

    ax = fig.add_subplot(1,2,1)
    ax.plot(r1, x1, label='$x(t)$')
    ax.plot(r1, y1, label='$y(t)$')
    ax.plot(r1, z1, label='$z(t)$')
    ax.set_title('4.1')
    ax.legend(loc='best')
    ax.grid()
    ax.set_xlim([1,4])

    ax = fig.add_subplot(1,2,2, projection='3d')
    ax.plot(x1,y1,z1)
    ax.set_title('4.2')

    plt.show()

def fifth_experiment(par: Params):
    r1, x1, y1, z1 = runge_kutta_default(get_f(par), np.array([1.2, 2.1, 4.28]), 1, 50,STEP) 

    fig = plt.figure()

    ax = fig.add_subplot(1,2,1)
    ax.plot(r1, x1, label='$x(t)$')
    ax.plot(r1, y1, label='$y(t)$')
    ax.plot(r1, z1, label='$z(t)$')
    ax.set_title('5.1')
    ax.legend(loc='best')
    ax.grid()
    ax.set_xlim([1,4])

    ax = fig.add_subplot(1,2,2, projection='3d')
    ax.plot(x1,y1,z1)
    ax.set_title('5.2')

    plt.show()


def sixth_experiment(par: Params):
    r1, x1, y1, z1 = runge_kutta_default(get_f(par), np.array([0.3, 2.4, 3.9]), 1, 50, STEP) 
    r2, x2, y2, z2 = runge_kutta_default(get_f(par), np.array([4.1, 2.2, 5.1]), 1, 50, STEP)
    r3, x3, y3, z3 = runge_kutta_default(get_f(par), np.array([2.1, 1.2, 1.1]), 1, 50, STEP)

    fig = plt.figure()

    ax = fig.add_subplot(2,2,1)
    ax.plot(r1, x1, label='$x(t)$')
    ax.plot(r1, y1, label='$y(t)$')
    ax.plot(r1, z1, label='$z(t)$')
    ax.set_title('6.1')
    ax.grid()
    ax.set_xlim([1,4])

    ax = fig.add_subplot(2,2,2)
    ax.plot(r2, x2, label='$x(t)$')
    ax.plot(r2, y2, label='$y(t)$')
    ax.plot(r2, z2, label='$z(t)$')
    ax.legend(loc='best')
    ax.set_title('6.2')
    ax.grid()
    ax.set_xlim([1,4])

    ax = fig.add_subplot(2,2,3)
    ax.plot(r3, x3, label='$x(t)$')
    ax.plot(r3, y3, label='$y(t)$')
    ax.plot(r3, z3, label='$z(t)$')
    ax.set_title('6.3')
    ax.grid()
    ax.set_xlim([1,4])

    ax = fig.add_subplot(2,2,4, projection='3d')
    ax.plot(x1,y1,z1,label='6.1')
    ax.plot(x2,y2,z2,label='6.2')
    ax.plot(x3,y3,z3,label='6.3')
    ax.set_title('6.4')
    ax.legend(loc='best')

    plt.show()
