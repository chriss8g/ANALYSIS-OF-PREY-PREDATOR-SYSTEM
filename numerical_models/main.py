import numpy as np
import matplotlib.pyplot as plt
from methods.rk4 import runge_kutta3, Params

# Definir los parametros de las funciones(beta, alfa, etc...)
#            r    beta  alfa  alfa1   n      n1    k    rho   m     miu
p1 = Params(0.82, 0.87, 1.56, 1.12, 2.41,   1.83, 12.0, 1.38, 0.13, 0.11)  # Primer set de experimentos
p2 = Params(1.32, 0.87, 1.56, 0.72, 2.41,   0.41, 2.8, 1.38, 0.23, 0.11)  # Segundo set de experimentos
p3 = Params(1.32, 0.87, 0.76, 0.72, 0.6,    0.41, 2.8, 0.78, 0.23, 0.11)  # Tercer set de experimentos
p4 = Params(0.82, 0.87, 0.76, 0.72, 1.2,    0.41, 2.8, 1.38, 0.23, 0.11)  # Cuarto set de experimentos
p5 = Params(1.32, 0.87, 1.16, 0.72, 0.3095, 0.41, 2.8, 0.78, 0.23, 0.11)  # Quinto set de experimentos
p6 = Params(1.32, 0.87, 1.16, 1.16, 0.3,    0.41, 2.8, 0.78, 0.23, 0.11)  # Quinto set de experimentos

def f(t, x, y, z, par: Params): 
    return par.r * x * (1 - x / par.K) - par.beta - par.alpha * x * z

def g(t, x, y, z, par: Params): 
    return par.beta * x - (par.n * y * z / (y + par.m)) - par.miu * y

def h(t, x, y, z, par: Params): 
    return par.alpha_1 * x * z + (par.ro * z * z) - (par.n_1 * z * z / (y + par.m))


def first_experiment():
    r1, x1, y1, z1 = runge_kutta3(f, g, h, 3.01, 5.05, 4.28,p1, 1, 3,2**-10) 
    r2, x2, y2, z2 = runge_kutta3(f, g, h, 4.6, 5.9, 3.1, p1, 1, 3,2**-10)

    fig = plt.figure()

    ax = fig.add_subplot(1,3,1)
    ax.plot(r1, x1, label='$x(t)$')
    ax.plot(r1, y1, label='$y(t)$')
    ax.plot(r1, z1, label='$z(t)$')
    ax.set_title('1.1')
    ax.legend(loc='best')
    ax.grid()

    ax = fig.add_subplot(1,3,2)
    ax.plot(r2, x2, label='$x(t)$')
    ax.plot(r2, y2, label='$y(t)$')
    ax.plot(r2, z2, label='$z(t)$')
    ax.set_title('1.2')
    ax.grid()

    ax = fig.add_subplot(1,3,3, projection='3d')
    ax.plot(x1,y1,z1, label='1.1')
    ax.plot(x2,y2,z2, label='1.2')
    ax.set_title('1.3')
    ax.legend(loc='best')

    plt.legend(loc='best')
    plt.show()


def second_experiment():
    r1, x1, y1, z1 = runge_kutta3(f, g, h, 0.3, 2.4, 3.9, p2, 1, 3,2**-10) 
    r2, x2, y2, z2 = runge_kutta3(f, g, h, 0.6, 2.4, 4.1, p2, 1, 3,2**-10)
    r3, x3, y3, z3 = runge_kutta3(f, g, h, 2.1, 1.2, 1.1, p2, 1, 3,2**-10)

    fig = plt.figure()

    ax = fig.add_subplot(2,2,1)
    ax.plot(r1, x1, label='$x(t)$')
    ax.plot(r1, y1, label='$y(t)$')
    ax.plot(r1, z1, label='$z(t)$')
    ax.set_title('2.1')
    ax.grid()

    ax = fig.add_subplot(2,2,2)
    ax.plot(r2, x2, label='$x(t)$')
    ax.plot(r2, y2, label='$y(t)$')
    ax.plot(r2, z2, label='$z(t)$')
    ax.legend(loc='best')
    ax.set_title('2.2')
    ax.grid()

    ax = fig.add_subplot(2,2,3)
    ax.plot(r2, x2, label='$x(t)$')
    ax.plot(r2, y2, label='$y(t)$')
    ax.plot(r2, z2, label='$z(t)$')
    ax.set_title('2.3')
    ax.grid()

    ax = fig.add_subplot(2,2,4, projection='3d')
    ax.plot(x1,y1,z1, label='2.1')
    ax.plot(x2,y2,z2, label='2.2')
    ax.plot(x3,y3,z3, label='2.3')
    ax.set_title('2.4')
    ax.legend(loc='best')

    plt.show()

def third_experiment():
    r1, x1, y1, z1 = runge_kutta3(f, g, h, 0.3, 2.4, 3.9, p3, 1, 3,2**-10) 

    fig = plt.figure()

    ax = fig.add_subplot(1,2,1)
    ax.plot(r1, x1, label='$x(t)$')
    ax.plot(r1, y1, label='$y(t)$')
    ax.plot(r1, z1, label='$z(t)$')
    ax.set_title('3.1')
    ax.legend(loc='best')
    ax.grid()

    ax = fig.add_subplot(1,2,2, projection='3d')
    ax.plot(x1,y1,z1)
    ax.set_title('3.2')

    plt.show()

def fourth_experiment():
    r1, x1, y1, z1 = runge_kutta3(f, g, h, 1.2, 2.1, 2.4, p4, 1, 3,2**-10) 

    fig = plt.figure()

    ax = fig.add_subplot(1,2,1)
    ax.plot(r1, x1, label='$x(t)$')
    ax.plot(r1, y1, label='$y(t)$')
    ax.plot(r1, z1, label='$z(t)$')
    ax.set_title('4.1')
    ax.legend(loc='best')
    ax.grid()

    ax = fig.add_subplot(1,2,2, projection='3d')
    ax.plot(x1,y1,z1)
    ax.set_title('4.2')

    plt.show()

def fifth_experiment():
    r1, x1, y1, z1 = runge_kutta3(f, g, h, 1.2, 2.1, 4.28, p5, 1, 3,2**-10) 

    fig = plt.figure()

    ax = fig.add_subplot(1,2,1)
    ax.plot(r1, x1, label='$x(t)$')
    ax.plot(r1, y1, label='$y(t)$')
    ax.plot(r1, z1, label='$z(t)$')
    ax.set_title('5.1')
    ax.legend(loc='best')
    ax.grid()

    ax = fig.add_subplot(1,2,2, projection='3d')
    ax.plot(x1,y1,z1)
    ax.set_title('5.2')

    plt.show()


def sixth_experiment():
    r1, x1, y1, z1 = runge_kutta3(f, g, h, 0.3, 2.4, 3.9, p6, 1, 4,2**-10) 
    r2, x2, y2, z2 = runge_kutta3(f, g, h, 4.1, 2.2, 5.1, p6, 1, 4,2**-10)
    r3, x3, y3, z3 = runge_kutta3(f, g, h, 2.1, 1.2, 1.1, p6, 1, 4,2**-10)

    fig = plt.figure()

    ax = fig.add_subplot(2,2,1)
    ax.plot(r1, x1, label='$x(t)$')
    ax.plot(r1, y1, label='$y(t)$')
    ax.plot(r1, z1, label='$z(t)$')
    ax.set_title('6.1')
    ax.grid()

    ax = fig.add_subplot(2,2,2)
    ax.plot(r2, x2, label='$x(t)$')
    ax.plot(r2, y2, label='$y(t)$')
    ax.plot(r2, z2, label='$z(t)$')
    ax.legend(loc='best')
    ax.set_title('6.2')
    ax.grid()

    ax = fig.add_subplot(2,2,3)
    ax.plot(r2, x2, label='$x(t)$')
    ax.plot(r2, y2, label='$y(t)$')
    ax.plot(r2, z2, label='$z(t)$')
    ax.set_title('6.3')
    ax.grid()

    ax = fig.add_subplot(2,2,4, projection='3d')
    ax.plot(x1,y1,z1,label='6.1')
    ax.plot(x2,y2,z2,label='6.2')
    ax.plot(x3,y3,z3,label='6.3')
    ax.set_title('6.4')
    ax.legend(loc='best')

    plt.show()


# first_experiment()
# second_experiment()
# third_experiment()
# fourth_experiment()
# fifth_experiment()
# sixth_experiment()