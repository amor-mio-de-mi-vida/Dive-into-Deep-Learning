import math
import torch
from d2l import torch as d2l

def f(x1, x2):  # Objective function
    return x1 ** 2 + 2 * x2 ** 2

def f_grad(x1, x2):  # Gradient of the objective function
    return 2 * x1, 4 * x2

def sgd(x1, x2, s1, s2, f_grad):
    g1, g2 = f_grad(x1, x2)
    # Simulate noisy gradient
    g1 += torch.normal(0.0, 1, (1,)).item()
    g2 += torch.normal(0.0, 1, (1,)).item()
    eta_t = eta * lr()
    return (x1 - eta_t * g1, x2 - eta_t * g2, 0, 0)

def constant_lr():
    return 1

def exponential_lr():
    # Global variable that is defined outside this function and updated inside
    global t
    t += 1
    return math.exp(-0.1 * t)

def polynomial_lr():
    # Global variable that is defined outside this function and updated inside
    global t
    t += 1
    return (1 + 0.1 * t) ** (-0.5)


if __name__ == '__main__':
    eta = 0.1
    lr = constant_lr  # Constant learning rate
    d2l.show_trace_2d(f, d2l.train_2d(sgd, steps=50, f_grad=f_grad))
    d2l.plt.show()

    t = 1
    lr = exponential_lr
    d2l.show_trace_2d(f, d2l.train_2d(sgd, steps=1000, f_grad=f_grad))
    d2l.plt.show()

    t = 1
    lr = polynomial_lr
    d2l.show_trace_2d(f, d2l.train_2d(sgd, steps=50, f_grad=f_grad))
    d2l.plt.show()
