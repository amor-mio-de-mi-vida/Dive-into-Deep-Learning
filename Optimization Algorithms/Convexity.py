import numpy as np
import torch
from mpl_toolkits import mplot3d
from d2l import torch as d2l

if __name__ == '__main__':
    f = lambda x: 0.5 * x ** 2  # Convex
    g = lambda x: torch.cos(np.pi * x)  # Nonconvex
    h = lambda x: torch.exp(0.5 * x)  # Convex

    x, segment = torch.arange(-2, 2, 0.01), torch.tensor([-1.5, 1])
    d2l.use_svg_display()
    _, axes = d2l.plt.subplots(1, 3, figsize=(9, 3))
    for ax, func in zip(axes, [f, g, h]):
        d2l.plot([x, segment], [func(x), func(segment)], axes=ax)

    f = lambda x: (x - 1) ** 2
    d2l.set_figsize()
    d2l.plot([x, segment], [f(x), f(segment)], 'x', 'f(x)')
    d2l.plt.show()

