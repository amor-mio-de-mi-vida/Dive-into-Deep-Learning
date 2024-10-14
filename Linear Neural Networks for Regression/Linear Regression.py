import math
import time
import numpy as np
import torch
from d2l import torch as d2l
from matplotlib import pyplot as plt


def normal(x, mu, sigma):
    p = 1 / math.sqrt(2 * math.pi * sigma**2)
    return p * np.exp(-0.5 * (x - mu)**2 / sigma**2)


if __name__ == '__main__':
    # Use NumPy again for visualization
    x = np.arange(-7, 7, 0.01)

    # Mean and standard deviation pairs
    params = [(0, 1), (0, 2), (3, 1)]
    d2l.plot(x, [normal(x, mu, sigma) for mu, sigma in params], xlabel='x',
             ylabel='p(x)', figsize=(4.5, 2.5),
             legend=[f'mean {mu}, std {sigma}' for mu, sigma in params])
    plt.show()