import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

# Define some kernels
def gaussian(x):
    return torch.exp(-x**2 / 2)

def boxcar(x):
    return torch.abs(x) < 1.0

def constant(x):
    return 1.0 + 0 * x

def epanechikov(x):
    return torch.max(1 - torch.abs(x), torch.zeros_like(x))

def f(x):
    return 2 * torch.sin(x) + x

def nadaraya_watson(x_train, y_train, x_val, kernel):
    dists = x_train.reshape((-1, 1)) - x_val.reshape((1, -1))
    # Each column/row corresponds to each query/key
    k = kernel(dists).type(torch.float32)
    # Normalization over keys for each query
    attention_w = k / k.sum(0)
    y_hat = y_train@attention_w
    return y_hat, attention_w

def plot(x_train, y_train, x_val, y_val, kernels, names, attention=False):
    fig, axes = d2l.plt.subplots(1, 4, sharey=True, figsize=(12, 3))
    for kernel, name, ax in zip(kernels, names, axes):
        y_hat, attention_w = nadaraya_watson(x_train, y_train, x_val, kernel)
        if attention:
            pcm = ax.imshow(attention_w.detach().numpy(), cmap='Reds')
        else:
            ax.plot(x_val, y_hat)
            ax.plot(x_val, y_val, 'm--')
            ax.plot(x_train, y_train, 'o', alpha=0.5);
        ax.set_xlabel(name)
        if not attention:
            ax.legend(['y_hat', 'y'])
    if attention:
        fig.colorbar(pcm, ax=axes, shrink=0.7)

def gaussian_with_width(sigma):
    return (lambda x: torch.exp(-x**2 / (2*sigma**2)))

if __name__ == '__main__':
    d2l.use_svg_display()
    fig, axes = d2l.plt.subplots(1, 4, sharey=True, figsize=(12, 3))

    kernels = (gaussian, boxcar, constant, epanechikov)
    names = ('Gaussian', 'Boxcar', 'Constant', 'Epanechikov')
    x = torch.arange(-2.5, 2.5, 0.1)
    for kernel, name, ax in zip(kernels, names, axes):
        ax.plot(x.detach().numpy(), kernel(x).detach().numpy())
        ax.set_xlabel(name)

    d2l.plt.show()

    n = 40
    x_train, _ = torch.sort(torch.rand(n) * 5)
    y_train = f(x_train) + torch.randn(n)
    x_val = torch.arange(0, 5, 0.1)
    y_val = f(x_val)

    plot(x_train, y_train, x_val, y_val, kernels, names)
    d2l.plt.show()

    plot(x_train, y_train, x_val, y_val, kernels, names, attention=True)
    d2l.plt.show()

    sigmas = (0.1, 0.2, 0.5, 1)
    names = ['Sigma ' + str(sigma) for sigma in sigmas]

    kernels = [gaussian_with_width(sigma) for sigma in sigmas]
    plot(x_train, y_train, x_val, y_val, kernels, names)

    plot(x_train, y_train, x_val, y_val, kernels, names, attention=True)
    d2l.plt.show()
