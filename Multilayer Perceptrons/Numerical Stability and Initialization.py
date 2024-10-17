import torch
from d2l import torch as d2l
from matplotlib import pyplot as plt

if __name__ == '__main__':
    x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
    y = torch.sigmoid(x)
    y.backward(torch.ones_like(x))

    d2l.plot(x.detach().numpy(), [y.detach().numpy(), x.grad.numpy()],
             legend=['sigmoid', 'gradient'], figsize=(4.5, 2.5))
    plt.show()

    M = torch.normal(0, 1, size=(4, 4))
    print('a single matrix \n', M)
    for i in range(100):
        M = M @ torch.normal(0, 1, size=(4, 4))
    print('after multiplying 100 matrices\n', M)

    