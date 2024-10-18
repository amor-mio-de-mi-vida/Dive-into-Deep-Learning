import torch
from torch import nn
from d2l import torch as d2l

def cpu():  #@save
    """Get the CPU device."""
    return torch.device('cpu')

def gpu(i=0):  #@save
    """Get a GPU device."""
    return torch.device(f'cuda:{i}')

def num_gpus():  #@save
    """Get the number of available GPUs."""
    return torch.cuda.device_count()

def try_gpu(i=0):  #@save
    """Return gpu(i) if exists, otherwise return cpu()."""
    if num_gpus() >= i + 1:
        return gpu(i)
    return cpu()

def try_all_gpus():  #@save
    """Return all available GPUs, or [cpu(),] if no GPU exists."""
    return [gpu(i) for i in range(num_gpus())]

@d2l.add_to_class(d2l.Trainer)  #@save
def __init__(self, max_epochs, num_gpus=0, gradient_clip_val=0):
    self.save_hyperparameters()
    self.gpus = [d2l.gpu(i) for i in range(min(num_gpus, d2l.num_gpus()))]

@d2l.add_to_class(d2l.Trainer)  #@save
def prepare_batch(self, batch):
    if self.gpus:
        batch = [a.to(self.gpus[0]) for a in batch]
    return batch

@d2l.add_to_class(d2l.Trainer)  #@save
def prepare_model(self, model):
    model.trainer = self
    model.board.xlim = [0, self.max_epochs]
    if self.gpus:
        model.to(self.gpus[0])
    self.model = model

if __name__ == '__main__':
    print(cpu(), gpu(), gpu(1))

    print(num_gpus())

    print(try_gpu(), try_gpu(10), try_all_gpus())

    x = torch.tensor([1, 2, 3])
    print(x.device)

    X = torch.ones(2, 3, device=try_gpu())
    print(X)

    Y = torch.rand(2, 3, device=try_gpu(1))
    print(Y)

    Z = X.cuda(1)
    print(X)
    print(Z)

    print(Y + Z)

    print(Z.cuda(1) is Z)

    net = nn.Sequential(nn.LazyLinear(1))
    net = net.to(device=try_gpu())

    print(net(X))

    print(net[0].weight.data.device)

