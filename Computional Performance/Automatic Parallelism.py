import torch
from d2l import torch as d2l

def copy_to_cpu(x, non_blocking=False):
    return [y.to('cpu', non_blocking=non_blocking) for y in x]


if __name__ == "__main__":
    devices = d2l.try_all_gpus()


    def run(x):
        return [x.mm(x) for _ in range(50)]


    x_gpu1 = torch.rand(size=(4000, 4000), device=devices[0])
    x_gpu2 = torch.rand(size=(4000, 4000), device=devices[1])

    run(x_gpu1)
    run(x_gpu2)  # Warm-up all devices
    torch.cuda.synchronize(devices[0])
    torch.cuda.synchronize(devices[1])

    with d2l.Benchmark('GPU1 time'):
        run(x_gpu1)
        torch.cuda.synchronize(devices[0])

    with d2l.Benchmark('GPU2 time'):
        run(x_gpu2)
        torch.cuda.synchronize(devices[1])

    with d2l.Benchmark('GPU1 & GPU2'):
        run(x_gpu1)
        run(x_gpu2)
        torch.cuda.synchronize()

    with d2l.Benchmark('Run on GPU1'):
        y = run(x_gpu1)
        torch.cuda.synchronize()

    with d2l.Benchmark('Copy to CPU'):
        y_cpu = copy_to_cpu(y)
        torch.cuda.synchronize()

    with d2l.Benchmark('Run on GPU1 and copy to CPU'):
        y = run(x_gpu1)
        y_cpu = copy_to_cpu(y, True)
        torch.cuda.synchronize()