import torch
import torchvision


if __name__ == "__main__":
    X = torch.arange(16.).reshape(1, 1, 4, 4)
    print(X)

    rois = torch.Tensor([[0, 0, 0, 20, 20], [0, 0, 10, 30, 30]])

    print(torchvision.ops.roi_pool(X, rois, output_size=(2, 2), spatial_scale=0.1))

    

