import torch
from torch import nn

# We define a helper function to calculate convolutions. It initializes the
# convolutional layer weights and performs corresponding dimensionality
# elevations and reductions on the input and output
def comp_conv2d(conv2d, X):
    # (1, 1) indicates that batch size and the number of channels are both 1
    X = X.reshape((1, 1) + X.shape)
    Y = conv2d(X)
    # Strip the first two dimensions: examples and channels
    return Y.reshape(Y.shape[2:])


if __name__ == '__main__':
    # 1 row and column is padded on either side, so a total of 2 rows or columns
    # are added
    conv2d = nn.LazyConv2d(1, kernel_size=3, padding=1)
    X = torch.rand(size=(8, 8))
    print(comp_conv2d(conv2d, X).shape)

    # We use a convolution kernel with height 5 and width 3. The padding on either
    # side of the height and width are 2 and 1, respectively
    conv2d = nn.LazyConv2d(1, kernel_size=(5, 3), padding=(2, 1))
    print(comp_conv2d(conv2d, X).shape)

    conv2d = nn.LazyConv2d(1, kernel_size=3, padding=1, stride=2)
    print(comp_conv2d(conv2d, X).shape)

    conv2d = nn.LazyConv2d(1, kernel_size=(3, 5), padding=(0, 1), stride=(3, 4))
    print(comp_conv2d(conv2d, X).shape)