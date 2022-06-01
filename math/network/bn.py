import random

import torch.nn as nn
import torch
from torchvision import models


def BN(feature, mean, var):
    shape = feature.shape
    for i in range(shape[1]):  # along channel axis
        f_t = feature[:, i, ...]
        mean_t = f_t.mean()
        std_t1 = f_t.std()  # 整个channel的std
        std_t2 = f_t.std(ddof=1)  # 样本标准差

        # - mean / std
        feature[:, i, ...] = (feature[:, i, ...] - mean_t) / std_t1

        # update running mean and var for inference
        mean[i] = mean[i] * (1 - 0.1) + mean_t * 0.1  # 0.1 momentum
        var[i] = var[i] * (1 - .01) + (std_t2 ** 2) * 0.1  # momentum
    return feature, mean, var


if __name__ == '__main__':
    random.seed(1)
    feature = torch.randn(2, 2, 2, 2)  # random NCHW
    print(feature)

    # initialize mean and var
    mean = [0., 0.]
    var = [1., 1.0]

    feature_bn, mean_bn, variance_bn = BN(feature.numpy().copy(), mean, var)
    print("BN by custom function:")
    print(feature_bn)
    print(mean_bn)
    print(variance_bn)

    bn = nn.BatchNorm2d(2)
    output = bn(feature)
    print("BN by torch")
    print(output)
