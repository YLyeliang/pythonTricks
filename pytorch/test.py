# -*- coding: utf-8 -*-
# @Time : 2021/9/18 上午10:50
# @Author: yl
# @File: test.py
import torch
from torch import nn
from torchvision.models import ResNet


def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Linear):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 0)


def small_test():
    """
    测试一下参数全是0时能否训练模型
    """
    model = nn.Sequential(nn.Linear(3, 1),)
    fill_fc_weights(model)
    loss = torch.nn.BCEWithLogitsLoss()
    sgd = torch.optim.SGD(model.parameters(), lr=0.1)
    params = list(model.named_parameters())
    for i in range(1000):
        x = torch.rand(3, 3)
        gt = torch.randint(0, 2, (3, 1), dtype=torch.float)

        p = model(x)

        l = loss(p, gt)
        sgd.zero_grad()
        l.backward()
        sgd.step()
        print(params[0][0])
        print(params[0][1].data)
        print(params[0][1].grad)

        # print(model.get_submodule("1").bias)
        # print(model.get_submodule("1").weight)
        debug = 1


if __name__ == '__main__':
    small_test()
