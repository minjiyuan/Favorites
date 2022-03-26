# -*- coding: utf-8 -*-
import random
import torch
import numpy as np
np.random.seed(999)


def set_seed(seed = 1):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def mixup_data(x, y, alpha=1.0, use_cuda=False):
    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]  # 自己和打乱的自己进行叠加
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam
