from typing import Callable, Tuple

import numpy as np
import torch
import torch.optim
from torch import nn
from torchpack.utils.config import configs
from torchpack.utils.typing import Dataset, Optimizer, Scheduler
from core.dataset.sensatUrban import sensatUrban
import torchpack.distributed as dist

def cosine_schedule_with_warmup(k, num_epochs, batch_size, dataset_size):
    batch_size *= dist.size()

    if dist.size() == 1:
        warmup_iters = 0
    else:
        warmup_iters = 1000 // dist.size()

    if k < warmup_iters:
        return (k + 1) / warmup_iters
    else:
        iter_per_epoch = (dataset_size + batch_size - 1) // batch_size
        ratio = (k - warmup_iters) / (num_epochs * iter_per_epoch)
        return 0.5 * (1 + np.cos(np.pi * ratio))

def make_dataset():
    return {
        'train': sensatUrban(root=configs.dataset.root, num_points=configs.dataset.num_points,
                             voxel_size=configs.dataset.voxel_size, split="train"),
        'test': sensatUrban(root=configs.dataset.root, num_points=configs.dataset.num_points,
                            voxel_size=configs.dataset.voxel_size, split="val"),
    }

def make_model():
    if configs.model.name == 'spvcnn':
        from core.models.SPVCNN import SPVCNN
        if 'cr' in configs.model:
            cr = configs.model.cr
        else:
            cr = 1.0
        model = SPVCNN(num_classes=configs.data.num_classes,
                       cr=cr,
                       pres=configs.dataset.voxel_size,
                       vres=configs.dataset.voxel_size)
    else:
        raise NotImplementedError(configs.model.name)
    return model


def make_criterion() -> Callable:
    if configs.criterion.name == 'cross_entropy':
        criterion = nn.CrossEntropyLoss()
    else:
        raise NotImplementedError(configs.criterion.name)
    return criterion


def make_optimizer(model: nn.Module) -> Optimizer:
    if configs.optimizer.name == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=configs.optimizer.lr,
                                    momentum=configs.optimizer.momentum,
                                    weight_decay=configs.optimizer.weight_decay,
                                    nesterov=configs.optimizer.nesterov)
    elif configs.optimizer.name == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=configs.optimizer.lr,
            weight_decay=configs.optimizer.weight_decay)
    elif configs.optimizer.name == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=configs.optimizer.lr,
            weight_decay=configs.optimizer.weight_decay)
    else:
        raise NotImplementedError(configs.optimizer.name)
    return optimizer


def make_scheduler(optimizer: Optimizer) -> Scheduler:
    if configs.scheduler.name == 'none':
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                      lr_lambda=lambda epoch: 1)
    elif configs.scheduler.name == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=configs.train.num_epochs)
    elif configs.scheduler.name == 'cosine_warmup':
        from functools import partial

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=partial(cosine_schedule_with_warmup,
                              num_epochs=configs.train.num_epochs,
                              batch_size=configs.train.batch_size,
                              dataset_size=configs.data.training_size))
    else:
        raise NotImplementedError(configs.scheduler.name)
    return scheduler
