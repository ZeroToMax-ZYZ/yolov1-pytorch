import torch
import torch.optim as optim

import torch.nn as nn

'''
优化器和学习率的工厂函数
目前支持的优化器
SGD
Adam

目前支持的学习率调度器
StepLR
CosineAnnealingLR
'''
def build_optimizer(model, cfg):
    optimizer = cfg["optimizer"]["type"]

    if optimizer == "SGD":
        optimizer = optim.SGD(
            model.parameters(),
            lr=cfg["optimizer"]["lr"],
            momentum=cfg["optimizer"]["momentum"],
            weight_decay=cfg["optimizer"]["weight_decay"],
        )

    elif optimizer == "Adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=cfg["optimizer"]["lr"],
            weight_decay=cfg["optimizer"]["weight_decay"],
        )

    else:
        raise ValueError(f"❗ Unsupported optimizer type: {optimizer}")

    return optimizer

def build_lr_scheduler(optimizer, cfg):
    lr_scheduler = cfg["optimizer"]["lr_scheduler"]["type"]

    if lr_scheduler == "StepLR":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=cfg["optimizer"]["lr_scheduler"]["step_size"],
            gamma=cfg["optimizer"]["lr_scheduler"]["gamma"],
        )

    elif lr_scheduler == "CosineAnnealingLR":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg["optimizer"]["lr_scheduler"]["T_max"],
            eta_min=cfg["optimizer"]["lr_scheduler"]["eta_min"],
        )
    else:
        raise ValueError(f"❗ Unsupported lr_scheduler type: {lr_scheduler}")
    
    return scheduler

def build_loss_fn(cfg):
    loss_fu = cfg["loss_fn"]

    if loss_fu == "CrossEntropyLoss":
        loss_function = nn.CrossEntropyLoss()

    else:
        raise ValueError(f"❗ Unsupported loss function type: {loss_fu}")
    
    return loss_function

