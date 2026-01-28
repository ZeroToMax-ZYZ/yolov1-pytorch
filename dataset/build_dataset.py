import torch
from torch.utils.data import DataLoader, Subset

from dataset.augment import build_yolov1_transforms
from dataset.VOC_dataset import VOCDataset


def build_dataset(cfg):
    train_transform, val_transform = build_yolov1_transforms(img_size=cfg["input_size"])
    train_dataset = VOCDataset(base_path=cfg["train_path"], transform=train_transform, img_size=cfg["input_size"], S=cfg["S"])
    test_dataset = VOCDataset(base_path=cfg["test_path"], transform=val_transform, img_size=cfg["input_size"], S=cfg["S"])
    if cfg["debug_mode"] is not None:
        # fast debug mode, use a smaller subset
        test_size = int(len(train_dataset) * cfg["debug_mode"])
        indices = torch.randperm(len(train_dataset))[:test_size]
        train_dataset = Subset(train_dataset, indices)
        print("⚠️ debug mode : training dataset len: ", len(train_dataset))
        print("⚠️ debug mode : validation dataset len: ", len(test_dataset))
    else:
        print("training dataset len: ", len(train_dataset))
        print("validation dataset len: ", len(test_dataset))

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        pin_memory=True,
        persistent_workers=cfg["persistent_workers"], # 优化win, 复用进程
        prefetch_factor=2,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=True,
        persistent_workers=cfg["persistent_workers"], # 优化win, 复用进程
        prefetch_factor=2,
    )

    return train_loader, test_loader