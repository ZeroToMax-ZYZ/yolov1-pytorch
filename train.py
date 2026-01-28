import torch

from dataset.build_dataset import build_dataset

import time

def base_config():
    exp_time = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    config = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "exp_name": "02_YOLOv1_backbone_SGD",
        "model_name": "YOLOv1_backbone",
        "save_interval": 10,
        # "train_path": r'D:\1AAAAAstudy\python_base\pytorch\all_dataset\image_classification\ImageNet\ImageNet100\train',
        # "val_path": r"D:\1AAAAAstudy\python_base\pytorch\all_dataset\image_classification\ImageNet\ImageNet100\val",
        "train_path": r"D:\1AAAAAstudy\python_base\pytorch\all_dataset\YOLOv1_dataset\train",
        "test_path": r"D:\1AAAAAstudy\python_base\pytorch\all_dataset\YOLOv1_dataset\test",
        # test model 
        "debug_mode": 0.1, # 当debug_mode为None时,表示正常模式; 否则为debug模式,使用部分数据训练
        "input_size": 448,
        "batch_size": 64,
        "num_workers": 8,
        "persistent_workers": True,
        "S": 7,
    }
    return config

def train():
    cfg = base_config()
    train_loader, test_loader = build_dataset(cfg)


if __name__ == '__main__':
    train()