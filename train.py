import torch

from dataset.build_dataset import build_dataset

from nets.build_model import build_model
from dataset.build_dataset import build_dataset
from utils.optim_lr_factory import build_optimizer, build_lr_scheduler
from utils.loss import YoloLoss
from utils.fit_one_epoch import fit_one_epoch
from utils.logger import save_logger, save_config




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
        "num_classes": 20,
        "input_size": 448,
        "batch_size": 64,
        "num_workers": 8,
        "persistent_workers": True,
        "S": 7,
        "B": 2,
        "lambda_coord": 5,
        "lambda_noobj": 0.5,
        
    }
    return config

def train():
    cfg = base_config()
    save_config(cfg)
    model = build_model(cfg).to(cfg["device"])
    train_loader, test_loader = build_dataset(cfg)
    optimizer = build_optimizer(model, cfg=cfg)
    lr_scheduler = build_lr_scheduler(optimizer, cfg=cfg)
    loss_fn = YoloLoss(bs=cfg["batch_size"], 
                       S=cfg["S"], 
                       B=cfg["B"], 
                       C=cfg["num_classes"], 
                       lambda_coord=cfg["lambda_coord"], 
                       lambda_noobj=cfg["lambda_noobj"], 
                       ic_debug=False)
    
    for epoch in range(cfg["epochs"]):
        metrics = fit_one_epoch(
            epoch, cfg, model, train_loader, val_loader, loss_fn, optimizer, lr_scheduler
        )
        # save logs and model
        save_logger(model, metrics, cfg)


if __name__ == '__main__':
    train()