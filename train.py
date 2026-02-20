import torch

from dataset.build_dataset import build_dataset

from nets.build_model import build_model
from dataset.build_dataset import build_dataset
from pre_weights.load_preweights import load_backbone_pretrained_to_detector

from utils.optim_lr_factory import build_optimizer, build_lr_scheduler
from utils.loss2 import YOLO_Loss
from utils.loss import YoloLoss
from utils.fit_one_epoch import fit_one_epoch
from utils.logger import save_logger, save_config




import time

def base_config():
    exp_time = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    # 获取当前device0的显卡型号
    GPU_model = torch.cuda.get_device_name(0)
    config = {
        "exp_time": exp_time,
        "GPU_model": GPU_model,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "exp_name": "exp3_full",
        "exp_note": "在loss中针对cls增加了softmax，测试一下",
        "model_name": "YOLOv1",
        "save_interval": 10,
        "train_path": r"/root/autodl-tmp/dataset_full/YOLOv1_dataset/train",
        "test_path": r"/root/autodl-tmp/dataset_full/YOLOv1_dataset/test",
        # "pre_weights": r"pre_weights\best_model.pth",
        "pre_weights": r"pre_weights/best_model.pth",
        # test model 
        "debug_mode": None, # 当debug_mode为None时,表示正常模式; 否则为debug模式,使用部分数据训练
        "num_classes": 20,
        "input_size": 448,
        "batch_size": 32,
        "epochs": 135,
        "metric_interval": 5, # 每间隔几轮评估一次
        "num_workers": 8,
        "persistent_workers": True,
        "S": 7,
        "B": 2,
        "lambda_coord": 5,
        "lambda_noobj": 0.5,
        "profile_time" : False,
        "profile_cuda_sync" : False,
        "optimizer": {
            "type": "SGD",
            "lr": 1e-3,
            "momentum": 0.9,
            "weight_decay": 0.0005,
            # "lr_scheduler":{
            #     "type": "StepLR",
            #     "step_size": 30,
            #     "gamma": 0.1,
            # }
            
            "lr_scheduler": {
                "type": "YOLOv1DetLR",
                "lr_warmup_start": 0.0001,
                "lr_base": 1e-3,
                "warmup_epochs": 10,
                "phase1_epochs": 75,
                "phase2_epochs": 30,
                "phase3_epochs": 30,
            },
        },
        # "optimizer": {
        #     "type": "Adam",
        #     "lr": 0.0001,
        #     "lr_scheduler": {
        #         "type": "CosineAnnealingLR",
        #         "T_max": 100,
        #         "eta_min": 1e-6,
        #     },
        #     "weight_decay": 1e-4,
        # }

    }
    config["exp_name"] += str("_" + exp_time)
    return config

def train():
    state = None
    cfg = base_config()
    save_config(cfg)

    model = build_model(cfg).to(cfg["device"])
    # 加载预训练权重
    if cfg["pre_weights"] is not None:
        load_backbone_pretrained_to_detector(
        detector=model,
        ckpt_path=cfg["pre_weights"],
        classifier_key_prefix="conv_backbone",  # 对齐你的 YOLOv1_Classifier
        detector_key_prefix="backbone",         # 对齐你的 YOLOv1
        map_location=cfg["device"],
        strict=False,
        verbose=True,
    )
        
    train_loader, test_loader = build_dataset(cfg)
    optimizer = build_optimizer(model, cfg=cfg)
    lr_scheduler = build_lr_scheduler(optimizer, cfg=cfg)
    loss_fn = YoloLoss(S=cfg["S"], 
                       B=cfg["B"], 
                       C=cfg["num_classes"], 
                       lambda_coord=cfg["lambda_coord"], 
                       lambda_noobj=cfg["lambda_noobj"], 
                       ic_debug=False)
    # loss_fn = YOLO_Loss(S=cfg["S"], 
    #                    B=cfg["B"], 
    #                    C=cfg["num_classes"], 
    #                    ic_debug=False)
    for epoch in range(cfg["epochs"]):
        metrics, state= fit_one_epoch(
            epoch, cfg, model, train_loader, test_loader, loss_fn, optimizer, lr_scheduler, state
        )
        # save logs and model
        save_logger(model, metrics, cfg, state)


if __name__ == '__main__':
    train()