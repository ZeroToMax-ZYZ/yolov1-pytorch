import torch

from nets.build_model import build_model
from utils.nms import nms
from utils.decode import decode_preds, decode_labels_list
from utils.logger import save_config
from utils.metrics import compute_map

from dataset.build_dataset import build_dataset

from tqdm import tqdm
import time

def base_config():
    exp_time = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    # 获取当前device0的显卡型号
    GPU_model = torch.cuda.get_device_name(0)
    config = {
        "exp_time": exp_time,
        "GPU_model": GPU_model,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "exp_name": "val_test",
        "model_name": "YOLOv1",
        "model_path": r'logs\logs_weights\last_model.pth',
        "train_path": r"D:\1AAAAAstudy\python_base\pytorch\all_dataset/YOLOv1_dataset/train",
        "test_path": r"D:\1AAAAAstudy\python_base\pytorch\all_dataset/YOLOv1_dataset/test",
        "save_path" : r'',
        "input_size": 448,
        "batch_size": 32,
        "num_classes": 20,
        "debug_mode": None,
        "nms": {
            "conf_thresh": 0.1,
            "iou_thresh": 0.5,
            "topk_per_class": 10
        },
        "S": 7,
        "B": 2,
        "num_workers": 8,
        "persistent_workers": True,
    }
    config["exp_name"] += str("_" + exp_time)
    return config

def val_dataset(cfg, model, test_loader):
    model.eval()
    epoch_preds = []
    epoch_gts = []
    val_bar = tqdm(test_loader, desc=f"[test : ]")

    with torch.no_grad():
        for i, (imgs, label) in enumerate(val_bar):
            imgs = imgs.to(cfg["device"])
            label = label.to(cfg["device"])
            outputs = model(imgs)
            out_decode = decode_preds(outputs.detach(), B=2, conf_thresh=0.01)
            out_boxes = nms(out_decode, cfg["nms"]["conf_thresh"], cfg["nms"]["iou_thresh"], cfg["nms"]["topk_per_class"])
            epoch_preds.extend([b.detach().cpu() for b in out_boxes])
            label_decode = decode_labels_list(label.detach())
            epoch_gts.extend([b.detach().cpu() for b in label_decode])
        
    metrics_dict = compute_map(epoch_preds, epoch_gts, cfg["num_classes"], metrics_dtype=torch.float32, eps=1e-6)
    print(metrics_dict)

def val_main():
    cfg = base_config()
    save_config(cfg)
    train_loader, test_loader = build_dataset(cfg)
    model = build_model(cfg)
    model.load_state_dict(torch.load(cfg["model_path"]))
    model.to(cfg["device"])
    val_dataset(cfg, model, test_loader)

if __name__ == "__main__":
    val_main()