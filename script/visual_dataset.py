# -*- coding: utf-8 -*-
"""
可视化 YOLOv1 数据增强效果（1 张原图 + 9 张增强图）

入口：
    visualize_yolov1_augmentations_grid(
        base_path=...,
        img_size=448,
        idx=None,
        n_aug=9,
        seed=123,
        save_path=None,
    )

出口：
    - 使用 matplotlib 弹窗显示 10 张图（2x5）
    - 可选：保存到本地图片文件

依赖假设（与你的工程一致）：
    1) base_path 目录结构：
        base_path/
            images/
            targets/   (csv: name,x_min,y_min,x_max,y_max)
    2) 你的增强函数可用：
        from dataset.augment import build_yolov1_transforms
    3) 你的 VOCDataset 可用（用于复用 samples 收集逻辑）：
        from dataset.VOC_dataset import VOCDataset, VOC_CLASSES, read_voc_csv

说明：
    - “原图+gt”：这里为了和增强图对齐显示尺寸，默认将原图 resize 到 img_size，并按比例缩放 bbox。
    - “增强图”：对同一张原图重复调用 train_transform 9 次（随机增强），并绘制变换后的 bbox 与类别名。
"""

from __future__ import annotations

import os
import random
from typing import Optional, Tuple, List, Sequence, Dict

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import torch

# 你的工程内模块
from dataset.augment import build_yolov1_transforms
from dataset.VOC_dataset import VOCDataset, VOC_CLASSES, read_voc_csv


def _set_seed(seed: int) -> None:
    """
    功能：
        统一设置随机种子，尽可能保证可复现（albumentations 主要依赖 numpy/random）

    输入：
        seed: 随机种子

    输出：
        无
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _resize_image_and_boxes(
    img_rgb: np.ndarray,
    bboxes_xyxy: Sequence[Sequence[float]],
    out_size: int,
) -> Tuple[np.ndarray, List[List[float]]]:
    """
    功能：
        将原图 resize 到 out_size x out_size，同时按比例缩放 bbox。

    输入：
        img_rgb: HxWx3, uint8
        bboxes_xyxy: [[x1,y1,x2,y2], ...]（像素坐标，基于原图）
        out_size: 目标尺寸

    输出：
        img_resized: out_size x out_size x 3
        bboxes_resized: 缩放后的 bbox 列表（float）
    """
    h0, w0 = img_rgb.shape[:2]
    img_resized = cv2.resize(img_rgb, (out_size, out_size), interpolation=cv2.INTER_LINEAR)

    sx = float(out_size) / float(w0)
    sy = float(out_size) / float(h0)

    bboxes_out: List[List[float]] = []
    for b in bboxes_xyxy:
        x1, y1, x2, y2 = float(b[0]), float(b[1]), float(b[2]), float(b[3])
        bboxes_out.append([x1 * sx, y1 * sy, x2 * sx, y2 * sy])

    return img_resized, bboxes_out


def _denormalize_chw_to_hwc_uint8(
    img_chw: np.ndarray,
    mean: Tuple[float, float, float],
    std: Tuple[float, float, float],
) -> np.ndarray:
    """
    功能：
        将 Normalize 后的 CHW float 图像反归一化，并转为 HWC uint8 以便 matplotlib 显示。

    输入：
        img_chw:
            (3,H,W)，通常来自 ToTensorV2 + Normalize，float32
        mean/std:
            Normalize 用的 mean/std

    输出：
        img_hwc_uint8:
            (H,W,3)，uint8，范围 [0,255]
    """
    # 反归一化：img = img * std + mean
    mean_arr = np.asarray(mean, dtype=np.float32).reshape(3, 1, 1)
    std_arr = np.asarray(std, dtype=np.float32).reshape(3, 1, 1)

    img = img_chw.astype(np.float32) * std_arr + mean_arr
    img = np.clip(img, 0.0, 1.0)  # Normalize 后反推回 [0,1] 并裁剪
    img_hwc = np.transpose(img, (1, 2, 0))  # CHW -> HWC
    img_uint8 = (img_hwc * 255.0 + 0.5).astype(np.uint8)
    return img_uint8


def _draw_boxes_on_ax(
    ax: plt.Axes,
    img_hwc_uint8: np.ndarray,
    bboxes_xyxy: Sequence[Sequence[float]],
    class_ids: Sequence[int],
    id_to_class: Dict[int, str],
    title: str,
) -> None:
    """
    功能：
        在 matplotlib 的 Axes 上显示图像并绘制 bbox 与类别名。

    输入：
        ax: matplotlib Axes
        img_hwc_uint8: (H,W,3) uint8
        bboxes_xyxy: [[x1,y1,x2,y2], ...]
        class_ids: [cid, ...]
        id_to_class: 类别 id -> 类别名
        title: 子图标题

    输出：
        无
    """
    ax.imshow(img_hwc_uint8)
    ax.set_title(title)
    ax.axis("off")

    if len(bboxes_xyxy) == 0:
        return

    # 简单配色：按类别 id 固定一个颜色（可重复但稳定）
    for b, cid in zip(bboxes_xyxy, class_ids):
        x1, y1, x2, y2 = float(b[0]), float(b[1]), float(b[2]), float(b[3])
        w = max(1.0, x2 - x1)
        h = max(1.0, y2 - y1)

        # 使用 HSV 生成稳定颜色（不同 cid 不同色）
        hue = (cid * 37) % 360  # 37 是个常用的“打散”系数
        color = plt.cm.hsv(hue / 360.0)

        rect = patches.Rectangle(
            (x1, y1),
            w,
            h,
            linewidth=2,
            edgecolor=color,
            facecolor="none",
        )
        ax.add_patch(rect)

        cls_name = id_to_class.get(int(cid), str(cid))
        ax.text(
            x1,
            max(0.0, y1 - 2.0),
            cls_name,
            fontsize=9,
            color=color,
            bbox=dict(facecolor="black", alpha=0.4, pad=1),
        )


def visualize_yolov1_augmentations_grid(
    base_path: str,
    img_size: int = 448,
    idx: Optional[int] = None,
    n_aug: int = 9,
    seed: int = 123,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    save_path: Optional[str] = None,
) -> None:
    """
    功能：
        可视化 YOLOv1 数据增强效果：
        - 1 张“原图 + GT”（resize 到 img_size 显示）
        - 9 张随机增强结果（同一张原图重复增强）

    输入：
        base_path:
            数据集根目录（包含 images/ 与 targets/）
        img_size:
            输出可视化尺寸（与增强 pipeline 对齐）
        idx:
            指定可视化第 idx 张样本；为 None 时随机抽一张
        n_aug:
            增强图数量（默认 9）
        seed:
            随机种子（便于复现实验）
        mean/std:
            用于反归一化显示（必须与你 Normalize 保持一致）
        save_path:
            若不为 None，则保存可视化网格图到该路径（如 "aug_vis.png"）

    输出：
        - 弹窗显示图像网格
        - 可选写盘保存
    """
    if n_aug != 9:
        # 你要求 1+9=10 张，这里仍允许改，但默认按 9 张增强
        pass

    _set_seed(seed)

    # 1) 构建增强
    train_transform, _ = build_yolov1_transforms(img_size=img_size, mean=mean, std=std)

    # 2) 用 VOCDataset 复用 samples 收集逻辑
    ds = VOCDataset(base_path=base_path, transform=None, img_size=img_size, S=7, classes=VOC_CLASSES)
    if len(ds) == 0:
        raise RuntimeError(f"在 {base_path} 下未找到可用样本（检查 images/ 与 targets/ 是否匹配）")

    if idx is None:
        idx = int(np.random.randint(0, len(ds)))
    idx = int(max(0, min(len(ds) - 1, idx)))

    sample = ds.samples[idx]
    class_to_id = {name: i for i, name in enumerate(VOC_CLASSES)}
    id_to_class = {i: name for name, i in class_to_id.items()}

    # 3) 读原图 + 原始 bbox
    img_bgr = cv2.imread(sample.img_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"读取图像失败：{sample.img_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    bboxes, class_ids = read_voc_csv(sample.csv_path, class_to_id)

    # 4) “原图+gt”：resize 到 img_size，并缩放 bbox，便于与增强图同尺度对比
    img0_resized, bboxes0_resized = _resize_image_and_boxes(img_rgb, bboxes, img_size)

    # 5) 对同一张原图重复做 n_aug 次增强
    aug_imgs: List[np.ndarray] = []
    aug_bboxes: List[List[List[float]]] = []
    aug_cids: List[List[int]] = []

    for _ in range(n_aug):
        out = train_transform(image=img_rgb, bboxes=bboxes, class_labels=class_ids)

        img_t = out["image"]          # torch.Tensor(CHW), float32, Normalize 后
        b_t = list(out["bboxes"])     # List[Tuple[float,float,float,float]]，像素坐标
        c_t = list(out["class_labels"])

        # 转 numpy 以反归一化显示
        if isinstance(img_t, torch.Tensor):
            img_chw = img_t.detach().cpu().numpy()
        else:
            # 理论上 ToTensorV2 会返回 torch.Tensor，这里做兜底
            img_chw = np.asarray(img_t)

        img_uint8 = _denormalize_chw_to_hwc_uint8(img_chw, mean=mean, std=std)

        aug_imgs.append(img_uint8)
        aug_bboxes.append([list(map(float, bb)) for bb in b_t])
        aug_cids.append([int(c) for c in c_t])

    # 6) 绘图：2x5（共 10 张）
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.reshape(-1)

    # 原图（不做 Normalize，直接 uint8 显示）
    _draw_boxes_on_ax(
        ax=axes[0],
        img_hwc_uint8=img0_resized,
        bboxes_xyxy=bboxes0_resized,
        class_ids=class_ids,
        id_to_class=id_to_class,
        title=f"original (idx={idx})",
    )

    # 增强图（反归一化后显示）
    for i in range(n_aug):
        _draw_boxes_on_ax(
            ax=axes[i + 1],
            img_hwc_uint8=aug_imgs[i],
            bboxes_xyxy=aug_bboxes[i],
            class_ids=aug_cids[i],
            id_to_class=id_to_class,
            title=f"aug #{i + 1}",
        )

    # 如果 n_aug < 9，空子图关掉（仍保持 2x5 版式）
    for j in range(n_aug + 1, 10):
        axes[j].axis("off")

    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=600)

    plt.show()


# =========================
# 直接运行示例（不使用 argparse，按你的偏好用配置字典）
# =========================
if __name__ == "__main__":
    cfg = {
        "train_path": r"D:\1AAAAAstudy\python_base\pytorch\all_dataset\YOLOv1_dataset\train",  # 改成你的 base_path
        "input_size": 448,
        "seed": 12,
        "idx": None,          # None 表示随机抽一张；也可以填具体整数
        "save_path": r'readme\img',    # 例如 r"./aug_vis.png"
    }

    visualize_yolov1_augmentations_grid(
        base_path=cfg["train_path"],
        img_size=cfg["input_size"],
        idx=cfg["idx"],
        n_aug=9,
        seed=cfg["seed"],
        save_path=cfg["save_path"],
    )
