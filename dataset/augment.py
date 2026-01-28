# -*- coding: utf-8 -*-
from typing import Tuple, Literal, Optional
import cv2
import os
# 避免albumentations更新警告
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
import albumentations as A
from albumentations.pytorch import ToTensorV2


def build_yolov1_transforms(
    img_size: int = 448,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    bbox_min_area: float = 1.0,
    bbox_min_visibility: float = 0.10,
    pad_value: int = 114,
) -> Tuple[A.Compose, A.Compose]:

    bbox_params = A.BboxParams(
        format="pascal_voc",
        label_fields=["class_labels"],
        min_area=bbox_min_area,
        min_visibility=bbox_min_visibility,
    )
    # -------------------------
    # 训练增强：jitter + flip + color distortion（YOLOv1风格近似）
    # -------------------------
    train_transform = A.Compose([
        # 1) 尺度抖动（近似 YOLOv1 的 random scale/jitter）
        A.RandomScale(scale_limit=0.20, p=0.50),

        # 2) Pad 再随机裁剪回固定尺寸（近似 random translation/crop）
        A.PadIfNeeded(
            min_height=img_size,
            min_width=img_size,
            border_mode=cv2.BORDER_CONSTANT,
            value=(pad_value, pad_value, pad_value),
            p=1.0,
        ),
        A.RandomCrop(height=img_size, width=img_size, p=0.50),

        # 3) 最终强制到固定尺寸（防止尺度变化后尺寸不一致）
        A.Resize(height=img_size, width=img_size, p=1.0),

        # 4) 水平翻转（YOLO/检测常见）
        A.HorizontalFlip(p=0.50),

        # 5) 颜色扰动（YOLOv1 论文里的 hue/sat/exposure 近似）
        A.HueSaturationValue(
            hue_shift_limit=10,     # hue 抖动
            sat_shift_limit=30,     # saturation 抖动
            val_shift_limit=30,     # value/exposure 抖动
            p=0.80,
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.20,
            contrast_limit=0.20,
            p=0.20,
        ),
        A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
        ToTensorV2(),
    ], bbox_params=bbox_params)

    # -------------------------
    # 验证增强：仅 Resize +（可选）Normalize
    # -------------------------
    val_transform = A.Compose([
        A.Resize(height=img_size, width=img_size, p=1.0),
        A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
        ToTensorV2()
    ], bbox_params=bbox_params)

    return train_transform, val_transform
