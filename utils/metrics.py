'''
计算目标检测相关指标
包括：
map50
map50:95
'''
import torch
import numpy as np


def box_iou(box1, box2):
    '''
    box1 [4]: xyxy 最大的box
    box2 [nums, 4]除了最大的以外所有的box
    '''
    inter_x1 = torch.max(box1[0], box2[:, 0])
    inter_x2 = torch.min(box1[2], box2[:, 2])
    inter_y1 = torch.max(box1[1], box2[:, 1])
    inter_y2 = torch.min(box1[3], box2[:, 3])
    
    inter = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
    
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    
    union = box1_area + box2_area - inter + 1e-6 # 1e-6 防止除0
    iou = inter / union
    
    return iou


def 



