'''
计算目标检测相关指标
包括:
map50
map50:95

输入约定:
1) preds_nms_all:
    List[torch.Tensor],长度=图片数
    每张图一个 tensor,形状:
        (Ni, 6) = [x1, y1, x2, y2, score, cls_id]
    - 坐标:grid 坐标系（0~S）
    - score:通常来自你的 nms 里 conf * cls_score
    - cls_id:在 nms 里被拼成 float（需要在 metrics 中转 long）

2) gts_all（ decode_labels_list 的输出）:
    List[torch.Tensor],长度=图片数
    每张图一个 tensor,形状:
        (Mi, 5) = [x1, y1, x2, y2, cls_id]
    - 坐标:grid 坐标系（0~S）
    - cls_id:是 float,需要转 long

'''

import torch
import numpy as np

from icecream import ic

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


def compute_map(preds_nms_all, gts_all):
    iou_thresh = [round(i, 2) for i in np.arange(0.50, 0.96, 0.05)]
    ic(iou_thresh)
    



if __name__ == "__main__":
    test_preds_nms_all = [torch.randn(5, 6),
                          torch.randn(3, 6),]
    test_gts_all = [torch.randn(4, 5),
                    torch.randn(2, 5),]
    compute_map(test_preds_nms_all, test_gts_all)
