'''
把模型的输出和标注文件
从xywh-conf-cls(相对于grid cell左上角的偏移【0-1】)
转化为xyxyconf cls(相对于全图，在grid cell坐标系下的偏移【0-7】)
'''
import torch

from icecream import ic
from dataclasses import dataclass

@dataclass
class LabelDecode:
    bboxes: torch.Tensor
    labels: torch.Tensor

@dataclass
class PredDecode:
    bboxes: torch.Tensor
    labels: torch.Tensor
    confs: torch.Tensor
    

def _meshgrid(gt_x):
    bs = gt_x.shape[0]
    S = gt_x.shape[1]
    last_dim = gt_x.shape[-1]
    device, dtype = gt_x.device, gt_x.dtype
    x = torch.arange(S, device=device, dtype=dtype)
    y = torch.arange(S, device=device, dtype=dtype)
    ii, jj = torch.meshgrid(x, y, indexing="ij")
    ii = ii.reshape(1, S, S, 1).expand(bs, S, S, last_dim).to(device)
    jj = jj.reshape(1, S, S, 1).expand(bs, S, S, last_dim).to(device)
    return ii, jj

def _xywh2xyxy(x, y, w, h):
    '''
    由grid内部偏移量转化为grid坐标系
    '''
    bs = x.shape[0]
    S = x.shape[1]

    ii, jj = _meshgrid(x)
    # 从中心点相对于所属grid cell左上角的偏移 转换到 相对于全图左上角的偏移量（grid 坐标系下）
    c_x = x + jj
    c_y = y + ii
    # w h 由归一化转化为grid坐标系
    w = w * S
    h = h * S

    x1 = c_x - w / 2
    y1 = c_y - h / 2
    x2 = c_x + w / 2
    y2 = c_y + h / 2

    return x1, y1, x2, y2

def decode_labels(gt):
    '''
    bs*7*7*5+C
    x-y-w-h-conf-cls
    从偏移量转化为grid坐标系
    '''
    bs = gt.shape[0]
    S = gt.shape[1]
    gt_x = gt[:, :, :, 0:1] # ([2, 7, 7, 1])
    gt_y = gt[:, :, :, 1:2] # ([2, 7, 7, 1])
    gt_w = gt[:, :, :, 2:3] # ([2, 7, 7, 1])
    gt_h = gt[:, :, :, 3:4] # ([2, 7, 7, 1])
    gt_conf = gt[:, :, :, 4:5] # ([2, 7, 7, 1])
    gt_cls = gt[:, :, :, 5:] # ([2, 7, 7, 1])
    # ([2, 7, 7, 1])
    gt_x1, gt_y1, gt_x2, gt_y2 = _xywh2xyxy(gt_x, gt_y, gt_w, gt_h)
    # [nums, xyxy-conf-cls]
    # 但是有一个隐患就是没有区分同batch的不同图片
    # 确实不可以忽略batch，因为后续还需要nms
    # 2-7-7-6
    out_bbox = torch.cat((gt_x1, gt_y1, gt_x2, gt_y2, gt_conf, gt_cls),dim=-1)
    # ic(out_bbox.shape)
    # ic(out_bbox.shape)
    return out_bbox

def decode_labels_list(gt):
    '''
    bs*7*7*5+C
    x-y-w-h-conf-cls
    从偏移量转化为grid坐标系
    '''
    bs = gt.shape[0]
    S = gt.shape[1]
    gt_x = gt[:, :, :, 0:1] # ([2, 7, 7, 1])
    gt_y = gt[:, :, :, 1:2] # ([2, 7, 7, 1])
    gt_w = gt[:, :, :, 2:3] # ([2, 7, 7, 1])
    gt_h = gt[:, :, :, 3:4] # ([2, 7, 7, 1])
    gt_conf = gt[:, :, :, 4:5] # ([2, 7, 7, 1])
    gt_cls = gt[:, :, :, 5:] # ([2, 7, 7, 20])
    # ([2, 7, 7, 1])
    gt_x1, gt_y1, gt_x2, gt_y2 = _xywh2xyxy(gt_x, gt_y, gt_w, gt_h)

    # ([2, 7, 7, 20]) --> ([2, 7, 7, 1])
    gt_cls_argmax = gt_cls.argmax(dim=-1,keepdim=True)
    # ([2, 7, 7, 5])
    comb_gt = torch.cat((gt_x1, gt_y1, gt_x2, gt_y2, gt_cls_argmax),dim=-1)
    obj_mask = (gt_conf[:, :, :, 0] > 0.5) # 2-7-7

    out_list = []
    # ic(obj_mask.shape)
    for batch in range(bs):
        batch_list = []
        batch_comb = comb_gt[batch]
        batch_mask = obj_mask[batch]

        picked = batch_comb[batch_mask]

        if picked.numel() == 0:
            out_list.append(torch.zeros((0, 5), device=gt.device, dtype=gt.dtype))
        else:
            out_list.append(picked)
            
    return out_list



def decode_preds(preds, B=2, conf_thresh=0.01):
    '''
    bs*7*7* 10 + C
    x-y-w-h-conf-cls
    '''
    bs = preds.shape[0]
    S = preds.shape[1]

    pred_2xywhc = preds[:, :, :, :10] # ([2, 7, 7, 10])
    pred_xywhc = pred_2xywhc.reshape(bs, S, S, 2, 5) # ([2, 7, 7, 2, 5])
    pred_x = pred_xywhc[:, :, :, :, 0] # ([2, 7, 7, 2])
    pred_y = pred_xywhc[:, :, :, :, 1] # ([2, 7, 7, 2])
    pred_w_sqrt = pred_xywhc[:, :, :, :, 2] # ([2, 7, 7, 2])
    pred_h_sqrt = pred_xywhc[:, :, :, :, 3] # ([2, 7, 7, 2])

    pred_w = pred_w_sqrt ** 2
    pred_h = pred_h_sqrt ** 2

    pred_conf = pred_xywhc[:, :, :, :, 4] # ([2, 7, 7, 2])
    pred_cls = preds[:, :, :, 10:]
    # ([2, 7, 7, 2])
    pred_x1, pred_y1, pred_x2, pred_y2 = _xywh2xyxy(pred_x, pred_y, pred_w, pred_h)
    # 扩展类别
    num_classes = pred_cls.shape[-1]
    # ([2, 7, 7, 2, 1])
    pred_cls = pred_cls.reshape(bs, S, S, 1, num_classes).expand(bs, S, S, B, num_classes)
    pred_x1 = pred_x1.reshape(bs, S, S, B, 1)
    pred_y1 = pred_y1.reshape(bs, S, S, B, 1)
    pred_x2 = pred_x2.reshape(bs, S, S, B, 1)
    pred_y2 = pred_y2.reshape(bs, S, S, B, 1)
    pred_conf = pred_conf.reshape(bs, S, S, B, 1)

    # ([2, 7, 7, 2, 6])
    out_pred = torch.cat((pred_x1, pred_y1, pred_x2, pred_y2, pred_conf, pred_cls),dim=-1)
    # conf filter 
    # ic(out_pred.shape)
    # mask = pred_conf.reshape(bs, S, S, B) > conf_thresh
    # out_bbox = out_pred[mask]
    # ic(out_bbox.shape)
    return out_pred



    # ic(out_pred.shape)

if __name__ == "__main__":
    test_gt = torch.randn(2, 7, 7, 25)
    decode_labels_list(test_gt) # [num, 6] num为标签的数量, 6为   x1-y1-x2-y2-conf-cls
    # test_pred = torch.randn(2, 7, 7, 11)
    # decode_preds(test_pred) # [num, 6] num为预测框中经过conf过滤后的数量, 6为   x1-y1-x2-y2-conf-cls
