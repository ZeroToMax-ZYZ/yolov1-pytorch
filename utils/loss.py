import torch
import torch.nn as nn
import torch.nn.functional as F

from icecream import ic


class YoloLoss(nn.Module):
    def __init__(self, S, B, C, lambda_coord, lambda_noobj, ic_debug=False):
        super().__init__()
        self.S, self.B, self.C = S, B, C
        self.lambda_coord, self.lambda_noobj = lambda_coord, lambda_noobj
        self.ic_debug = ic_debug

    def _grid_mesh(self, sample_x):
        bs = sample_x.shape[0]
        device = sample_x.device   
        x = torch.arange(self.S).to(device)
        y = torch.arange(self.S).to(device)
        # 行 列
        I, J = torch.meshgrid(x, y, indexing='ij')
        # ic(I, J)
        I = I.reshape(1, self.S, self.S, 1).expand(bs, self.S, self.S, 1).to(device)
        J = J.reshape(1, self.S, self.S, 1).expand(bs, self.S, self.S, 1).to(device)
        return I, J

    
    def _xywh2xyxy(self, x, y, w, h):
        '''
        从grid cell的偏移量转换到全局的xyxy坐标
        '''
        I, J = self._grid_mesh(x)
        # x y 从偏移量转化为grid cell 坐标系
        cx, cy = x + J, y + I
        # w,h 从整图归一化 -> grid cell 坐标系
        w_cell, h_cell = w * self.S, h * self.S

        x1, y1 = cx - w_cell/2, cy - h_cell/2
        x2, y2 = cx + w_cell/2, cy + h_cell/2
        return x1, y1, x2, y2

    def _iou_xyxy(self, pred_x1, pred_y1, pred_x2, pred_y2, gt_x1, gt_y1, gt_x2, gt_y2):
        '''
        pred_x1: (2, 7, 7, 2)
        gt_x1: (2, 7, 7, 1)
        '''
        roi_x1 = torch.max(pred_x1, gt_x1)
        roi_y1 = torch.max(pred_y1, gt_y1)
        roi_x2 = torch.min(pred_x2, gt_x2)
        roi_y2 = torch.min(pred_y2, gt_y2)


        roi_w = torch.clamp(roi_x2 - roi_x1, min=0)
        roi_h = torch.clamp(roi_y2 - roi_y1, min=0)
        inter = roi_w * roi_h
        bbox1_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
        bbox2_area = (gt_x2 - gt_x1) * (gt_y2 - gt_y1)
        union = bbox1_area + bbox2_area - inter

        iou = inter / (union + 1e-6)
        return iou

        

    def forward(self, pred, target):
        '''
        pred: (batch_size, S, S, 10+1) 
        xywhc: xy相比于grid左上角的偏移，wh为归一化的尺寸， conf
        target: (batch_size, S, S, 5+1)
        '''
        bs = target.shape[0]
        gt_x = target[:, :, :, 0:1] # ([2, 7, 7, 1])
        gt_y = target[:, :, :, 1:2] # ([2, 7, 7, 1])
        gt_w = target[:, :, :, 2:3] # ([2, 7, 7, 1])
        gt_h = target[:, :, :, 3:4] # ([2, 7, 7, 1])
        gt_conf = target[:, :, :, 4:5] # ([2, 7, 7, 1])
        gt_cls = target[:, :, :, 5: 5+self.C] # ([2, 7, 7, 1])
        # pred : (2, 7, 7, 11)
        pred_2xywhc = pred[:, :, :, :5*self.B] # ([2, 7, 7, 10])
        pred_cls = pred[:, :, :, 5*self.B : 5*self.B + self.C] # ([2, 7, 7, 1])

        pred_xywhc = pred_2xywhc.reshape(bs, self.S, self.S, self.B, 5)
        pred_x = pred_xywhc[:, :, :, :, 0] # ([2, 7, 7, 2])
        pred_y = pred_xywhc[:, :, :, :, 1] # ([2, 7, 7, 2])
        pred_w_sqrt = pred_xywhc[:, :, :, :, 2] # ([2, 7, 7, 2])
        pred_h_sqrt = pred_xywhc[:, :, :, :, 3] # ([2, 7, 7, 2])

        pred_w = pred_w_sqrt ** 2
        pred_h = pred_h_sqrt ** 2

        pred_conf = pred_xywhc[:, :, :, :, 4] # ([2, 7, 7, 2])

        pred_x1, pred_y1, pred_x2, pred_y2 = self._xywh2xyxy(pred_x, pred_y, pred_w, pred_h)
        gt_x1, gt_y1, gt_x2, gt_y2 = self._xywh2xyxy(gt_x, gt_y, gt_w, gt_h)
        # iou : ([2, 7, 7, 2])
        iou = self._iou_xyxy(pred_x1, pred_y1, pred_x2, pred_y2, gt_x1, gt_y1, gt_x2, gt_y2)
        # iou_max: ([2, 7, 7])
        # iou_index: ([2, 7, 7])
        iou_max, iou_index = torch.max(iou, dim=3)
        # 负责的bbox的mask
        iou_index_mask = F.one_hot(iou_index, num_classes=self.B) # ([2, 7, 7, 2])
        # 有物体的grid cell的mask
        grid_mask = gt_conf.expand(bs, self.S, self.S, self.B) # ([2, 7, 7, 1]) --> ([2, 7, 7, 2])
        # 找到有物体的grid cell，同时bbox负责的mask
        grid_bbox_mask = grid_mask * iou_index_mask
        grid_cell_mask = gt_conf
        # 对齐维度
        gt_x = gt_x.expand(bs, self.S, self.S, self.B)
        gt_y = gt_y.expand(bs, self.S, self.S, self.B)
        gt_w = gt_w.expand(bs, self.S, self.S, self.B)
        gt_h = gt_h.expand(bs, self.S, self.S, self.B)
        gt_conf = gt_conf.expand(bs, self.S, self.S, self.B)

        iou_max = iou_max.reshape(bs, self.S, self.S, 1).expand(bs, self.S, self.S, self.B)

        # 第一行：xy坐标损失 预测-gt
        loss_xy = grid_bbox_mask * ((pred_x- gt_x)**2 + (pred_y - gt_y)**2)

        # 第二行：wh损失
        # 注意，模型预测的是根号wh
        loss_wh = grid_bbox_mask * ((pred_w_sqrt - torch.sqrt(gt_w))**2 + (pred_h_sqrt - torch.sqrt(gt_h))**2)

        # 第三行：conf损失
        loss_conf = grid_bbox_mask * (pred_conf - iou_max.detach())**2

        # 第四行：不负责预测的conf损失
        loss_conf_no_obj = (1 - grid_bbox_mask) * (pred_conf)**2

        # 第五行： 类别损失
        loss_classes = grid_cell_mask * (pred_cls - gt_cls)**2

        loss = self.lambda_coord * loss_xy.sum() + self.lambda_coord * loss_wh.sum() + loss_conf.sum() + self.lambda_noobj * loss_conf_no_obj.sum() + loss_classes.sum()

        return loss / bs
    
        if self.ic_debug:
            ic(gt_x.shape)
            ic(pred_2xywhc.shape)
            ic(pred_xywhc.shape)
            ic(pred_x.shape)
            ic(pred_cls.shape)
            ic(iou.shape)
            ic(iou_max.shape)
            ic(iou_index.shape)
            ic(iou_index_mask.shape)
            ic(gt_conf.shape)
            ic(grid_mask.shape)
    
if __name__ == "__main__":
    test_label = torch.randn(2, 7, 7, 6)
    test_pred = torch.randn(2, 7, 7, 11)
    loss = YoloLoss(S=7, B=2, C=1, lambda_coord=5, lambda_noobj=0.5, ic_debug=True)
    ic(loss(test_pred, test_label))