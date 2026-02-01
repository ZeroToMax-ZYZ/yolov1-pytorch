import torch
import torch.nn as nn
import torch.nn.functional as F
from icecream import ic

class YOLO_Loss(nn.Module):
    def __init__(self, S=7, B=2, C=20, ic_debug=False):
        super().__init__()
        self.ic_debug = ic_debug
        self.S = int(S)
        self.B = int(B)
        self.C = int(C)
        
        # 权重超参 (YOLOv1论文标准)
        self.lambda_coord = 5.0
        self.lambda_noobj = 0.5
        
        self.eps = 1e-6

    def _meshgrid(self, x):
        """
        生成 grid 网格坐标
        """
        device = x.device
        dtype = x.dtype
        N = x.shape[0]
        
        # ys: 0,1,..,6 (行)
        # xs: 0,1,..,6 (列)
        ys = torch.arange(self.S, device=device, dtype=dtype)
        xs = torch.arange(self.S, device=device, dtype=dtype)
        
        # gy: y坐标, gx: x坐标
        gy, gx = torch.meshgrid(ys, xs, indexing="ij") 
        
        # [N, S, S, 1]
        grid_x = gx.view(1, self.S, self.S, 1).expand(N, self.S, self.S, 1)
        grid_y = gy.view(1, self.S, self.S, 1).expand(N, self.S, self.S, 1)
        return grid_x, grid_y

    def _xywh2xyxy(self, x, y, w, h):
        """
        将 (cell_x, cell_y, w, h) 转换为 (x1, y1, x2, y2) 用于计算 IoU
        
        输入:
            x, y: cell 内部偏移 (0~1)
            w, h: 必须是【线性】的宽和高 (0~1 归一化值)，不能是 sqrt
        输出:
            x1, y1, x2, y2: 相对 grid 坐标系 (0~S)
        """
        grid_x, grid_y = self._meshgrid(x)
        
        # 转换到 grid 坐标系 (0~7)
        cx = x + grid_x
        cy = y + grid_y
        
        # w, h 转换为 grid 单位
        w_cell = w * self.S
        h_cell = h * self.S
        
        x1 = cx - 0.5 * w_cell
        y1 = cy - 0.5 * h_cell
        x2 = cx + 0.5 * w_cell
        y2 = cy + 0.5 * h_cell
        return x1, y1, x2, y2

    def _iou_xyxy(self, b1_x1, b1_y1, b1_x2, b1_y2, b2_x1, b2_y1, b2_x2, b2_y2):
        """
        计算 IoU
        输入形状通常为 [N, S, S, B]
        """
        inter_x1 = torch.max(b1_x1, b2_x1)
        inter_y1 = torch.max(b1_y1, b2_y1)
        inter_x2 = torch.min(b1_x2, b2_x2)
        inter_y2 = torch.min(b1_y2, b2_y2)
        
        inter_w = (inter_x2 - inter_x1).clamp(min=0)
        inter_h = (inter_y2 - inter_y1).clamp(min=0)
        inter_area = inter_w * inter_h
        
        area1 = (b1_x2 - b1_x1).clamp(min=0) * (b1_y2 - b1_y1).clamp(min=0)
        area2 = (b2_x2 - b2_x1).clamp(min=0) * (b2_y2 - b2_y1).clamp(min=0)
        
        union = area1 + area2 - inter_area
        return inter_area / (union + self.eps)

    def forward(self, pred, label):
        """
        pred:  (N, S, S, 5*B + C) -> [x, y, sqrt(w), sqrt(h), conf, ..., classes]
        label: (N, S, S, 5 + C)   -> [x, y, w, h, obj, classes]
        """
        N = pred.shape[0]
        
        # ==========================
        # 1. 解析 Label (GT)
        # ==========================
        # GT 中的 w, h 是线性的 0~1
        gt_x = label[..., 0:1]
        gt_y = label[..., 1:2]
        gt_w = label[..., 2:3]
        gt_h = label[..., 3:4]
        gt_obj_mask = label[..., 4:5] # (N, S, S, 1) 有物体为 1
        gt_class = label[..., 5:]
        
        # ==========================
        # 2. 解析 Pred
        # ==========================
        # pred_bbox: (N, S, S, B, 5)
        pred_bbox = pred[..., :self.B*5].view(N, self.S, self.S, self.B, 5)
        
        pred_x = pred_bbox[..., 0] 
        pred_y = pred_bbox[..., 1]
        pred_sqrt_w = pred_bbox[..., 2] # 模型输出的是 sqrt(w)
        pred_sqrt_h = pred_bbox[..., 3] # 模型输出的是 sqrt(h)
        pred_conf = pred_bbox[..., 4]
        
        pred_class = pred[..., self.B*5:] # (N, S, S, C)

        # ==========================
        # 3. 准备 IoU 计算所需的数据
        # ==========================
        # 将 GT 广播到 B 个 anchor，以便跟 Pred 对应计算
        # GT shape: (N, S, S, B)
        gt_x_expand = gt_x.expand(-1, -1, -1, self.B)
        gt_y_expand = gt_y.expand(-1, -1, -1, self.B)
        gt_w_expand = gt_w.expand(-1, -1, -1, self.B)
        gt_h_expand = gt_h.expand(-1, -1, -1, self.B)

        # 【关键点1】还原 Pred 的线性宽高用于计算 IoU
        # 模型输出是 sqrt，所以平方回来。加上 clamp 防止负数平方导致的异常（虽然理论上不会）
        pred_w_linear = pred_sqrt_w ** 2
        pred_h_linear = pred_sqrt_h ** 2
        
        # 转换坐标系 -> (x1, y1, x2, y2)
        gt_x1, gt_y1, gt_x2, gt_y2 = self._xywh2xyxy(gt_x_expand, gt_y_expand, gt_w_expand, gt_h_expand)
        pr_x1, pr_y1, pr_x2, pr_y2 = self._xywh2xyxy(pred_x, pred_y, pred_w_linear, pred_h_linear)
        
        # 计算 IoU: (N, S, S, B)
        iou_scores = self._iou_xyxy(gt_x1, gt_y1, gt_x2, gt_y2, pr_x1, pr_y1, pr_x2, pr_y2)
        
        # ==========================
        # 4. 负责框选择 (Best IoU)
        # ==========================
        # 找到每个 cell 中 IoU 最大的那个 bbox 索引
        # max_iou_val: (N, S, S, 1)
        # max_iou_idx: (N, S, S, 1)
        max_iou_val, max_iou_idx = iou_scores.max(dim=-1, keepdim=True)
        
        # 生成负责框 Mask (N, S, S, B)
        # 只有 IoU 最大的那个位置是 1
        is_best_box_mask = torch.zeros_like(iou_scores).scatter_(-1, max_iou_idx, 1.0)
        
        # 最终的 Mask
        # obj_mask: 有物体 且 是负责框 (N, S, S, B)
        obj_mask = gt_obj_mask.expand(-1, -1, -1, self.B) * is_best_box_mask
        
        

        # ==========================
        # 5. 计算 Loss
        # ==========================
        
        # --- (A) Coordinate Loss (x, y) ---
        # 只在 obj_mask 激活的地方计算
        loss_xy = torch.sum(
            obj_mask * ((pred_x - gt_x_expand)**2 + (pred_y - gt_y_expand)**2)
        )
        
        # --- (B) Coordinate Loss (w, h) ---
        # 【关键点2】Loss 回归是在 Sqrt 空间进行的
        # Pred 已经是 sqrt 了，所以要把 GT 开根号
        sqrt_gt_w = torch.sqrt(gt_w_expand.clamp(min=self.eps))
        sqrt_gt_h = torch.sqrt(gt_h_expand.clamp(min=self.eps))
        
        # 允许 pred_sqrt 为负 (虽然 logic 上不应该，但 MSE 会把它拉回来)
        # 如果你希望严格正数，可以在 decode 时处理，loss 这里直接 MSE 即可
        loss_wh = torch.sum(
            obj_mask * ((pred_sqrt_w - sqrt_gt_w)**2 + (pred_sqrt_h - sqrt_gt_h)**2)
        )
        
        # --- (C) Confidence Loss (Object) ---
        # 目标值：使用真实的 IoU (YOLOv1 推荐)
        # 【关键点3】维度修复：iou_scores 是 (N,S,S,B)，可以直接减，不需要手动 expand
        # 只要保证 iou_scores 不要传梯度回去 (detach)
        conf_target_obj = iou_scores.detach()
        loss_conf_obj = torch.sum(
            obj_mask * (pred_conf - conf_target_obj)**2
        )
        
        # --- (D) Confidence Loss (No Object) ---
        # 目标值：0
        # noobj_mask: (N, S, S, B)
        # 1. 本来就没物体的 cell
        # 2. 有物体但不是负责的那个 bbox
        
        noobj_mask = 1.0 - obj_mask
        # expand_gt_grid_mask = gt_obj_mask.expand(-1, -1, -1, self.B)
        # noobj_mask = (1 - expand_gt_grid_mask) + expand_gt_grid_mask*(1 - obj_mask)
        loss_conf_noobj = torch.sum(
            noobj_mask * (pred_conf - 0.0)**2
        )
        
        # --- (E) Class Loss ---
        # 只在有物体的 cell 计算 (N, S, S, 1)
        # 【关键点4】去掉 Softmax，直接使用 MSE
        # gt_class: (N, S, S, C), pred_class: (N, S, S, C)
        loss_class = torch.sum(
            gt_obj_mask * ((pred_class - gt_class)**2).sum(dim=-1, keepdim=True)
        )
        
        # ==========================
        # 6. 汇总
        # ==========================
        total_loss = (
            self.lambda_coord * (loss_xy + loss_wh) + 
            loss_conf_obj + 
            self.lambda_noobj * loss_conf_noobj + 
            loss_class
        )
        
        # Debug 打印
        if self.ic_debug:
            print(f"\n[Loss Debug]")
            print(f"  xy: {loss_xy.item():.4f}")
            print(f"  wh: {loss_wh.item():.4f}")
            print(f"  conf_obj: {loss_conf_obj.item():.4f}")
            print(f"  conf_noobj: {loss_conf_noobj.item():.4f}")
            print(f"  class: {loss_class.item():.4f}")
            print(f"  total (sum): {total_loss.item():.4f}")

        # 按 Batch 平均
        return total_loss / float(N)

if __name__ == '__main__':
    # 测试代码
    B, S, C = 2, 7, 20
    pred = torch.sigmoid(torch.randn(2, S, S, B*5 + C)) # 模拟输出
    label = torch.rand(2, S, S, 5 + C)
    label[..., 4] = (label[..., 4] > 0.5).float() # obj mask 0/1
    
    loss_func = YOLO_Loss(S=S, B=B, C=C, ic_debug=True)
    loss = loss_func(pred, label)
    print(f"Final Batch Loss: {loss.item()}")