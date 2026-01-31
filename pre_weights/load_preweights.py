"""
函数签名：
    def load_backbone_pretrained_to_detector(
        detector: torch.nn.Module,
        ckpt_path: str,
        *,
        classifier_key_prefix: str = "conv_backbone",
        detector_key_prefix: str = "backbone",
        map_location: str = "cpu",
        strict: bool = False,
        verbose: bool = True,
    ) -> dict:

参数解释：
    detector:
        - 你的检测网络实例（YOLOv1），要求内部有 detector.backbone（nn.Sequential）。
    ckpt_path:
        - 预训练权重的路径（.pt/.pth），可来自你训练 YOLOv1_Classifier 保存的 state_dict 或 checkpoint。
    classifier_key_prefix:
        - 你分类网络里承载 backbone 的参数前缀。
        - 你当前 YOLOv1_Classifier 里骨干是 self.conv_backbone，所以默认是 "conv_backbone"。
        - 若你保存的权重是 YOLOv1_Classifier 整体 state_dict，则 key 通常形如：
          "conv_backbone.0.conv.weight" ...
    detector_key_prefix:
        - 你检测网络里 backbone 的参数前缀。
        - 你当前 YOLOv1 里骨干是 self.backbone，所以默认是 "backbone"。
        - 检测网络需要的 key 形如：
          "backbone.0.conv.weight" ...
    map_location:
        - torch.load 用的 map_location，一般 "cpu" 最稳。
    strict:
        - 是否严格匹配 detector 的全部参数。
        - 检测网络只需要加载 backbone 部分，默认 strict=False 更合理。
    verbose:
        - 是否打印加载统计信息（匹配/缺失/多余 key）。

"""

from typing import Dict, Any, Tuple
import torch
import torch.nn as nn
from nets.yolov1 import YOLOv1

def load_backbone_pretrained_to_detector(
    detector: nn.Module,
    ckpt_path: str,
    *,
    classifier_key_prefix: str = "conv_backbone",
    detector_key_prefix: str = "backbone",
    map_location: str = "cpu",
    strict: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    功能：
        将你训练好的分类骨干（YOLOv1_Classifier.conv_backbone）权重加载到检测网络（YOLOv1.backbone）。

    输入：
        detector:
            - 检测网络 YOLOv1 实例，要求有属性 detector.backbone。
        ckpt_path:
            - 预训练权重路径（.pt/.pth），可以是：
                A) 直接 state_dict（dict[str, Tensor]）
                B) checkpoint dict，包含 "state_dict" 或 "model" 等字段
        classifier_key_prefix:
            - 分类模型 backbone 的 key 前缀（默认 conv_backbone）
        detector_key_prefix:
            - 检测模型 backbone 的 key 前缀（默认 backbone）
        map_location:
            - torch.load 的 map_location（默认 cpu）
        strict:
            - 是否严格匹配 detector 全部参数（默认 False，仅加载能匹配的部分）
        verbose:
            - 是否打印加载统计

    输出：
        report: dict
            - loaded_keys: 实际成功加载到 detector 的 key 列表
            - missing_keys: detector 期望但没加载到的 key
            - unexpected_keys: 加载 dict 中多余的 key
            - total_src_keys / total_dst_keys 等统计信息
    """
    # -------------------------
    # 1) 读取 checkpoint / state_dict
    # -------------------------
    ckpt = torch.load(ckpt_path, map_location=map_location)

    # 兼容多种保存格式
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            state = ckpt["state_dict"]
        elif "model" in ckpt and isinstance(ckpt["model"], dict):
            state = ckpt["model"]
        else:
            # 可能本身就是 state_dict
            state = ckpt
    else:
        raise TypeError(f"ckpt_path 加载结果不是 dict，实际类型={type(ckpt)}")

    # -------------------------
    # 2) 处理 DataParallel 前缀：module.
    # -------------------------
    def _strip_module_prefix(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if not sd:
            return sd
        # 只要存在一个 module. 就统一剥掉
        if any(k.startswith("module.") for k in sd.keys()):
            return {k[len("module."):]: v for k, v in sd.items()}
        return sd

    state = _strip_module_prefix(state)

    # -------------------------
    # 3) 抽取分类骨干参数，并改前缀映射到检测骨干
    # -------------------------
    src_prefix = classifier_key_prefix + "."
    dst_prefix = detector_key_prefix + "."

    mapped_state: Dict[str, torch.Tensor] = {}
    for k, v in state.items():
        if k.startswith(src_prefix):
            new_k = dst_prefix + k[len(src_prefix):]
            mapped_state[new_k] = v

    if len(mapped_state) == 0:
        # 说明没找到任何 conv_backbone.* 的参数
        # 这通常意味着：你保存的不是 YOLOv1_Classifier 整体 state_dict，
        # 而是只保存了 backbone 自身（key 可能是 "0.conv.weight" 这种）
        # 这里做一次兜底：尝试把“无前缀”的骨干 key 映射到 detector.backbone
        # 兜底策略：只加载能在 detector.state_dict() 中对上的 key
        det_sd = detector.state_dict()
        for k, v in state.items():
            # k 可能是 "0.conv.weight"，那么映射到 "backbone.0.conv.weight"
            cand = dst_prefix + k
            if cand in det_sd and det_sd[cand].shape == v.shape:
                mapped_state[cand] = v

    # -------------------------
    # 4) 按 shape 过滤（防止 silent mismatch）
    # -------------------------
    det_sd = detector.state_dict()
    filtered_state: Dict[str, torch.Tensor] = {}
    for k, v in mapped_state.items():
        if k in det_sd and det_sd[k].shape == v.shape:
            filtered_state[k] = v

    # -------------------------
    # 5) 加载
    # -------------------------
    load_ret = detector.load_state_dict(filtered_state, strict=strict)

    # PyTorch 的返回是 IncompatibleKeys(missing_keys, unexpected_keys)
    missing_keys = list(load_ret.missing_keys) if hasattr(load_ret, "missing_keys") else []
    unexpected_keys = list(load_ret.unexpected_keys) if hasattr(load_ret, "unexpected_keys") else []

    loaded_keys = sorted(list(filtered_state.keys()))

    report = {
        "total_src_keys": len(state),
        "total_mapped_keys": len(mapped_state),
        "total_loaded_keys": len(loaded_keys),
        "loaded_keys": loaded_keys,
        "missing_keys": missing_keys,
        "unexpected_keys": unexpected_keys,
        "classifier_key_prefix": classifier_key_prefix,
        "detector_key_prefix": detector_key_prefix,
        "ckpt_path": ckpt_path,
    }

    if verbose:
        print("[load_backbone_pretrained_to_detector] done")
        print(f"  ckpt_path           : {ckpt_path}")
        print(f"  src_total_keys      : {report['total_src_keys']}")
        print(f"  mapped_keys         : {report['total_mapped_keys']}")
        print(f"  loaded_keys         : {report['total_loaded_keys']}")
        # missing_keys 会包含 head_conv / fc 层等，这很正常（我们只加载 backbone）
        if len(missing_keys) > 0:
            print(f"  missing_keys (top10): {missing_keys[:10]}")
        if len(unexpected_keys) > 0:
            print(f"  unexpected_keys(top10): {unexpected_keys[:10]}")

    return report


# -------------------------
# 使用示例（你复制到工程里即可）
# -------------------------
if __name__ == "__main__":
    # 例：你训练分类骨干保存的权重路径
    backbone_ckpt = r"pre_weights\best_model.pth"

    # 检测模型
    det = YOLOv1(num_classes=20, B=2, ic_debug=False)

    # 加载骨干预训练
    rep = load_backbone_pretrained_to_detector(
        detector=det,
        ckpt_path=backbone_ckpt,
        classifier_key_prefix="conv_backbone",  # 对齐你的 YOLOv1_Classifier
        detector_key_prefix="backbone",         # 对齐你的 YOLOv1
        map_location="cpu",
        strict=False,
        verbose=True,
    )

    # rep 里可以看 loaded_keys / missing_keys 是否符合预期
    # 一般 missing_keys 会包含 head_conv、fc1、fc2 等（正常）
