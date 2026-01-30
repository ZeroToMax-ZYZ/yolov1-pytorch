from nets.yolov1 import YOLOv1

def build_model(cfg):
    model_name = cfg["model_name"]

    if model_name == "YOLOv1":
        model = YOLOv1(ic_debug=False)

    else:
        raise ValueError(f"❗Unsupported model name: {model_name}")
    
    return model