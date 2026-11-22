import os
import json
import torch
from model import BioMamba

def export_offline():
    weights_path = os.path.join("weights", "biomamba_best.pth")
    if not os.path.exists(weights_path):
        weights_path = os.path.join("weights", "checkpoint_latest.pth")

    device = torch.device("cpu")
    checkpoint = torch.load(weights_path, map_location=device)
    class_names = checkpoint['class_names']
    if 'plots' in class_names:
        class_names = [c for c in class_names if c != 'plots']

    print(f"📦 Loading weights from {weights_path} with {len(class_names)} classes...")
    model = BioMamba(num_classes=len(class_names))
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()

    # Input: [Batch=1, Channels=3, Height=224, Width=224]
    dummy_input = torch.randn(1, 3, 224, 224)

    os.makedirs("weights", exist_ok=True)

    # 1. Export Universal ONNX Model (Primary for Android, Flutter, React Native)
    onnx_path = os.path.join("weights", "biomamba_model.onnx")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['logits', 'ssm_out', 'feat_map']
    )
    print(f"✅ ONNX model successfully saved: {onnx_path}")

    # 2. Export Standard TorchScript & Lite Interpreter (Without XNNPACK dependency)
    try:
        traced_script = torch.jit.trace(model, dummy_input)
        
        # Standard TorchScript (.pt)
        pt_path = os.path.join("weights", "biomamba_traced.pt")
        traced_script.save(pt_path)
        print(f"✅ Standard TorchScript model saved: {pt_path}")

        # PyTorch Lite (.ptl) saved directly without optimize_for_mobile
        ptl_path = os.path.join("weights", "biomamba_model.ptl")
        traced_script._save_for_lite_interpreter(ptl_path)
        print(f"✅ PyTorch Lite model saved: {ptl_path}")
    except Exception as e:
        print(f"ℹ️ TorchScript notice: {e}")

    # 3. Export class labels and metadata to JSON
    metadata = {
        "classes": class_names,
        "input_shape": [1, 3, 224, 224],
        "normalization": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225]
        }
    }
    meta_path = os.path.join("weights", "model_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"✅ Model metadata saved: {meta_path}")

if __name__ == "__main__":
    export_offline()