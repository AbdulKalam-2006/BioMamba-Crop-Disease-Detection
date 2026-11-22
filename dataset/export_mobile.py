import os
import torch
from model import BioMamba

def export_for_mobile():
    weights_candidates = ["weights/biomamba_best.pth", "weights/checkpoint_latest.pth"]
    weights_path = next((f for f in weights_candidates if os.path.exists(f)), None)

    if weights_path is None:
        raise FileNotFoundError("No checkpoint found. Complete training first.")

    checkpoint = torch.load(weights_path, map_location="cpu")
    class_names = checkpoint['class_names']
    model = BioMamba(num_classes=len(class_names))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224)

    # 1. Export TorchScript for Native Android/iOS (PyTorch Mobile Lite)
    traced_model = torch.jit.trace(model, dummy_input)
    torchscript_path = os.path.join("weights", "biomamba_mobile.pt")
    traced_model.save(torchscript_path)
    print(f"✅ Exported TorchScript Mobile model: {torchscript_path}")

    # 2. Export ONNX for Flutter / React Native
    onnx_path = os.path.join("weights", "biomamba.onnx")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=14,
        input_names=['input_leaf'],
        output_names=['logits', 'ssm_out', 'feat_map']
    )
    print(f"✅ Exported ONNX model: {onnx_path}")

if __name__ == "__main__":
    export_for_mobile()