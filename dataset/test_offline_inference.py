import json
import numpy as np
import onnxruntime as ort
from PIL import Image

# 1. Load exported metadata
with open("weights/model_metadata.json", "r") as f:
    meta = json.load(f)

class_names = meta["classes"]
print(f"Loaded {len(class_names)} classes.")

# 2. Start offline ONNX runtime session (Zero network connection needed)
session = ort.InferenceSession("weights/biomamba_model.onnx")

# 3. Dummy leaf tensor: [1, 3, 224, 224] normalized
dummy_leaf = np.random.randn(1, 3, 224, 224).astype(np.float32)

# 4. Run local inference
inputs = {session.get_inputs()[0].name: dummy_leaf}
outputs = session.run(None, inputs)

logits = outputs[0]
predicted_index = int(np.argmax(logits, axis=1)[0])
print(f"✅ Offline Inference Successful! Predicted: {class_names[predicted_index]}")