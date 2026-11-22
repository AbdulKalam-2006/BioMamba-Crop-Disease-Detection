import io
import base64
import os
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
import cv2
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from model import BioMamba

app = FastAPI(title="BioMamba Mobile API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TREATMENT_ADVISORY = {
    "Bacterial Blight": "Spray Copper Oxychloride (0.2%) or Streptocycline. Avoid overhead irrigation.",
    "Corn___Common_Rust": "Apply Mancozeb or Azoxystrobin spray. Rotate crops with non-grasses.",
    "Corn___Gray_Leaf_Spot": "Use resistant hybrids. Apply Pyraclostrobin or Propiconazole at first sign.",
    "Corn___Healthy": "Crop is healthy. Continue regular field monitoring and balanced irrigation.",
    "Corn___Northern_Leaf_Blight": "Fungicide sprays containing Chlorothalonil; avoid overhead irrigation.",
    "Healthy": "Crop is healthy. Continue balanced crop management.",
    "Potato___Early_Blight": "Apply Mancozeb or Copper oxychloride. Remove and burn infected lower foliage.",
    "Potato___Healthy": "Crop is healthy. Maintain proper ridge spacing and soil drainage.",
    "Potato___Late_Blight": "Apply systemic fungicides (Metalaxyl + Mancozeb). Destroy severely infected tubers.",
    "Red Rot": "Use certified disease-free setts. Drench soil with Carbendazim. Practice crop rotation.",
    "Rice___Brown_Spot": "Treat seeds with Carbendazim. Maintain proper soil potash and zinc levels.",
    "Rice___Healthy": "Crop is healthy. Ensure systematic water management and balanced nitrogen.",
    "Rice___Leaf_Blast": "Spray Tricyclazole 75 WP. Avoid excessive nitrogenous fertilizers.",
    "Rice___Neck_Blast": "Apply Isoprothiolane or Tricyclazole before 5% flowering stage.",
    "Wheat___Brown_Rust": "Foliar spray of Propiconazole 25 EC. Use rust-resistant certified cultivars.",
    "Wheat___Healthy": "Crop is healthy. Maintain standard weed control and nitrogen scheduling.",
    "Wheat___Yellow_Rust": "Early spraying with Tebuconazole or Triadimefon upon first yellow stripe."
}

CROP_MAP = {
    "Potato": ["Potato___Early_Blight", "Potato___Healthy", "Potato___Late_Blight"],
    "Corn": ["Corn___Common_Rust", "Corn___Gray_Leaf_Spot", "Corn___Healthy", "Corn___Northern_Leaf_Blight"],
    "Rice": ["Rice___Brown_Spot", "Rice___Healthy", "Rice___Leaf_Blast", "Rice___Neck_Blast"],
    "Wheat": ["Wheat___Brown_Rust", "Wheat___Healthy", "Wheat___Yellow_Rust"],
    "Sugarcane": ["Bacterial Blight", "Healthy", "Red Rot"]
}

# Load Model Checkpoint
weights_file = "weights/biomamba_best.pth"
checkpoint = torch.load(weights_file, map_location="cpu")
class_names = [c for c in checkpoint['class_names'] if c != 'plots']
model = BioMamba(num_classes=len(class_names))
model.load_state_dict(checkpoint['model_state_dict'], strict=False)
model.eval()

def generate_vascular_vein_trace_b64(orig_img):
    img_rgb = np.array(orig_img.resize((400, 400)))
    g_channel = img_rgb[:, :, 1]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    vein_ridges = cv2.morphologyEx(g_channel, cv2.MORPH_BLACKHAT, kernel)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    enhanced_veins = clahe.apply(vein_ridges)
    _, binary_veins = cv2.threshold(enhanced_veins, 28, 255, cv2.THRESH_BINARY)
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    leaf_mask = cv2.inRange(hsv, (20, 25, 20), (95, 255, 255))
    clean_veins = cv2.bitwise_and(binary_veins, binary_veins, mask=leaf_mask)
    clean_veins = cv2.morphologyEx(clean_veins, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)))
    overlay = img_rgb.copy()
    overlay[clean_veins > 0] = [255, 20, 20]
    traced_bgr = cv2.cvtColor(cv2.addWeighted(img_rgb, 0.4, overlay, 0.6, 0), cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode('.jpg', traced_bgr)
    return base64.b64encode(buffer).decode('utf-8')

@app.post("/predict")
async def predict(file: UploadFile = File(...), crop: str = Form("Auto-Detect")):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    tfms = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    tensor_img = tfms(image).unsqueeze(0)

    with torch.no_grad():
        logits, _, _ = model(tensor_img)
        if crop != "Auto-Detect" and crop in CROP_MAP:
            allowed = CROP_MAP[crop]
            mask = torch.full_like(logits, float('-inf'))
            for i, name in enumerate(class_names):
                if name in allowed:
                    mask[0, i] = logits[0, i]
            probs = torch.softmax(mask, dim=1)[0]
        else:
            probs = torch.softmax(logits, dim=1)[0]

        top_prob, top_idx = torch.topk(probs, 1)

    conf = float(top_prob[0].item() * 100)
    pred_class = class_names[top_idx[0].item()]
    vein_trace_b64 = generate_vascular_vein_trace_b64(image)

    # Out of distribution check
    if conf < 70.0:
        return {
            "status": "unrecognized",
            "message": "Condition Not in Dataset / Physiological Stress",
            "confidence": conf,
            "treatment": "Soil nutrient testing suggested. Check for interveinal micronutrient deficiency.",
            "vein_trace_base64": vein_trace_b64
        }

    return {
        "status": "success",
        "predicted_condition": pred_class,
        "confidence": round(conf, 2),
        "treatment": TREATMENT_ADVISORY.get(pred_class, "Standard agronomic extension advisory."),
        "vein_trace_base64": vein_trace_b64
    }