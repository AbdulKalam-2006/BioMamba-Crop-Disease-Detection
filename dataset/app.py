import os
import math
import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
import streamlit as st
from model import BioMamba

st.set_page_config(page_title="BioMamba Crop Disease Diagnostic", layout="wide")

# ─── CLEAN WHITE BACKGROUND & BLACK FONT ─────────────────────────────────────
st.markdown("""
<style>
    .stApp, [data-testid="stAppViewContainer"] {
        background-color: #FFFFFF !important;
        color: #000000 !important;
    }
    [data-testid="stSidebar"], [data-testid="stSidebarContent"] {
        background-color: #F8F9FA !important;
        color: #000000 !important;
    }
    h1, h2, h3, h4, h5, h6, p, span, label, div {
        color: #000000 !important;
    }
    [data-testid="stMetricValue"], [data-testid="stMetricLabel"] {
        color: #000000 !important;
    }
    .stRadio label, .stFileUploader label {
        color: #000000 !important;
        font-weight: 600 !important;
    }
    .stCaption, [data-testid="stCaptionContainer"] p {
        color: #4A5568 !important;
    }
    [data-testid="stExpander"] {
        background-color: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
    }
</style>
""", unsafe_allow_html=True)

TREATMENT_ADVISORY = {
    "Healthy": "No pathogen detected. The leaf exhibits normal physiological vigor. Maintain standard irrigation and nutrient scheduling.",
    "Bacterial Blight": "Spray Copper Oxychloride (0.2%) or Streptocycline. Avoid overhead sprinkler irrigation to limit bacterial spread.",
    "Common Rust": "Apply Mancozeb (2g/L) or Azoxystrobin spray. Practice crop rotation with non-host varieties.",
    "Gray Leaf Spot": "Apply Pyraclostrobin or Propiconazole at first sign of lesions. Improve canopy airflow and manage crop residue.",
    "Northern Leaf Blight": "Apply foliar sprays containing Chlorothalonil or Mancozeb. Destroy infected debris post-harvest.",
    "Early Blight": "Spray Mancozeb or Copper Oxychloride. Prune lower affected leaves to stop soil splash-up.",
    "Late Blight": "Apply systemic fungicides (Metalaxyl + Mancozeb). Destroy severely blighted tissue immediately.",
    "Red Rot": "Use certified disease-free planting material. Drench root zone with Carbendazim. Practice crop rotation.",
    "Brown Spot": "Treat seeds with Carbendazim before sowing. Supplement soil with zinc sulfate and balanced potash.",
    "Leaf Blast": "Foliar application of Tricyclazole 75 WP. Avoid excess nitrogenous fertilizer during humid weather.",
    "Neck Blast": "Apply Isoprothiolane or Tricyclazole just prior to 5% panicle emergence stage.",
    "Brown Rust": "Spray Propiconazole 25 EC. Utilize rust-resistant certified seed stock.",
    "Yellow Rust": "Spray Tebuconazole or Triadimefon immediately upon emergence of parallel yellow pustule stripes."
}

TRAINED_CROPS = ["Corn", "Potato", "Rice", "Sugarcane", "Wheat"]

WORK_SIDE = 480          # longest side used for segmentation / vein tracing
LOW_CONF_THRESHOLD = 60.0  # below this, the result is shown as "uncertain"


# ─────────────────────────────────────────────────────────────────────────────
# LEAF ISOLATION  (shape/center prior + GrabCut, NOT green-colour only)
# ─────────────────────────────────────────────────────────────────────────────
def _pick_component(binary, seed_xy):
    """Return boolean mask of the connected component that contains the seed point,
    or (if the seed is on background) the large component closest to the centre."""
    n, labels, stats, cents = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n <= 1:
        return None
    h, w = binary.shape
    sx, sy = seed_xy
    lbl = labels[sy, sx]
    if lbl == 0:
        diag = math.hypot(w, h)
        best, best_s = None, -1.0
        for i in range(1, n):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < 0.01 * h * w:
                continue
            d = math.hypot(cents[i][0] - sx, cents[i][1] - sy) / diag
            s = area * max(0.2, 1.0 - 1.5 * d)
            if s > best_s:
                best, best_s = i, s
        if best is None:
            return None
        lbl = best
    return labels == lbl


def _fill_holes(mask_u8):
    cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(mask_u8)
    if cnts:
        cv2.drawContours(out, [max(cnts, key=cv2.contourArea)], -1, 255, thickness=cv2.FILLED)
    return out


def _grabcut_leaf(img_rgb):
    """Segment the main (centred, foreground) leaf blade with GrabCut. Works for pale,
    yellow, grey or reddish diseased leaves because it models colour statistics of the
    actual frame instead of requiring 'green'."""
    h, w = img_rgb.shape[:2]
    scale = min(1.0, WORK_SIDE / float(max(h, w)))
    small = cv2.resize(img_rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA) if scale < 1 else img_rgb.copy()
    hs, ws = small.shape[:2]

    gc = np.full((hs, ws), cv2.GC_PR_BGD, np.uint8)
    b = max(2, int(0.03 * min(hs, ws)))
    gc[:b, :] = cv2.GC_BGD
    gc[-b:, :] = cv2.GC_BGD
    gc[:, :b] = cv2.GC_BGD
    gc[:, -b:] = cv2.GC_BGD
    cv2.ellipse(gc, (ws // 2, hs // 2), (int(ws * 0.30), int(hs * 0.40)), 0, 0, 360, cv2.GC_PR_FGD, -1)
    cv2.ellipse(gc, (ws // 2, hs // 2), (max(3, int(ws * 0.07)), int(hs * 0.18)), 0, 0, 360, cv2.GC_FGD, -1)

    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(cv2.cvtColor(small, cv2.COLOR_RGB2BGR), gc, None, bgd, fgd, 4, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return None

    fg = ((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)).astype(np.uint8) * 255
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))

    comp = _pick_component(fg, (ws // 2, hs // 2))
    if comp is None:
        return None
    mask_small = _fill_holes(comp.astype(np.uint8) * 255)

    frac = np.count_nonzero(mask_small) / float(hs * ws)
    if frac < 0.02 or frac > 0.90:
        return None
    return cv2.resize(mask_small, (w, h), interpolation=cv2.INTER_NEAREST)


def _color_leaf_fallback(img_rgb):
    """Fallback if GrabCut fails: broad vegetation colour mask -> central component."""
    h, w = img_rgb.shape[:2]
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    m = cv2.inRange(hsv, np.array([15, 20, 25]), np.array([95, 255, 255]))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    comp = _pick_component(m, (w // 2, h // 2))
    if comp is None:
        return None
    return _fill_holes(comp.astype(np.uint8) * 255)


def _validate_leaf_region(img_rgb, mask):
    """Checks that the isolated region really looks like plant tissue.
    Returns (ok, reason, metrics)."""
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    m = mask > 0
    n = float(np.count_nonzero(m))
    if n < 50:
        return False, "no leaf region found", {}
    H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    R = img_rgb[:, :, 0].astype(np.int32); G = img_rgb[:, :, 1].astype(np.int32); B = img_rgb[:, :, 2].astype(np.int32)

    plant = m & (H >= 10) & (H <= 95) & (S >= 28) & (V >= 45)
    reddish_leaf = m & ((H <= 12) | (H >= 160)) & (S >= 60) & (V >= 50) & (R > G + 15)   # red/brown lesions
    dark_neutral = m & (V < 95) & (S < 70)           # black / grey / metal / tyre
    bright_neutral = m & (V > 190) & (S < 25)        # white sky, walls, paper
    blue = m & (H >= 95) & (H <= 135) & (S >= 40)

    pct = lambda a: np.count_nonzero(a) / n * 100.0
    met = dict(plant=pct(plant) + pct(reddish_leaf), dark=pct(dark_neutral),
               bright=pct(bright_neutral), blue=pct(blue))

    if met["dark"] > 18.0:
        return False, "dark non-plant object inside the detected region", met
    if met["blue"] > 6.0 or met["bright"] > 35.0:
        return False, "sky / non-plant background inside the detected region", met
    if met["plant"] < 45.0:
        return False, "detected region does not have plant colours/texture", met
    return True, "ok", met


def extract_primary_visible_leaf(image_pil):
    """
    Returns (is_leaf, foliage_ratio, primary_mask, leaf_aspect_ratio, bbox, reason).
    The image itself is NEVER modified - the mask is only used for the vein overlay,
    the optional crop and the optional colour prior.
    """
    img_rgb = np.array(image_pil)
    h_img, w_img = img_rgb.shape[:2]
    total_pixels = float(h_img * w_img)
    empty = np.zeros((h_img, w_img), dtype=np.uint8)

    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    R = img_rgb[:, :, 0].astype(np.int32)
    G = img_rgb[:, :, 1].astype(np.int32)
    B = img_rgb[:, :, 2].astype(np.int32)

    # 1. Face gate (webcam selfies)
    has_face = False
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    if os.path.exists(cascade_path):
        faces = cv2.CascadeClassifier(cascade_path).detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=4, minSize=(45, 45))
        has_face = len(faces) > 0

    # 2. Outdoor-scene / vehicle gate (landscapes, cars, buildings are not leaf photos)
    top = hsv[:max(1, int(h_img * 0.40)), :, :]
    blue = cv2.inRange(top, np.array([95, 40, 50]), np.array([135, 255, 255]))
    sky_ratio = np.count_nonzero(blue) / (top.shape[0] * top.shape[1] + 1e-6) * 100.0
    ctr = hsv[int(h_img * 0.2):int(h_img * 0.8), int(w_img * 0.2):int(w_img * 0.8), :]
    veh = cv2.inRange(ctr, np.array([0, 0, 10]), np.array([180, 60, 100]))
    veh_ratio = np.count_nonzero(veh) / (ctr.shape[0] * ctr.shape[1] + 1e-6) * 100.0
    if sky_ratio > 20.0:
        return False, 0.0, empty, 1.0, None, "outdoor scene / sky visible"
    if veh_ratio > 25.0:
        return False, 0.0, empty, 1.0, None, "large dark non-plant object in frame"

    # 3. Is there ANY vegetation in the frame? (loose gate: weeds / healthy parts count)
    if has_face:
        veg = cv2.inRange(hsv, np.array([35, 38, 35]), np.array([88, 255, 255]))
        veg = ((veg > 0) & (G > (R + 3)) & (G > (B + 5))).astype(np.uint8)
        min_ratio = 1.2
    else:
        veg = cv2.inRange(hsv, np.array([20, 20, 20]), np.array([94, 255, 255]))
        veg = ((veg > 0) & ((2 * G - R - B) > 3)).astype(np.uint8)
        min_ratio = 2.5   # was 7.0 - pale/diseased leaves have little green
    foliage_ratio = np.count_nonzero(veg) / total_pixels * 100.0
    if foliage_ratio < min_ratio:
        return False, foliage_ratio, empty, 1.0, None, "too little vegetation in frame"

    # 4. Isolate the main leaf blade (shape + centre prior, not colour only)
    primary_mask = _grabcut_leaf(img_rgb)
    if primary_mask is None:
        primary_mask = _color_leaf_fallback(img_rgb)
    if primary_mask is None or np.count_nonzero(primary_mask) < 0.01 * total_pixels:
        return False, foliage_ratio, empty, 1.0, None, "no distinct leaf found"

    ok, reason, _ = _validate_leaf_region(img_rgb, primary_mask)
    if not ok:
        return False, foliage_ratio, empty, 1.0, None, reason

    cnts, _ = cv2.findContours(primary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    main_cnt = max(cnts, key=cv2.contourArea)
    (rw, rh) = cv2.minAreaRect(main_cnt)[1]
    leaf_ar = max(rw, rh) / (min(rw, rh) + 1e-6) if rw > 0 and rh > 0 else 1.0
    bbox = cv2.boundingRect(main_cnt)  # x, y, w, h
    return True, foliage_ratio, primary_mask, leaf_ar, bbox, "ok"


def crop_to_leaf(image_pil, bbox, margin=0.12):
    """Natural-pixel crop around the leaf (no masking / blurring / recolouring)."""
    x, y, w, h = bbox
    W, H = image_pil.size
    mx, my = int(w * margin), int(h * margin)
    x0, y0 = max(0, x - mx), max(0, y - my)
    x1, y1 = min(W, x + w + mx), min(H, y + h + my)
    return image_pil.crop((x0, y0, x1, y1))


# ─────────────────────────────────────────────────────────────────────────────
# OPTIONAL, GENTLE COLOUR PRIOR  (OFF by default)
# ─────────────────────────────────────────────────────────────────────────────
def gentle_color_prior(probs_tensor, class_names, orig_img, primary_mask, max_conf=0.65, boost=1.6):
    """
    Only used when the toggle is on AND the network is unsure (top-1 < max_conf).
    It multiplies a matching class by a small factor and renormalises. It never
    forces a class to 87-90 % and never suppresses other classes.
    """
    probs = probs_tensor.clone()
    if float(probs.max().item()) >= max_conf:
        return probs

    img = np.array(orig_img.resize((224, 224)))
    m = cv2.resize(primary_mask, (224, 224), interpolation=cv2.INTER_NEAREST) > 0
    n = float(np.count_nonzero(m))
    if n < 50:
        return probs

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    R = img[:, :, 0].astype(np.int32); G = img[:, :, 1].astype(np.int32); B = img[:, :, 2].astype(np.int32)
    H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    crimson = m & ((H <= 14) | (H >= 166)) & (S >= 75) & (V >= 50) & (R > G + 24) & (R > B + 35)
    ygreen = m & (H >= 22) & (H <= 55) & (S >= 40) & (V >= 65)
    gray_tan = m & (H >= 12) & (H <= 36) & (S >= 22) & (S <= 82) & (V >= 110)
    green = m & (H >= 34) & (H <= 88) & (S >= 42) & (V >= 45)
    dbrown = m & (H >= 6) & (H <= 25) & (S >= 40) & (V >= 20) & (V <= 112) & (R > B + 12)

    pct = lambda a: np.count_nonzero(a) / n * 100.0
    crimson_p, yg_p, gray_p, green_p, brown_p = map(pct, (crimson, ygreen, gray_tan, green, dbrown))

    target = None
    if 0.3 <= crimson_p <= 22.0 and yg_p >= 18.0:
        target = "Red Rot"
    elif 3.0 <= gray_p <= 45.0 and green_p >= 35.0 and crimson_p < 0.3:
        target = "Gray Leaf Spot"
    elif 2.5 <= brown_p <= 35.0 and green_p >= 25.0 and crimson_p < 0.3 and gray_p < 12.0:
        target = "Brown Spot"

    if target:
        for i, cname in enumerate(class_names):
            if clean_disease_name(cname) == target:
                probs[i] = probs[i] * boost
        probs = probs / (probs.sum() + 1e-12)
    return probs


def clean_disease_name(raw_name):
    name = raw_name
    for prefix in ["sugarcane___", "Sugarcane___", "Potato___", "Corn___", "Rice___", "Wheat___"]:
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    return name.replace("___", " ").replace("_", " ").strip()


def detect_untrained_leaf(probs_tensor, conf):
    """
    Out-of-distribution screening based ONLY on the model's own uncertainty
    (normalised Shannon entropy / very low top-1 confidence).
    NOTE: the old 'aspect ratio >= 2.9' rule is removed - corn, sugarcane and wheat
    leaves are naturally long and narrow, so it flagged valid trained crops.
    """
    p = probs_tensor.cpu().numpy()
    entropy = -np.sum(p * np.log(p + 1e-12))
    norm_entropy = entropy / (math.log(len(p)) + 1e-12)
    return bool(norm_entropy > 0.72 or conf < 30.0)


@st.cache_resource
def load_model(weights_path):
    checkpoint = torch.load(weights_path, map_location="cpu")
    class_names = [c for c in checkpoint['class_names'] if c not in ['plots', 'plots_review']]
    model = BioMamba(num_classes=len(class_names))
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    return model, class_names


# ─────────────────────────────────────────────────────────────────────────────
# VEIN TRACE  (aspect-preserving, restricted to the isolated leaf, oriented)
# ─────────────────────────────────────────────────────────────────────────────
def generate_focused_vein_trace(orig_img, primary_mask, show_outline=True):
    img = np.array(orig_img)
    h, w = img.shape[:2]
    s = min(1.0, WORK_SIDE / float(max(h, w)))
    size = (max(1, int(w * s)), max(1, int(h * s)))
    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    m = cv2.resize(primary_mask, size, interpolation=cv2.INTER_NEAREST)

    inner = cv2.erode(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    if np.count_nonzero(inner) < 100:
        inner = m
    ys, xs = np.nonzero(inner)
    if xs.size < 50:
        return img

    g_eq = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(img[:, :, 1])
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    ridge = cv2.max(cv2.morphologyEx(g_eq, cv2.MORPH_BLACKHAT, k),
                    cv2.morphologyEx(g_eq, cv2.MORPH_TOPHAT, k))
    ridge = cv2.bitwise_and(ridge, ridge, mask=inner)
    vals = ridge[inner > 0]
    thr = max(12.0, float(np.percentile(vals, 70))) if vals.size else 12.0
    binary = ((ridge > thr).astype(np.uint8)) * 255

    # Principal axis of the leaf -> veins run (roughly) along it
    pts = np.column_stack([xs, ys]).astype(np.float64)
    pts -= pts.mean(axis=0)
    evals, evecs = np.linalg.eigh(np.cov(pts.T))
    v = evecs[:, int(np.argmax(evals))]
    axis_deg = math.degrees(math.atan2(v[1], v[0]))

    L = 19
    c = L // 2
    keep = np.zeros_like(binary)
    for da in range(-30, 31, 10):
        a = math.radians(axis_deg + da)
        dx, dy = int(round(math.cos(a) * c)), int(round(math.sin(a) * c))
        kern = np.zeros((L, L), np.uint8)
        cv2.line(kern, (c - dx, c - dy), (c + dx, c + dy), 1, 1)
        keep = cv2.bitwise_or(keep, cv2.morphologyEx(binary, cv2.MORPH_OPEN, kern))

    n, lab, stats_, _ = cv2.connectedComponentsWithStats(keep, connectivity=8)
    clean = np.zeros_like(keep)
    for i in range(1, n):
        if stats_[i, cv2.CC_STAT_AREA] >= 25 and max(stats_[i, cv2.CC_STAT_WIDTH], stats_[i, cv2.CC_STAT_HEIGHT]) >= 14:
            clean[lab == i] = 255
    clean = cv2.bitwise_and(clean, inner)

    overlay = img.copy()
    overlay[clean > 0] = [255, 20, 20]
    out = cv2.addWeighted(img, 0.35, overlay, 0.65, 0)
    if show_outline:
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, (30, 120, 255), 2)
    return out


# ─── HEADER ───────────────────────────────────────────────────────────────────
st.title("🌾 BioMamba: Bio-Inspired Crop Disease Diagnostic System")
st.markdown("**Selective State-Space Modeling with Vascular Scanning & Bio-Diffusion**")
st.caption("🛡️ *Active Guardrail: Primary-leaf isolation & OOD screening enabled*")

weights_candidates = [
    os.path.join("weights", "biomamba_best.pth"),
    os.path.join("dataset", "weights", "biomamba_best.pth"),
    os.path.join("..", "weights", "biomamba_best.pth"),
    r"C:\Users\Abdul Kalam\OneDrive\Desktop\Abdul\Bio Mamba\Crop Diseases\dataset\weights\biomamba_best.pth",
    r"C:\Users\Abdul Kalam\OneDrive\Desktop\Abdul\Bio Mamba\Crop Diseases\weights\biomamba_best.pth"
]
weights_file = next((f for f in weights_candidates if os.path.exists(f)), None)

if weights_file is not None:
    model, class_names = load_model(weights_file)
    st.sidebar.success(f"✅ Model Loaded: {os.path.basename(weights_file)}")
    st.sidebar.info("🌿 **Trained Crops:**\n" + "\n".join([f"- {c}" for c in TRAINED_CROPS]))
else:
    st.sidebar.error("⚠️ Model checkpoint not found in 'weights/' directory.")
    st.stop()

st.sidebar.markdown("### ⚙️ Settings")
model_input_mode = st.sidebar.radio(
    "Model input",
    ["Full original frame (recommended)", "Tight leaf crop (natural pixels)"],
)
use_color_prior = st.sidebar.checkbox("Experimental colour prior (only when model is unsure)", value=False)
show_outline = st.sidebar.checkbox("Show isolated-leaf outline (blue)", value=True)

input_mode = st.radio("Select Input Source:", ["📁 Upload Image File", "📸 Use Phone/PC Camera"], horizontal=True)

uploaded_file = None
if input_mode == "📁 Upload Image File":
    uploaded_file = st.file_uploader("Upload Leaf Image (JPEG/PNG)", type=["jpg", "jpeg", "png"])
else:
    uploaded_file = st.camera_input("Capture Leaf Directly via Camera")

if uploaded_file is not None:
    full_image = Image.open(uploaded_file).convert("RGB")

    # 1. Isolate the primary leaf (used for gating, vein trace, optional crop)
    is_leaf, foliage_score, primary_mask, leaf_ar, bbox, reject_reason = extract_primary_visible_leaf(full_image)

    if not is_leaf:
        st.error("🚫 **Diagnostic Blocked: No Plant Leaf Detected**")
        st.warning("This image does not look like a clear crop leaf (" + reject_reason + "). "
                   "Non-plant objects, vehicles, faces, landscapes and clothing are blocked.")
        st.info("👉 **Please upload a clear leaf image, or capture one:** fill the frame with a single leaf, "
                "keep it in focus, in daylight, against a plain background.")
        st.image(full_image, caption="Rejected Frame (Non-Leaf Content)", width=350)
        st.stop()

    # 2. Inference on UNALTERED pixels (matches the training distribution)
    if model_input_mode.startswith("Tight") and bbox is not None:
        model_input = crop_to_leaf(full_image, bbox)
    else:
        model_input = full_image

    col1, col2, col3 = st.columns([1, 1, 1])
    col1.subheader("1. Original Leaf")
    col1.image(full_image, width="stretch", caption="Captured Leaf Frame")

    tfms = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    tensor_img = tfms(model_input).unsqueeze(0)

    with torch.no_grad():
        logits, _, _ = model(tensor_img)
        probs = torch.softmax(logits, dim=1)[0]
        if use_color_prior:
            probs = gentle_color_prior(probs, class_names, full_image, primary_mask)
        top_probs, top_indices = torch.topk(probs, min(3, len(class_names)))

    pred_raw_class = class_names[top_indices[0].item()]
    conf = float(top_probs[0].item() * 100)
    pure_disease_name = clean_disease_name(pred_raw_class)

    # 3. Vein trace on the isolated leaf only
    vein_img = generate_focused_vein_trace(full_image, primary_mask, show_outline=show_outline)
    col2.subheader("2. Vascular Vein Trace (Red)")
    col2.image(vein_img, caption="Focused Vein Scan (Primary Visible Leaf Only)", width="stretch")

    # 4. Decision
    col3.subheader("3. Diagnostic Decision")
    is_ood = detect_untrained_leaf(probs, conf)
    is_uncertain = conf < LOW_CONF_THRESHOLD

    col3.metric("Detected Condition", f"{pure_disease_name}" + (" (uncertain)" if is_uncertain else ""))
    col3.progress(min(int(conf), 100))
    col3.write(f"**Confidence:** {conf:.2f}%")

    if is_ood:
        col3.warning(
            "⚠️ **Possibly an untrained leaf / unclear image:** the model is uncertain across many classes. "
            "Trained crops are Corn, Potato, Rice, Sugarcane, Wheat."
        )
    elif is_uncertain:
        col3.warning(
            "⚠️ **Low confidence.** Retake the photo: fill the frame with one leaf, keep it in focus, "
            "use daylight without glare, and avoid busy backgrounds. Check the Top-3 list below."
        )

    col3.subheader("Clinical Treatment Protocol")
    if is_uncertain or is_ood:
        col3.info("Result is not reliable enough for a treatment recommendation. "
                  "Confirm with an agricultural extension officer, or retake the image.")
    else:
        col3.info(TREATMENT_ADVISORY.get(
            pure_disease_name,
            "Consult your local agricultural extension service for tailored field remediation."))

    with col3.expander("📊 View Top Probabilities", expanded=is_uncertain):
        for rank in range(min(3, len(class_names))):
            idx = top_indices[rank].item()
            st.write(f"**{rank+1}. {clean_disease_name(class_names[idx])}**: {top_probs[rank].item()*100:.2f}%")