"""
╔══════════════════════════════════════════════════════════════════╗
║   TRACE — Grad-CAM Visual Explainability Check                 ║
║                                                                  ║
║   Runs Grad-CAM on the first 10 tiger crops detected by YOLO.  ║
║   Generates heatmap overlays showing WHERE ResNet50 is looking. ║
║                                                                  ║
║   Good result: heatmap glows on tiger body / stripes            ║
║   Bad result : heatmap glows on background / trees / sky        ║
║                                                                  ║
║   Does NOT change tiger_pipeline.py in any way.                 ║
╚══════════════════════════════════════════════════════════════════╝
"""

import cv2
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from ultralytics import YOLO
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent.parent
TARGET_DIR  = BASE_DIR / "data" / "sample" / "tigers" / "train"
GRADCAM_DIR = BASE_DIR / "reports" / "gradcam_outputs"
GRADCAM_DIR.mkdir(exist_ok=True)

DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"
SUPPORTED_EXTS   = {".jpg", ".jpeg", ".png", ".webp"}
TIGER_PROXY_CLASSES = {15, 16, 17, 24}
MAX_CROPS        = 10   # generate Grad-CAM for first 10 tiger crops found

print(f"[INFO] Device: {DEVICE.upper()}")
print(f"[INFO] Grad-CAM outputs will be saved to: {GRADCAM_DIR}")

# ══════════════════════════════════════════════════════════════════════════════
# LOAD MODELS
# ══════════════════════════════════════════════════════════════════════════════

print("[INFO] Loading YOLOv8...")
detector = YOLO("yolov8n.pt")

print("[INFO] Loading ResNet50 with hooks for Grad-CAM...")
resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT).to(DEVICE)
resnet.eval()

# ── Standard ImageNet transform ────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ══════════════════════════════════════════════════════════════════════════════
# GRAD-CAM IMPLEMENTATION
# Uses the last convolutional layer of ResNet50 (layer4).
# Grad-CAM = gradient of the top predicted class score
#            with respect to the feature maps of the last conv layer.
# High gradient × high activation = important region for the prediction.
# ══════════════════════════════════════════════════════════════════════════════

# Storage for forward and backward hooks
feature_maps = {}
gradients    = {}

def save_feature_maps(module, input, output):
    """Forward hook — saves the feature map output of layer4."""
    feature_maps["layer4"] = output

def save_gradients(module, grad_input, grad_output):
    """Backward hook — saves the gradient flowing into layer4."""
    gradients["layer4"] = grad_output[0]

# Register hooks on layer4 (last conv block of ResNet50)
resnet.layer4.register_forward_hook(save_feature_maps)
resnet.layer4.register_full_backward_hook(save_gradients)

def generate_gradcam(crop_bgr, save_path, label=""):
    """
    Generate Grad-CAM heatmap for a tiger crop.

    Steps:
    1. Forward pass → get class scores
    2. Pick the top predicted class
    3. Backward pass → compute gradients of that class w.r.t. layer4 feature maps
    4. Global average pool the gradients → importance weights per channel
    5. Weighted sum of feature maps → raw CAM
    6. ReLU + normalise + resize to original crop size
    7. Overlay heatmap on original image
    """
    # Convert crop to RGB PIL
    rgb     = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    tensor  = transform(pil_img).unsqueeze(0).to(DEVICE)
    tensor.requires_grad_(True)

    # Step 1 & 2: Forward pass
    resnet.zero_grad()
    output = resnet(tensor)                          # shape: (1, 1000)
    top_class = output.argmax(dim=1).item()          # highest scoring class

    # Step 3: Backward pass for top class
    score = output[0, top_class]
    score.backward()

    # Step 4: Global average pool gradients → channel weights
    grads   = gradients["layer4"]                    # shape: (1, 2048, 7, 7)
    fmaps   = feature_maps["layer4"]                 # shape: (1, 2048, 7, 7)
    weights = grads.mean(dim=[2, 3], keepdim=True)   # shape: (1, 2048, 1, 1)

    # Step 5: Weighted sum of feature maps
    cam = (weights * fmaps).sum(dim=1).squeeze()     # shape: (7, 7)

    # Step 6: ReLU (keep only positive influence) + normalise
    cam = torch.relu(cam).detach().cpu().numpy()
    if cam.max() > 0:
        cam = cam / cam.max()

    # Resize CAM to match original crop size
    h, w  = crop_bgr.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))

    # Step 7: Convert to heatmap and overlay on original crop
    heatmap    = cv2.applyColorMap(
        (cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET
    )
    overlay    = cv2.addWeighted(crop_bgr, 0.5, heatmap, 0.5, 0)

    # Add label text
    if label:
        cv2.putText(overlay, label, (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.putText(overlay, f"Class: {top_class}", (5, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    # Save side-by-side: original | heatmap | overlay
    orig_resized = cv2.resize(crop_bgr, (w, h))
    heat_resized = cv2.resize(heatmap,  (w, h))

    # Pad all to same height if needed
    max_h = max(orig_resized.shape[0], heat_resized.shape[0], overlay.shape[0])
    def pad(img, h):
        if img.shape[0] < h:
            img = cv2.copyMakeBorder(img, 0, h-img.shape[0], 0, 0,
                                     cv2.BORDER_CONSTANT, value=0)
        return img

    combined = np.hstack([
        pad(orig_resized, max_h),
        pad(heat_resized, max_h),
        pad(overlay, max_h)
    ])

    # Add column headers
    header = np.zeros((30, combined.shape[1], 3), dtype=np.uint8)
    col_w  = w
    cv2.putText(header, "Original",  (col_w//4,    20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    cv2.putText(header, "Grad-CAM",  (col_w + col_w//4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    cv2.putText(header, "Overlay",   (2*col_w + col_w//4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    final = np.vstack([header, combined])
    cv2.imwrite(str(save_path), final)
    return top_class

# ══════════════════════════════════════════════════════════════════════════════
# MAIN — Find tiger crops and generate Grad-CAM
# ══════════════════════════════════════════════════════════════════════════════

all_images = [p for p in TARGET_DIR.rglob("*")
              if p.suffix.lower() in SUPPORTED_EXTS]

print(f"\n[INFO] Scanning {len(all_images)} images for tiger crops...")
print(f"[INFO] Will generate Grad-CAM for first {MAX_CROPS} crops found.\n")

crop_count = 0

for img_path in all_images:
    if crop_count >= MAX_CROPS:
        break

    img = cv2.imread(str(img_path))
    if img is None:
        try:
            pil = Image.open(img_path).convert("RGB")
            img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        except:
            continue

    boxes = detector(str(img_path), verbose=False)[0].boxes
    if len(boxes) == 0:
        continue

    img_area = img.shape[0] * img.shape[1]

    for box in boxes:
        if crop_count >= MAX_CROPS:
            break

        conf     = float(box.conf[0])
        cls      = int(box.cls[0])
        x1,y1,x2,y2 = map(int, box.xyxy[0])

        if cls not in TIGER_PROXY_CLASSES: continue
        if conf < 0.35: continue
        if (x2-x1)*(y2-y1)/img_area < 0.03: continue

        crop = img[y1:y2, x1:x2]
        if crop.size == 0: continue

        # Ensure crop is large enough for ResNet
        if crop.shape[0] < 64 or crop.shape[1] < 64:
            crop = cv2.resize(crop, (224, 224))

        crop_count += 1
        save_path  = GRADCAM_DIR / f"gradcam_{crop_count:02d}_{img_path.stem}.jpg"
        label      = f"Crop {crop_count} | {img_path.name} | conf={conf:.2f}"

        top_class  = generate_gradcam(crop, save_path, label)

        print(f"  [{crop_count:02d}] {img_path.name} | conf={conf:.2f} | "
              f"YOLO class={cls} | ResNet top class={top_class} | saved: {save_path.name}")

# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n{'='*60}")
print(f"  Grad-CAM complete — {crop_count} heatmaps generated")
print(f"  Location: {GRADCAM_DIR}")
print(f"{'='*60}")
print("""
HOW TO READ YOUR GRAD-CAM OUTPUT
Each image has 3 panels:
  Left   : Original tiger crop
  Middle : Grad-CAM heatmap (RED = model focused here, BLUE = ignored)
  Right  : Overlay (heatmap on original image)

GOOD RESULT: Red/yellow area covers tiger body and stripes
BAD RESULT : Red/yellow area on background, trees, or corners
""")

# Open the output folder
import subprocess, sys
subprocess.Popen(f'explorer "{GRADCAM_DIR}"')
