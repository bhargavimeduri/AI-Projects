"""
generate_team_ppt.py
Creates the Group 4 Team Presentation PPT for Sundarban Tiger Enumeration.
Run: python presentation/generate_team_ppt.py
Output: presentation/Tiger_Enumeration_Group4.pptx
"""

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt
import copy
from pathlib import Path

# ── Colour palette ──────────────────────────────────────────────────────────
ORANGE      = RGBColor(0xE8, 0x6B, 0x00)   # tiger orange
DARK_BLUE   = RGBColor(0x1A, 0x29, 0x5A)   # IIM dark blue
WHITE       = RGBColor(0xFF, 0xFF, 0xFF)
LIGHT_GREY  = RGBColor(0xF4, 0xF4, 0xF4)
GREEN       = RGBColor(0x2E, 0x7D, 0x32)
YELLOW      = RGBColor(0xFF, 0xC1, 0x07)
MID_BLUE    = RGBColor(0x15, 0x65, 0xC0)
DARK_GREY   = RGBColor(0x33, 0x33, 0x33)
STEP1_COLOR = RGBColor(0x15, 0x65, 0xC0)   # blue  – detection
STEP2_COLOR = RGBColor(0xE8, 0x6B, 0x00)   # orange – ID + count
STEP3_COLOR = RGBColor(0x2E, 0x7D, 0x32)   # green  – metadata

prs = Presentation()
prs.slide_width  = Inches(13.33)
prs.slide_height = Inches(7.5)

BLANK = prs.slide_layouts[6]   # completely blank layout


# ── Helper functions ─────────────────────────────────────────────────────────

def add_rect(slide, l, t, w, h, fill=None, line=None, line_w=Pt(0)):
    shape = slide.shapes.add_shape(1, Inches(l), Inches(t), Inches(w), Inches(h))
    shape.line.width = line_w
    if fill:
        shape.fill.solid()
        shape.fill.fore_color.rgb = fill
    else:
        shape.fill.background()
    if line:
        shape.line.color.rgb = line
    else:
        shape.line.fill.background()
    return shape


def add_text(slide, text, l, t, w, h,
             size=18, bold=False, color=DARK_GREY,
             align=PP_ALIGN.LEFT, italic=False, wrap=True):
    txb = slide.shapes.add_textbox(Inches(l), Inches(t), Inches(w), Inches(h))
    txb.word_wrap = wrap
    tf = txb.text_frame
    tf.word_wrap = wrap
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = color
    return txb


def add_para(tf, text, size=14, bold=False, color=DARK_GREY,
             align=PP_ALIGN.LEFT, space_before=Pt(4)):
    p = tf.add_paragraph()
    p.alignment = align
    p.space_before = space_before
    run = p.add_run()
    run.text = text
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    return p


def slide_header(slide, title, subtitle=None,
                 bg=DARK_BLUE, title_color=WHITE, sub_color=YELLOW):
    add_rect(slide, 0, 0, 13.33, 1.4, fill=bg)
    add_text(slide, title, 0.4, 0.12, 12, 0.75,
             size=28, bold=True, color=title_color, align=PP_ALIGN.LEFT)
    if subtitle:
        add_text(slide, subtitle, 0.4, 0.82, 12, 0.45,
                 size=14, color=sub_color, align=PP_ALIGN.LEFT)


def footer(slide, text="EPAIB Batch 05 | Group 4 | IIM Lucknow | Sundarban Tiger Enumeration"):
    add_rect(slide, 0, 7.1, 13.33, 0.4, fill=DARK_BLUE)
    add_text(slide, text, 0.3, 7.13, 12.7, 0.3,
             size=9, color=WHITE, align=PP_ALIGN.CENTER)


def step_box(slide, l, t, w, h, number, label, color, bullets):
    """Coloured step box with number + bullets."""
    add_rect(slide, l, t, w, h, fill=color)
    add_text(slide, f"STEP {number}", l+0.1, t+0.08, w-0.2, 0.35,
             size=11, bold=True, color=WHITE, align=PP_ALIGN.LEFT)
    add_text(slide, label, l+0.1, t+0.38, w-0.2, 0.45,
             size=16, bold=True, color=WHITE, align=PP_ALIGN.LEFT)
    y = t + 0.88
    for b in bullets:
        add_text(slide, f"▸  {b}", l+0.15, y, w-0.25, 0.32,
                 size=11, color=WHITE, align=PP_ALIGN.LEFT)
        y += 0.33


def algo_table(slide, l, t, rows, col_widths, header_bg=DARK_BLUE):
    """Simple table rendered as coloured rectangles."""
    row_h = 0.42
    headers, *data = rows
    # header row
    x = l
    for i, (hdr, cw) in enumerate(zip(headers, col_widths)):
        add_rect(slide, x, t, cw, row_h, fill=header_bg)
        add_text(slide, hdr, x+0.08, t+0.07, cw-0.1, row_h-0.1,
                 size=11, bold=True, color=WHITE, align=PP_ALIGN.LEFT)
        x += cw
    # data rows
    for ri, row in enumerate(data):
        row_bg = LIGHT_GREY if ri % 2 == 0 else WHITE
        x = l
        for ci, (cell, cw) in enumerate(zip(row, col_widths)):
            add_rect(slide, x, t + row_h*(ri+1), cw, row_h,
                     fill=row_bg, line=RGBColor(0xCC,0xCC,0xCC), line_w=Pt(0.5))
            cell_color = STEP1_COLOR if ci==0 and ri==0 else DARK_GREY
            add_text(slide, cell, x+0.08,
                     t + row_h*(ri+1) + 0.06, cw-0.1, row_h-0.1,
                     size=10, color=DARK_GREY, align=PP_ALIGN.LEFT)
            x += cw


# ════════════════════════════════════════════════════════════════════════════
# SLIDE 1 — TITLE SLIDE
# ════════════════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(BLANK)

add_rect(slide, 0, 0, 13.33, 7.5, fill=DARK_BLUE)
add_rect(slide, 0, 0, 13.33, 0.12, fill=ORANGE)   # top stripe
add_rect(slide, 0, 7.38, 13.33, 0.12, fill=ORANGE) # bottom stripe

add_text(slide, "🐯", 5.8, 0.5, 2, 1.2, size=60, align=PP_ALIGN.CENTER, color=WHITE)

add_text(slide, "AI-Driven Tiger Enumeration",
         1, 1.7, 11.33, 1, size=36, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
add_text(slide, "Using Computer Vision & Advanced Machine Learning",
         1, 2.65, 11.33, 0.6, size=20, color=YELLOW, align=PP_ALIGN.CENTER)

add_rect(slide, 3.5, 3.4, 6.33, 0.04, fill=ORANGE)

add_text(slide, "Sundarbans Tiger Reserve  |  Government of West Bengal",
         1, 3.6, 11.33, 0.5, size=14, color=WHITE, align=PP_ALIGN.CENTER)
add_text(slide, "EPAIB Batch 05  |  Group 4  |  IIM Lucknow",
         1, 4.1, 11.33, 0.5, size=14, color=YELLOW, align=PP_ALIGN.CENTER)
add_text(slide, "Faculty Supervisor: Dr. Sowmya S",
         1, 4.6, 11.33, 0.4, size=13, color=WHITE, align=PP_ALIGN.CENTER)

# Team names
team = ["Bhargavi Meduri", "Anamitra Lahiri", "Member 3", "Member 4", "Member 5"]
x = 1.0
for name in team:
    add_rect(slide, x, 5.3, 2.1, 0.55, fill=RGBColor(0x2A,0x3F,0x7A))
    add_text(slide, name, x+0.08, 5.38, 1.95, 0.4,
             size=11, color=WHITE, align=PP_ALIGN.CENTER)
    x += 2.27


# ════════════════════════════════════════════════════════════════════════════
# SLIDE 2 — THE PROBLEM
# ════════════════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(BLANK)
add_rect(slide, 0, 0, 13.33, 7.5, fill=WHITE)
slide_header(slide, "The Problem", "Why manual tiger counting is broken")
footer(slide)

problems = [
    ("📸  Volume",        "Thousands of camera trap images every week across Sundarbans"),
    ("⏱  Time",          "Rangers take weeks to manually review one batch of images"),
    ("😴  Human fatigue", "Tired rangers miss tigers — data gaps in population count"),
    ("🔍  Individual ID", "Identifying tigers by stripe pattern requires expert knowledge"),
    ("🚨  No alerts",     "Poaching or injured tiger detection is delayed by days"),
    ("📊  No scale",      "Manual process cannot cover all 50+ tiger reserves in India"),
]

y = 1.6
for icon_label, desc in problems:
    add_rect(slide, 0.4, y, 12.5, 0.65, fill=LIGHT_GREY,
             line=RGBColor(0xDD,0xDD,0xDD), line_w=Pt(0.5))
    add_text(slide, icon_label, 0.55, y+0.1, 2.8, 0.48,
             size=13, bold=True, color=DARK_BLUE)
    add_text(slide, desc, 3.5, y+0.12, 9.2, 0.48, size=13, color=DARK_GREY)
    y += 0.73

add_rect(slide, 0.4, y+0.1, 12.5, 0.55, fill=ORANGE)
add_text(slide, "🤝  Real data partnership: Anamitra Lahiri → Programme Director, Wildlife Dept, Govt of West Bengal  |  ~1 TB images available",
         0.6, y+0.18, 12.2, 0.4, size=11, bold=True, color=WHITE)


# ════════════════════════════════════════════════════════════════════════════
# SLIDE 3 — OUR SOLUTION — 3-STEP PIPELINE
# ════════════════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(BLANK)
add_rect(slide, 0, 0, 13.33, 7.5, fill=WHITE)
slide_header(slide, "Our Solution — 3-Step AI Pipeline",
             "Every camera trap image flows through all three steps automatically")
footer(slide)

# Step boxes
step_box(slide, 0.3, 1.55, 3.9, 4.8, "1", "DETECTION",
         STEP1_COLOR,
         ["ResNet50 (Transfer Learning)",
          "Tiger present? Yes / No",
          "Binary classification",
          "Target: > 90% accuracy",
          "< 2 seconds per image"])

step_box(slide, 4.7, 1.55, 3.9, 4.8, "2", "IDENTIFICATION + COUNT",
         STEP2_COLOR,
         ["EfficientNetB3 — stripe analysis",
          "Which individual tiger is this?",
          "YOLOv8 — count tigers per frame",
          "ArcFace loss for ID accuracy",
          "Target: > 80% individual ID"])

step_box(slide, 9.1, 1.55, 3.9, 4.8, "3", "METADATA + HABITAT",
         STEP3_COLOR,
         ["Read GPS + timestamp from photo",
          "Map location → habitat zone",
          "Assign tiger name / ID",
          "Activity pattern analysis",
          "Territory + movement profile"])

# Arrows between boxes
for ax in [4.2, 8.6]:
    add_text(slide, "▶", ax, 3.7, 0.45, 0.6, size=22, color=DARK_GREY, align=PP_ALIGN.CENTER)

# Output banner
add_rect(slide, 0.3, 6.45, 12.73, 0.55, fill=DARK_BLUE)
add_text(slide,
         '📊  OUTPUT:  "T-17  |  2:14 AM  |  Camera B-12  |  Sector Sajnekhali  |  Habitat: mangrove-core  |  3rd sighting this week"',
         0.5, 6.52, 12.5, 0.4, size=12, bold=True, color=WHITE, align=PP_ALIGN.CENTER)


# ════════════════════════════════════════════════════════════════════════════
# SLIDE 4 — ADVANCED ALGORITHMS — STEP 1
# ════════════════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(BLANK)
add_rect(slide, 0, 0, 13.33, 7.5, fill=WHITE)
slide_header(slide, "Step 1 — Detection Algorithms",
             "Is a tiger present in this image?", bg=STEP1_COLOR, sub_color=WHITE)
footer(slide)

add_rect(slide, 0.3, 1.5, 12.7, 0.04, fill=STEP1_COLOR)

algo_table(slide, 0.3, 1.65,
    rows=[
        ["Algorithm",        "What it does",                          "Why relevant"],
        ["ResNet50 ★",       "50-layer CNN, ImageNet transfer",       "Our baseline — proven, fast"],
        ["EfficientNetV2",   "CNN optimised for accuracy/speed",      "Better than B3, recommended upgrade"],
        ["Vision Transformer\n(ViT)", "Attention across whole image", "Handles cluttered jungle background"],
        ["ConvNeXt",         "CNN + Transformer hybrid",              "State-of-art image classification"],
    ],
    col_widths=[3.0, 4.8, 4.9],
    header_bg=STEP1_COLOR
)

add_rect(slide, 0.3, 5.05, 12.7, 0.9, fill=RGBColor(0xE3,0xF2,0xFD))
add_text(slide, "💡  Transfer Learning Concept",
         0.5, 5.1, 4, 0.35, size=13, bold=True, color=STEP1_COLOR)
add_text(slide,
         "ResNet50 was trained on 1.2 million ImageNet photos. It already knows edges, textures, shapes.\n"
         "We freeze those layers and only train the final layer to say 'tiger' or 'no tiger'.\n"
         "This means we need far fewer tiger photos to get good accuracy.",
         0.5, 5.42, 12.3, 0.9, size=11, color=DARK_GREY)

add_rect(slide, 0.3, 6.1, 12.7, 0.55, fill=GREEN)
add_text(slide, "✅  Recommended: Start with ResNet50 → compare EfficientNetV2 → upgrade if accuracy < 90%",
         0.5, 6.18, 12.3, 0.38, size=12, bold=True, color=WHITE)


# ════════════════════════════════════════════════════════════════════════════
# SLIDE 5 — ADVANCED ALGORITHMS — STEP 2
# ════════════════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(BLANK)
add_rect(slide, 0, 0, 13.33, 7.5, fill=WHITE)
slide_header(slide, "Step 2 — Identification + Counting Algorithms",
             "Which tiger? How many tigers?", bg=STEP2_COLOR, sub_color=WHITE)
footer(slide)

# Individual ID section
add_text(slide, "2A — Individual Tiger ID (Stripe Pattern Recognition)",
         0.3, 1.55, 12.7, 0.45, size=14, bold=True, color=STEP2_COLOR)

algo_table(slide, 0.3, 2.05,
    rows=[
        ["Algorithm",           "What it does",                            "Why relevant"],
        ["Siamese Network",     "Compares 2 images — same tiger or not?",  "Works with few reference photos per tiger"],
        ["Triplet Loss ★",      "Pulls same-tiger embeddings together",     "Best for stripe-based biometric ID"],
        ["ArcFace / CosFace ★", "Advanced loss for biometric recognition",  "Used in wildlife ID research (leopards, whales)"],
        ["Contrastive Learning","Self-supervised — learns without labels",   "Useful when labeled data is scarce"],
    ],
    col_widths=[3.0, 5.0, 4.7],
    header_bg=STEP2_COLOR
)

# Counting section
add_text(slide, "2B — Tiger Counting (Object Detection)",
         0.3, 4.35, 12.7, 0.4, size=14, bold=True, color=STEP2_COLOR)

algo_table(slide, 0.3, 4.78,
    rows=[
        ["Algorithm",    "What it does",                     "Why relevant"],
        ["YOLOv8 ★",    "Bounding box per tiger, real-time", "Industry standard, our main tool"],
        ["Mask R-CNN",  "Detects + segments tiger body",     "Exact body outline — useful for health assessment"],
        ["ByteTrack",   "Tracks tiger across video frames",  "If video available — no double-counting"],
    ],
    col_widths=[2.5, 5.5, 4.7],
    header_bg=RGBColor(0xBF,0x36,0x0C)
)

add_rect(slide, 0.3, 6.55, 12.7, 0.55, fill=STEP2_COLOR)
add_text(slide, "✅  Recommended: Triplet Loss + ArcFace for ID  |  YOLOv8 for counting  |  ByteTrack if video frames available",
         0.5, 6.63, 12.3, 0.38, size=12, bold=True, color=WHITE)


# ════════════════════════════════════════════════════════════════════════════
# SLIDE 6 — ADVANCED ALGORITHMS — STEP 3
# ════════════════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(BLANK)
add_rect(slide, 0, 0, 13.33, 7.5, fill=WHITE)
slide_header(slide, "Step 3 — Metadata + Habitat Algorithms",
             "GPS + Timestamp → Tiger name, territory, movement pattern", bg=STEP3_COLOR, sub_color=WHITE)
footer(slide)

algo_table(slide, 0.3, 1.55,
    rows=[
        ["Algorithm",                  "What it does",                          "Tiger use case"],
        ["DBSCAN Clustering",          "Groups GPS points into zones",          "Find tiger territory automatically — no fixed zones needed"],
        ["Kernel Density Estimation",  "Heatmap of tiger density by GPS",       "Visual hotspot map for ranger deployment"],
        ["XGBoost / Random Forest",    "ML on metadata features",               "Predict: which tiger appears here at this time?"],
        ["LSTM (Time Series)",         "Sequential pattern in timestamps",       "Tiger daily rhythm — predict next appearance window"],
        ["Graph Neural Network (GNN)", "Models trap-to-trap relationships",      "Which cameras are in the same tiger's territory?"],
    ],
    col_widths=[3.3, 4.7, 5.0],
    header_bg=STEP3_COLOR
)

add_rect(slide, 0.3, 5.35, 12.7, 1.6, fill=RGBColor(0xE8,0xF5,0xE9))
add_text(slide, "📍  What metadata comes from the forest department:",
         0.5, 5.42, 12.3, 0.38, size=13, bold=True, color=STEP3_COLOR)

meta_items = [
    "GPS coordinates embedded in each photo (EXIF) OR separate CSV from forest dept",
    "Camera trap ID → maps to sector name (Sajnekhali, Dobanki, etc.)",
    "Timestamp of each image (date + time)",
    "Existing tiger registry: T-17, T-23 etc. with reference photos",
]
y = 5.82
for item in meta_items:
    add_text(slide, f"▸  {item}", 0.6, y, 12.2, 0.3, size=11, color=DARK_GREY)
    y += 0.3

add_rect(slide, 0.3, 6.62, 12.7, 0.48, fill=STEP3_COLOR)
add_text(slide, "✅  Recommended: DBSCAN + KDE for territory mapping  |  LSTM for activity prediction  |  XGBoost for patrol scheduling",
         0.5, 6.69, 12.3, 0.35, size=11, bold=True, color=WHITE)


# ════════════════════════════════════════════════════════════════════════════
# SLIDE 7 — EXPLAINABILITY (WHY THE MODEL DECIDED THIS)
# ════════════════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(BLANK)
add_rect(slide, 0, 0, 13.33, 7.5, fill=WHITE)
slide_header(slide, "Explainability — Why Did the Model Say This?",
             "Critical for government presentation and conservation trust")
footer(slide)

add_text(slide,
         "The Forest Department and IIM panel will ask: \"How do we know the model is right?\"  "
         "These 3 tools give visual proof.",
         0.4, 1.55, 12.5, 0.55, size=13, color=DARK_GREY)

explain_items = [
    ("Grad-CAM", STEP1_COLOR,
     "Gradient-weighted Class Activation Map",
     "Shows WHICH PART of the image made the model say 'tiger'.",
     "A heatmap overlay on the photo — red = model focused here.",
     "Proof: model looks at stripe pattern, NOT trees or background."),
    ("SHAP", STEP2_COLOR,
     "SHapley Additive exPlanations",
     "Shows WHICH FEATURES drove the prediction.",
     "For metadata: 'Time=2AM contributed +0.34 to tiger probability'",
     "Helps rangers understand why an alert was triggered."),
    ("LIME", STEP3_COLOR,
     "Local Interpretable Model-agnostic Explanations",
     "Per-image explanation: 'Why did model say T-17?'",
     "Highlights the exact stripe region that matched T-17's profile.",
     "Useful for edge cases and model debugging."),
]

y = 2.2
for name, color, full_name, line1, line2, line3 in explain_items:
    add_rect(slide, 0.3, y, 1.2, 1.6, fill=color)
    add_text(slide, name, 0.35, y+0.5, 1.1, 0.7,
             size=22, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    add_rect(slide, 1.6, y, 11.4, 1.6, fill=LIGHT_GREY,
             line=RGBColor(0xCC,0xCC,0xCC), line_w=Pt(0.5))
    add_text(slide, full_name, 1.75, y+0.05, 11.1, 0.35,
             size=12, bold=True, color=color)
    add_text(slide, line1, 1.75, y+0.38, 11.1, 0.32, size=11, color=DARK_GREY)
    add_text(slide, line2, 1.75, y+0.68, 11.1, 0.32, size=11, color=DARK_GREY)
    add_text(slide, line3, 1.75, y+0.98, 11.1, 0.35, size=11, color=DARK_GREY, italic=True)
    y += 1.78


# ════════════════════════════════════════════════════════════════════════════
# SLIDE 8 — FULL TECHNOLOGY STACK
# ════════════════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(BLANK)
add_rect(slide, 0, 0, 13.33, 7.5, fill=WHITE)
slide_header(slide, "Full Technology Stack")
footer(slide)

stack = [
    ("Step 1 — Detection",       STEP1_COLOR,  ["ResNet50 / EfficientNetV2", "TensorFlow / Keras", "Transfer Learning (ImageNet)"]),
    ("Step 2A — Individual ID",  STEP2_COLOR,  ["EfficientNetB3", "Triplet Loss + ArcFace", "512-dim Embedding Vectors"]),
    ("Step 2B — Counting",       STEP2_COLOR,  ["YOLOv8 (Ultralytics)", "ByteTrack (video)", "mAP@0.5 evaluation"]),
    ("Step 3 — Metadata",        STEP3_COLOR,  ["DBSCAN + KDE (scikit-learn)", "LSTM (time series)", "XGBoost (patrol scheduling)"]),
    ("Explainability",           DARK_BLUE,    ["Grad-CAM (TF GradientTape)", "SHAP", "LIME"]),
    ("Augmentation",             RGBColor(0x6A,0x1B,0x9A), ["Albumentations", "CLAHE contrast", "5× dataset multiplier"]),
    ("Experiment Tracking",      RGBColor(0x00,0x69,0x5C), ["MLflow", "Model checkpointing", "Hyperparameter logging"]),
    ("Language / Infrastructure",DARK_GREY,    ["Python 3.10+", "Jupyter Notebooks", "Git / GitHub"]),
]

cols = 4
box_w = 3.1
box_h = 1.35
x_start = 0.3
y_start = 1.6

for i, (label, color, items) in enumerate(stack):
    col = i % cols
    row = i // cols
    x = x_start + col * (box_w + 0.1)
    y = y_start + row * (box_h + 0.15)

    add_rect(slide, x, y, box_w, 0.38, fill=color)
    add_text(slide, label, x+0.1, y+0.05, box_w-0.15, 0.3,
             size=10, bold=True, color=WHITE)
    add_rect(slide, x, y+0.38, box_w, box_h-0.38,
             fill=LIGHT_GREY, line=RGBColor(0xCC,0xCC,0xCC), line_w=Pt(0.5))
    yi = y + 0.44
    for item in items:
        add_text(slide, f"• {item}", x+0.1, yi, box_w-0.15, 0.28, size=10, color=DARK_GREY)
        yi += 0.29


# ════════════════════════════════════════════════════════════════════════════
# SLIDE 9 — DATA REQUIREMENTS (for Anamitra)
# ════════════════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(BLANK)
add_rect(slide, 0, 0, 13.33, 7.5, fill=WHITE)
slide_header(slide, "Data Requirements",
             "What we need from Anamitra & the Forest Department")
footer(slide)

add_rect(slide, 0.3, 1.55, 6.1, 0.45, fill=GREEN)
add_text(slide, "✅  What we HAVE", 0.5, 1.6, 5.8, 0.35,
         size=13, bold=True, color=WHITE)

have = [
    "Official request letter from Dr. Sowmya S to Directorate of Forests, Kolkata",
    "Government partnership via Anamitra Lahiri (Wildlife Dept, Govt of West Bengal)",
    "~1 TB historical camera trap images (access pending)",
    "No data masking / anonymisation constraints",
    "Full project codebase — model, preprocessing, training scripts ready",
]
y = 2.1
for item in have:
    add_text(slide, f"▸  {item}", 0.5, y, 5.9, 0.35, size=11, color=DARK_GREY)
    y += 0.37

add_rect(slide, 6.9, 1.55, 6.1, 0.45, fill=ORANGE)
add_text(slide, "📋  What we NEED from Forest Dept", 7.1, 1.6, 5.8, 0.35,
         size=13, bold=True, color=WHITE)

need = [
    ("Images",          "50–100 photos per known tiger for ID model training"),
    ("GPS data",        "EXIF coordinates in photos OR separate CSV with trap locations"),
    ("Camera register", "Trap ID → GPS → sector name mapping file"),
    ("Tiger registry",  "Existing named tigers (T-17, T-23 etc.) with reference photos"),
    ("Timestamps",      "Date + time embedded in each image (usually automatic)"),
    ("Survey data",     "Any existing population count to validate our model output"),
]
y = 2.1
for label, desc in need:
    add_rect(slide, 6.9, y, 1.6, 0.33, fill=DARK_BLUE)
    add_text(slide, label, 6.95, y+0.04, 1.5, 0.28, size=10, bold=True, color=WHITE)
    add_text(slide, desc, 8.6, y+0.04, 4.3, 0.35, size=10, color=DARK_GREY)
    y += 0.41

add_rect(slide, 0.3, 6.45, 12.7, 0.65, fill=DARK_BLUE)
add_text(slide,
         "📌  Even a sample of 20–30 images is enough to start.  "
         "Share whatever format Anamitra receives — we will adapt the pipeline to it.",
         0.5, 6.52, 12.3, 0.48, size=12, bold=True, color=WHITE, align=PP_ALIGN.CENTER)


# ════════════════════════════════════════════════════════════════════════════
# SLIDE 10 — 20-DAY TIMELINE
# ════════════════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(BLANK)
add_rect(slide, 0, 0, 13.33, 7.5, fill=WHITE)
slide_header(slide, "20-Day Delivery Plan",
             "May 23 → June 10, 2026  |  Submission date: June 10")
footer(slide)

phases = [
    ("Phase 0\nMay 23–25",  DARK_BLUE,   "Setup & Prerequisites",
     ["Environment setup — Python, TensorFlow, YOLOv8",
      "Learn: CNN, Transfer Learning, Object Detection",
      "Configure MLflow experiment tracking"]),

    ("Phase 1\nMay 26–29",  STEP1_COLOR, "Data & Exploration  ⚠ Awaiting data",
     ["Notebook 01 — data inventory, class balance",
      "CLAHE preprocessing visualisation",
      "Augmentation pipeline — 5× dataset multiplier"]),

    ("Phase 2\nMay 30–Jun 3", STEP2_COLOR, "Detection Model",
     ["Train ResNet50 tiger/no-tiger classifier",
      "Grad-CAM heatmap — visual proof",
      "Target: >90% accuracy, FNR <5%"]),

    ("Phase 3\nJun 4–7",    STEP3_COLOR, "ID + Count + Metadata",
     ["EfficientNetB3 + ArcFace individual ID",
      "YOLOv8 tiger counter per frame",
      "GPS metadata → habitat zone mapping"]),

    ("Phase 4\nJun 8–10",   ORANGE,      "Evaluation & Submission",
     ["Notebook 04 — full evaluation + conservation insights",
      "Team presentation preparation",
      "GitHub repo final push + IIM submission"]),
]

box_w = 2.4
x = 0.3
for phase, color, title, tasks in phases:
    add_rect(slide, x, 1.55, box_w, 0.9, fill=color)
    add_text(slide, phase, x+0.1, 1.6, box_w-0.15, 0.82,
             size=11, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    add_rect(slide, x, 2.45, box_w, 0.55, fill=RGBColor(0xEE,0xEE,0xEE))
    add_text(slide, title, x+0.1, 2.5, box_w-0.15, 0.45,
             size=10, bold=True, color=color)
    y_task = 3.08
    for task in tasks:
        add_text(slide, f"▸ {task}", x+0.1, y_task, box_w-0.15, 0.4,
                 size=9, color=DARK_GREY)
        y_task += 0.42
    # Connector arrow (except last)
    if x < 12:
        add_text(slide, "▶", x+box_w+0.02, 1.85, 0.28, 0.5,
                 size=14, color=DARK_GREY, align=PP_ALIGN.CENTER)
    x += box_w + 0.3

add_rect(slide, 0.3, 5.55, 12.7, 0.6, fill=LIGHT_GREY,
         line=RGBColor(0xCC,0xCC,0xCC), line_w=Pt(0.5))
add_text(slide, "Success Metrics:", 0.5, 5.62, 2.5, 0.4, size=12, bold=True, color=DARK_BLUE)
metrics = "Detection accuracy > 90%   |   False Negative Rate < 5%   |   Individual ID accuracy > 80%   |   Processing < 2 sec/image"
add_text(slide, metrics, 3.0, 5.64, 10, 0.4, size=12, color=DARK_GREY)

add_rect(slide, 0.3, 6.28, 12.7, 0.6, fill=DARK_BLUE)
add_text(slide,
         "🚀  If this works in Sundarbans, it works across all 50+ tiger reserves in India — national scale impact",
         0.5, 6.36, 12.3, 0.42, size=13, bold=True, color=WHITE, align=PP_ALIGN.CENTER)


# ════════════════════════════════════════════════════════════════════════════
# SLIDE 11 — THANK YOU
# ════════════════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(BLANK)
add_rect(slide, 0, 0, 13.33, 7.5, fill=DARK_BLUE)
add_rect(slide, 0, 0, 13.33, 0.15, fill=ORANGE)
add_rect(slide, 0, 7.35, 13.33, 0.15, fill=ORANGE)

add_text(slide, "🐯", 6.0, 1.0, 1.5, 1.2, size=60, color=WHITE, align=PP_ALIGN.CENTER)
add_text(slide, "Built with purpose —",
         1, 2.3, 11.33, 0.7, size=30, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
add_text(slide, "for the tigers of Sundarbans and the future of AI-driven conservation.",
         1, 2.95, 11.33, 0.6, size=18, color=YELLOW, align=PP_ALIGN.CENTER)
add_rect(slide, 3.5, 3.7, 6.33, 0.05, fill=ORANGE)
add_text(slide, "If this works in Sundarbans, it works everywhere.",
         1, 3.85, 11.33, 0.5, size=16, italic=True, color=WHITE, align=PP_ALIGN.CENTER)

add_text(slide, "EPAIB Batch 05  |  Group 4  |  IIM Lucknow",
         1, 5.2, 11.33, 0.5, size=15, color=YELLOW, align=PP_ALIGN.CENTER)
add_text(slide, "Faculty: Dr. Sowmya S  |  sowmya@iiml.ac.in",
         1, 5.68, 11.33, 0.4, size=13, color=WHITE, align=PP_ALIGN.CENTER)
add_text(slide, "Data Partner: Directorate of Forests, Wildlife Wing, Govt of West Bengal",
         1, 6.08, 11.33, 0.4, size=12, color=WHITE, align=PP_ALIGN.CENTER)


# ── Save ─────────────────────────────────────────────────────────────────────
output = Path(__file__).parent / "Tiger_Enumeration_Group4.pptx"
prs.save(str(output))
print(f"\nSaved: {output}")
print(f"Slides: {len(prs.slides)}")
print(f"\n 1.  Title slide")
print(f" 2.  The Problem")
print(f" 3.  3-Step Pipeline")
print(f" 4.  Step 1 - Detection Algorithms")
print(f" 5.  Step 2 - ID + Counting Algorithms")
print(f" 6.  Step 3 - Metadata + Habitat Algorithms")
print(f" 7.  Explainability (Grad-CAM, SHAP, LIME)")
print(f" 8.  Full Technology Stack")
print(f" 9.  Data Requirements (for Anamitra)")
print(f"10.  20-Day Timeline")
print(f"11.  Thank You")
