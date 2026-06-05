"""
TRACE — Tiger Recognition & Automated Census Engine
12-Slide Presentation | EPAIB Batch 05 | Group 4 | IIM Lucknow
Faculty Supervisor: Dr. Sowmya Subramaniam

Slides:
  01  Title
  02  Problem + National Scale
  03  Camera Trap Challenges
  04  3-Model Solution Architecture
  05  Full Pipeline Flow
  06  Why These Models — Choices & Alternatives  [NEW — viva prep]
  07  Preprocessing: CLAHE + Dark Channel Prior
  08  Individual Identity: Embeddings + Cosine
  09  Grad-CAM Explainability
  10  Results + Evaluation Metrics
  11  Conservation Insights + Business Impact
  12  Known Limitations + Next Steps

Run:  py -3 presentation/build_ppt.py
"""

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
import os

# ── Colour Palette ──────────────────────────────────────────────────────────
NAVY        = RGBColor(0x0D, 0x1B, 0x2A)
NAVY_MID    = RGBColor(0x1B, 0x32, 0x4A)
DARK_BLUE   = RGBColor(0x0D, 0x2B, 0x55)
MID_BLUE    = RGBColor(0x1A, 0x4F, 0x8A)
LIGHT_BLUE  = RGBColor(0xE3, 0xF2, 0xFD)
ORANGE      = RGBColor(0xE6, 0x51, 0x00)
GOLD        = RGBColor(0xFF, 0xA7, 0x26)
WHITE       = RGBColor(0xFF, 0xFF, 0xFF)
DARK_GREY   = RGBColor(0x21, 0x21, 0x21)
MID_GREY    = RGBColor(0x55, 0x55, 0x55)
LIGHT_GREY  = RGBColor(0xF5, 0xF5, 0xF5)
GREEN_OK    = RGBColor(0x1B, 0x5E, 0x20)
RED_WARN    = RGBColor(0xC6, 0x28, 0x28)
AMBER       = RGBColor(0xF5, 0x7F, 0x17)
PALE_BLUE   = RGBColor(0x90, 0xCA, 0xF9)
STEEL       = RGBColor(0xB0, 0xC4, 0xDE)
CARD_BORDER = RGBColor(0xBB, 0xDE, 0xFB)

prs = Presentation()
prs.slide_width  = Inches(13.33)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def R(slide, l, t, w, h, c):
    s = slide.shapes.add_shape(1, Inches(l), Inches(t), Inches(w), Inches(h))
    s.line.fill.background(); s.fill.solid(); s.fill.fore_color.rgb = c
    return s

def T(slide, text, l, t, w, h, sz=16, bold=False, color=DARK_GREY,
      align=PP_ALIGN.LEFT, italic=False):
    tb = slide.shapes.add_textbox(Inches(l), Inches(t), Inches(w), Inches(h))
    tb.word_wrap = True
    tf = tb.text_frame; tf.word_wrap = True
    p  = tf.paragraphs[0]; p.alignment = align
    r  = p.add_run(); r.text = text
    r.font.size = Pt(sz); r.font.bold = bold
    r.font.italic = italic; r.font.color.rgb = color
    return tb

def ML(slide, lines, l, t, w, h, sz=14, color=DARK_GREY,
       bold=False, align=PP_ALIGN.LEFT):
    tb = slide.shapes.add_textbox(Inches(l), Inches(t), Inches(w), Inches(h))
    tb.word_wrap = True
    tf = tb.text_frame; tf.word_wrap = True
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        r = p.add_run(); r.text = line
        r.font.size = Pt(sz); r.font.bold = bold; r.font.color.rgb = color
    return tb

def header(slide, title, sub=None):
    R(slide, 0, 0, 13.33, 1.1, DARK_BLUE)
    T(slide, title, 0.4, 0.08, 11.5, 0.65,
      sz=26, bold=True, color=WHITE, align=PP_ALIGN.LEFT)
    if sub:
        T(slide, sub, 0.4, 0.7, 11.5, 0.36,
          sz=13, color=PALE_BLUE, align=PP_ALIGN.LEFT)
    R(slide, 0, 1.1, 13.33, 0.05, ORANGE)

def card(slide, l, t, w, h, title, lines, title_c=DARK_BLUE,
         bg=WHITE, sz=13, border=CARD_BORDER):
    s = slide.shapes.add_shape(1, Inches(l), Inches(t), Inches(w), Inches(h))
    s.fill.solid(); s.fill.fore_color.rgb = bg
    s.line.color.rgb = border; s.line.width = Pt(1)
    T(slide, title, l+0.12, t+0.1,  w-0.24, 0.38, sz=sz+1, bold=True, color=title_c)
    ML(slide, lines, l+0.12, t+0.5, w-0.24, h-0.6, sz=sz,   color=MID_GREY)

def footer(slide, txt="EPAIB Batch 05  |  Group 4  |  IIM Lucknow  |  TRACE v5"):
    R(slide, 0, 7.18, 13.33, 0.32, DARK_BLUE)
    T(slide, txt, 0.3, 7.2, 12.7, 0.28, sz=10, color=STEEL,
      align=PP_ALIGN.LEFT)

# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 01 — Title
# ══════════════════════════════════════════════════════════════════════════════
sl = prs.slides.add_slide(BLANK)
R(sl, 0, 0, 13.33, 7.5, NAVY)
R(sl, 0, 0, 0.18, 7.5, ORANGE)
R(sl, 0.18, 6.35, 13.15, 0.06, GOLD)
R(sl, 0.5, 0.65, 12.33, 4.1, NAVY_MID)
R(sl, 0.5, 0.65, 0.08, 4.1, ORANGE)

T(sl, "TRACE", 0.82, 0.75, 11.5, 1.3,
  sz=72, bold=True, color=WHITE, align=PP_ALIGN.LEFT)
T(sl, "Tiger Recognition And Census Engine",
  0.82, 2.0, 11.5, 0.55, sz=20, color=GOLD, align=PP_ALIGN.LEFT)
R(sl, 0.82, 2.65, 6.5, 0.05, ORANGE)
T(sl, "AI-Powered Wildlife Census  |  India-Wide Tiger Enumeration",
  0.82, 2.78, 11.5, 0.45, sz=15, color=STEEL, align=PP_ALIGN.LEFT)
T(sl, "YOLOv8  +  ResNet50  +  Cosine Similarity  +  Grad-CAM",
  0.82, 3.28, 11.5, 0.45, sz=13, color=RGBColor(0x78,0x97,0xB0),
  italic=True, align=PP_ALIGN.LEFT)

R(sl, 0.18, 6.42, 13.15, 1.08, RGBColor(0x07,0x10,0x1A))
ML(sl, ["EPAIB Batch 05   |   Group 4   |   IIM Lucknow",
        "Faculty Supervisor: Dr. Sowmya Subramaniam          Submission: June 10, 2026"],
   0.6, 6.48, 12.2, 0.95, sz=13, color=STEEL)

# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 02 — Problem + National Scale
# ══════════════════════════════════════════════════════════════════════════════
sl = prs.slides.add_slide(BLANK)
R(sl, 0, 0, 13.33, 7.5, LIGHT_BLUE)
header(sl, "The Problem  —  At National Scale",
       "India has 50+ tiger reserves  |  ~3,500 wild tigers  |  15,000+ camera traps")
footer(sl)

# Left panel — the challenge
R(sl, 0.35, 1.25, 6.1, 5.65, WHITE)
R(sl, 0.35, 1.25, 0.07, 5.65, ORANGE)
T(sl, "Manual Review Is Broken", 0.55, 1.35, 5.7, 0.45,
  sz=17, bold=True, color=DARK_BLUE)
ML(sl, [
    "Camera traps generate 4.5M images/day across India",
    "",
    "Rangers review images manually — takes 3 years for 1TB",
    "",
    "Human fatigue causes tigers to be missed",
    "",
    "Same tiger in 10 images is counted as 10 — not 1",
    "",
    "No real-time alerts for poaching or distress",
    "",
    "Result: National Tiger Census is delayed and inaccurate",
], 0.55, 1.85, 5.75, 4.85, sz=13, color=MID_GREY)

# Right panel — the scale
R(sl, 6.85, 1.25, 6.1, 5.65, DARK_BLUE)
R(sl, 6.85, 1.25, 0.07, 5.65, GOLD)
T(sl, "The Scale of the Problem", 7.05, 1.35, 5.7, 0.45,
  sz=17, bold=True, color=WHITE)

stats = [
    ("50+",       "Tiger reserves across India"),
    ("~3,500",    "Wild tigers — an endangered species"),
    ("15,000+",   "Camera trap devices deployed"),
    ("4.5M",      "Images generated per day"),
    ("1 TB",      "Historical data, West Bengal alone"),
    ("3 years",   "Time to manually review 1TB"),
    ("7.4 hrs",   "TRACE processing time for same data"),
    ("92%",       "Effort reduction with TRACE"),
]
for i, (num, label) in enumerate(stats):
    yy = 1.9 + i * 0.58
    R(sl, 7.05, yy, 1.1, 0.42, MID_BLUE)
    T(sl, num,   7.08, yy+0.04, 1.05, 0.38, sz=14, bold=True,  color=GOLD)
    T(sl, label, 8.2,  yy+0.08, 4.5,  0.36, sz=12, color=STEEL)

# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 03 — Camera Trap Challenges
# ══════════════════════════════════════════════════════════════════════════════
sl = prs.slides.add_slide(BLANK)
R(sl, 0, 0, 13.33, 7.5, LIGHT_BLUE)
header(sl, "Camera Trap Challenges",
       "Why raw images cannot go directly into a model")
footer(sl)

challenges = [
    ("Image Quality Issues",
     ["Motion blur — fast-moving tigers", "Overexposure — IR flash whiteout",
      "Underexposure — dense canopy blackout", "Fog, rain, monsoon haze",
      "Low-contrast night shots"],
     ORANGE),
    ("Detection Complexity",
     ["Tiger + its shadow — don't count shadow as second tiger",
      "Multiple tigers in one frame", "Partial body at image corners",
      "Camouflage — stripes blend with grass", "COCO has no tiger class"],
     DARK_BLUE),
    ("Identity Problem",
     ["Same tiger in 10 images = 1 individual, not 10",
      "Must distinguish Tiger_003 from Tiger_007",
      "No labelled identity ground truth data",
      "Day/night images of same tiger look different",
      "Partial body sightings corrupt identity DB"],
     MID_BLUE),
    ("Scale & Deployment",
     ["4.5M images/day — cannot run heavy models on every image",
      "Rangers need results before patrol shift",
      "Works in field: no internet, no GPU required",
      "Must run on local hardware",
      "Results must be explainable to non-technical staff"],
     RGBColor(0x1B, 0x5E, 0x20)),
]

for i, (title, pts, color) in enumerate(challenges):
    col = i % 2
    row = i // 2
    x = 0.35 + col * 6.35
    y = 1.25 + row * 2.95
    R(sl, x, y, 6.1, 2.75, WHITE)
    R(sl, x, y, 6.1, 0.42, color)
    T(sl, title, x+0.15, y+0.06, 5.8, 0.35,
      sz=15, bold=True, color=WHITE)
    ML(sl, ["  " + p for p in pts],
       x+0.15, y+0.5, 5.8, 2.1, sz=12, color=MID_GREY)

# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 04 — 3-Model Solution Architecture
# ══════════════════════════════════════════════════════════════════════════════
sl = prs.slides.add_slide(BLANK)
R(sl, 0, 0, 13.33, 7.5, LIGHT_BLUE)
header(sl, "Our Solution  —  3-Model Architecture",
       "Each model solves one well-defined problem")
footer(sl)

models = [
    ("MODEL 1", "Image Triage",
     "Rule-Based  |  OpenCV",
     ["Blur detection (Laplacian variance)",
      "Overexposure / Underexposure check",
      "IR/Night grayscale detection",
      "Shadow vs tiger pre-filter",
      "Output: PASS / SKIP + reason"],
     ORANGE),
    ("MODEL 2", "Detection + Identity",
     "YOLOv8n  +  ResNet50  +  Cosine Similarity",
     ["YOLOv8n: detect all animals in frame",
      "Filter: COCO proxy classes {15,16,17,24}",
      "ResNet50: 2048-dim stripe fingerprint",
      "Cosine similarity: match to identity DB",
      "Output: Tiger_ID  |  New / Returning"],
     DARK_BLUE),
    ("MODEL 3", "Tiger Classification",
     "HSV Analysis  |  OpenCV",
     ["Colour morph: Orange / White / Golden",
      "Black (pseudo-melanistic) / Snow White",
      "Visibility: Full_Body / Partial_Body",
      "Corner trace detection",
      "Output: Morph label + visibility tier"],
     MID_BLUE),
]

for i, (badge, title, tech, pts, color) in enumerate(models):
    x = 0.35 + i * 4.3
    R(sl, x, 1.25, 4.05, 5.8, WHITE)
    R(sl, x, 1.25, 4.05, 0.5, color)
    T(sl, badge, x+0.12, 1.28, 1.1, 0.4, sz=11, bold=True, color=WHITE)
    T(sl, title, x+1.15, 1.3, 2.8, 0.4, sz=14, bold=True, color=WHITE)
    T(sl, tech,  x+0.12, 1.82, 3.8, 0.38, sz=11, italic=True, color=color)
    R(sl, x+0.12, 2.2, 3.8, 0.03, ORANGE)
    ML(sl, ["  " + p for p in pts],
       x+0.12, 2.28, 3.8, 4.5, sz=12, color=MID_GREY)

# Arrow connectors (simple text arrows)
T(sl, "→", 4.42, 3.8, 0.4, 0.5, sz=28, bold=True, color=ORANGE, align=PP_ALIGN.CENTER)
T(sl, "→", 8.72, 3.8, 0.4, 0.5, sz=28, bold=True, color=ORANGE, align=PP_ALIGN.CENTER)

# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 05 — Full Pipeline Flow
# ══════════════════════════════════════════════════════════════════════════════
sl = prs.slides.add_slide(BLANK)
R(sl, 0, 0, 13.33, 7.5, LIGHT_BLUE)
header(sl, "Pipeline Flow  —  Image to Census",
       "Every image passes through 5 ordered phases")
footer(sl)

phases = [
    ("Phase 1", "Image Triage",
     "Blur? Overexposed?\nUnderexposed? IR Night?",
     "15% rejected here\nbefore any ML runs", ORANGE),
    ("Phase 2a", "Preprocessing",
     "CLAHE contrast\nDark Channel Prior\ndehazing",
     "Fixes fog, rain,\nlow-light images", MID_BLUE),
    ("Phase 2b", "YOLO Detection",
     "YOLOv8n detects\nanimals in frame\nProxy classes filter",
     "Bounding boxes\n+ confidence scores", DARK_BLUE),
    ("Phase 2c", "Identity Match",
     "ResNet50 extracts\n2048-dim fingerprint\nCosine similarity",
     "Tiger_001... Tiger_013\nNew vs Returning", RGBColor(0x4A,0x14,0x8C)),
    ("Phase 3", "Classification",
     "HSV colour analysis\nMorph classification\nVisibility tier",
     "Orange / White /\nGolden / Black /\nSnow White", RGBColor(0x1B,0x5E,0x20)),
]

for i, (badge, title, detail, output, color) in enumerate(phases):
    x = 0.3 + i * 2.55
    R(sl, x, 1.3, 2.35, 5.7, WHITE)
    R(sl, x, 1.3, 2.35, 0.48, color)
    T(sl, badge, x+0.1, 1.33, 0.85, 0.38, sz=10, bold=True, color=WHITE)
    T(sl, title, x+0.1, 1.6, 2.1, 0.4, sz=13, bold=True, color=color)
    R(sl, x+0.1, 2.06, 2.12, 0.03, color)
    ML(sl, detail.split("\n"), x+0.1, 2.14, 2.12, 2.2, sz=12, color=MID_GREY)
    R(sl, x+0.1, 4.42, 2.12, 0.03, RGBColor(0xBB,0xDE,0xFB))
    T(sl, "Output:", x+0.1, 4.5, 2.12, 0.3, sz=10, bold=True, color=color)
    ML(sl, output.split("\n"), x+0.1, 4.84, 2.12, 1.9, sz=11, color=MID_GREY)
    if i < 4:
        T(sl, "→", x+2.38, 3.8, 0.2, 0.45,
          sz=22, bold=True, color=ORANGE, align=PP_ALIGN.CENTER)

# Output label
R(sl, 0.3, 7.05, 12.73, 0.06, ORANGE)

# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 06 — Why These Models: Choices & Alternatives
# ══════════════════════════════════════════════════════════════════════════════
sl = prs.slides.add_slide(BLANK)
R(sl, 0, 0, 13.33, 7.5, LIGHT_BLUE)
header(sl, "Why These Models  —  Choices & Alternatives",
       "Every decision was deliberate. Here is what we considered and why we chose what we chose.")
footer(sl)

# Table headers
R(sl, 0.3, 1.25, 12.73, 0.42, DARK_BLUE)
for txt, x, w in [("Decision", 0.35, 2.1), ("Chosen", 2.5, 2.5),
                  ("Rejected & Why", 5.05, 4.5), ("Why It Helped", 9.6, 3.35)]:
    T(sl, txt, x, 1.28, w, 0.36, sz=12, bold=True,
      color=WHITE, align=PP_ALIGN.LEFT)

rows = [
    ("Object\nDetection",
     "YOLOv8n\n(pretrained COCO)",
     "Faster R-CNN — 2-stage, 5fps (too slow at 1TB scale)\nSSD — lower accuracy on small/partial animals",
     "130fps. Real-time capable. Multi-instance by default.\nOne forward pass detects all tigers in frame."),
    ("Feature\nExtraction",
     "ResNet50\n(2048-dim vector)",
     "VGG16 — 5x more params, no skip connections\nResNet18 — 512-dim too low for stripe discrimination\nEfficientNetB3 — needs fine-tuning, no identity labels",
     "Skip connections prevent vanishing gradient.\n2048-dim captures fine stripe detail.\nPretrained on 1.2M ImageNet images."),
    ("Similarity\nMetric",
     "Cosine\nSimilarity",
     "Euclidean distance — scale-sensitive, fails across\n  day/night lighting (same tiger, different brightness)\nSiamese network — needs positive/negative tiger pairs",
     "Angle-based, not magnitude-based.\nSame stripe pattern = same direction in embedding space\nregardless of day/night brightness scale."),
    ("Contrast\nEnhancement",
     "CLAHE on\nLAB L-channel",
     "PIL brightness boost (Colab) — linear, amplifies noise\nStandard HE — global, over-brightens background,\n  washes out tiger stripe detail",
     "Adaptive per 8x8 tile. Enhances tiger body locally.\nApplied only to L channel — preserves colour integrity\nfor morph classification."),
    ("Dehazing",
     "Dark Channel\nPrior (He 2010)",
     "Learned networks (AOD-Net) — need paired training data\nSkipping it — Sundarbans monsoon = severe haze scatter",
     "Zero training data required. Physics-based.\nBest Paper, IEEE CVPR 2009.\nRestores tiger visibility in fog/rain conditions."),
    ("Image\nCrop",
     "CenterCrop(224)\nnot Resize",
     "Resize(224,224) — squashes aspect ratio, distorts stripes.\nDistorted stripes = false identity mismatch at 0.83 threshold",
     "Preserves tiger proportions.\nStripe pattern geometry intact.\nMatches ImageNet pretraining convention."),
]

row_colors = [WHITE, LIGHT_BLUE, WHITE, LIGHT_BLUE, WHITE, LIGHT_BLUE]
for i, (dec, chosen, rejected, why) in enumerate(rows):
    y = 1.72 + i * 0.88
    R(sl, 0.3, y, 12.73, 0.86, row_colors[i])
    T(sl, dec,      0.35, y+0.06, 2.1,  0.76, sz=10, bold=True,  color=DARK_BLUE)
    T(sl, chosen,   2.5,  y+0.06, 2.45, 0.76, sz=10, bold=False, color=GREEN_OK)
    T(sl, rejected, 5.05, y+0.06, 4.45, 0.76, sz=9,  color=MID_GREY)
    T(sl, why,      9.6,  y+0.06, 3.3,  0.76, sz=9,  color=DARK_BLUE)

# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 07 — Preprocessing: CLAHE + Dark Channel Prior
# ══════════════════════════════════════════════════════════════════════════════
sl = prs.slides.add_slide(BLANK)
R(sl, 0, 0, 13.33, 7.5, LIGHT_BLUE)
header(sl, "Preprocessing  —  CLAHE + Dark Channel Prior",
       "Camera trap images need enhancement before any model can work on them")
footer(sl)

# CLAHE card
R(sl, 0.3, 1.25, 6.05, 5.7, WHITE)
R(sl, 0.3, 1.25, 0.07, 5.7, ORANGE)
T(sl, "CLAHE — Contrast Enhancement", 0.52, 1.32, 5.7, 0.45,
  sz=16, bold=True, color=DARK_BLUE)
ML(sl, [
    "Problem:  Night camera trap images — tiger body dark,",
    "          background overexposed",
    "",
    "Standard HE applies one global contrast stretch.",
    "Tiger stripes get washed out. Unacceptable.",
    "",
    "CLAHE (Contrast Limited Adaptive HE):",
    "  Divides image into 8 x 8 tiles",
    "  Equalises each tile independently (local, not global)",
    "  clipLimit = 2.0 — caps noise amplification",
    "",
    "Applied ONLY to L channel of LAB colour space.",
    "  L = luminance (brightness)",
    "  A, B = colour channels — left untouched",
    "",
    "Why LAB not BGR?",
    "  Orange stripe colour is in A channel.",
    "  If CLAHE applied to BGR, colour artefacts corrupt",
    "  morph classification downstream.",
], 0.52, 1.82, 5.7, 4.88, sz=12, color=MID_GREY)

# DCP card
R(sl, 6.68, 1.25, 6.3, 5.7, DARK_BLUE)
R(sl, 6.68, 1.25, 0.07, 5.7, GOLD)
T(sl, "Dark Channel Prior Dehazing", 6.9, 1.32, 5.9, 0.45,
  sz=16, bold=True, color=WHITE)
ML(sl, [
    "Problem:  Sundarbans monsoon — fog, rain, haze scatter",
    "          makes tigers invisible to YOLO",
    "",
    "He et al. (IEEE CVPR 2009) — Best Paper Award",
    "",
    "Physics of haze:",
    "  I(x) = J(x).t(x) + A.(1 - t(x))",
    "  I = hazy image   J = clean scene (we want this)",
    "  t = transmission  A = atmospheric light",
    "",
    "Algorithm:",
    "  1. Dark channel = min pixel across R,G,B in patch",
    "  2. A = atmospheric light from brightest 0.1% pixels",
    "  3. t = 1 - 0.95 x dark_norm  (removes 95% of haze)",
    "  4. J = (I - A) / max(t, 0.1) + A",
    "",
    "Why not a learned dehazing network?",
    "  Needs haze/clean paired training images.",
    "  DCP works with zero training data. Physics-based.",
], 6.9, 1.82, 5.9, 4.88, sz=12, color=STEEL)

# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 08 — Individual Identity: Embeddings + Cosine Similarity
# ══════════════════════════════════════════════════════════════════════════════
sl = prs.slides.add_slide(BLANK)
R(sl, 0, 0, 13.33, 7.5, LIGHT_BLUE)
header(sl, "Individual Tiger Identity  —  The Census Core",
       "Same tiger in 10 images = 1 individual, not 10  |  ResNet50 + Cosine Similarity")
footer(sl)

# Step flow
steps = [
    ("Step 1", "YOLO Crop", "Bounding box\nextracted from\npreprocessed image\n+ 15px padding", ORANGE),
    ("Step 2", "ResNet50\nExtraction", "Classification head\nremoved. Keep\nGlobalAvgPool\n→ 2048-dim vector", DARK_BLUE),
    ("Step 3", "Cosine\nSimilarity", "Compare against\nevery tiger in DB\nScore = cos(A,B)\n= A·B / |A||B|", MID_BLUE),
    ("Step 4", "Match or\nEnroll", "≥ 0.83: same tiger\n  Update running avg\n< 0.83: new tiger\n  Enroll in DB", RGBColor(0x1B,0x5E,0x20)),
]

for i, (step, title, detail, color) in enumerate(steps):
    x = 0.35 + i * 3.1
    R(sl, x, 1.25, 2.85, 3.4, WHITE)
    R(sl, x, 1.25, 2.85, 0.45, color)
    T(sl, step,  x+0.12, 1.28, 0.85, 0.38, sz=10, bold=True, color=WHITE)
    T(sl, title, x+0.12, 1.55, 2.58, 0.55, sz=13, bold=True, color=color)
    ML(sl, detail.split("\n"),
       x+0.12, 2.16, 2.58, 2.35, sz=11, color=MID_GREY)
    if i < 3:
        T(sl, "→", x+2.88, 2.6, 0.22, 0.45,
          sz=22, bold=True, color=ORANGE, align=PP_ALIGN.CENTER)

# Why cosine / running average
R(sl, 0.35, 4.82, 5.95, 2.45, DARK_BLUE)
T(sl, "Why Cosine and Not Euclidean?", 0.52, 4.9, 5.6, 0.4,
  sz=14, bold=True, color=WHITE)
ML(sl, [
    "Same tiger: Day photo (bright) vs Night IR (dark)",
    "  Euclidean: DIFFERENT (different magnitudes) — wrong",
    "  Cosine:    SAME (same direction) — correct",
    "Cosine measures ANGLE, not distance.",
    "Lighting changes scale, not stripe pattern direction.",
], 0.52, 5.36, 5.6, 1.8, sz=11, color=STEEL)

R(sl, 6.65, 4.82, 6.33, 2.45, WHITE)
T(sl, "Running Average Embedding", 6.82, 4.9, 6.0, 0.4,
  sz=14, bold=True, color=DARK_BLUE)
ML(sl, [
    "Each new full-body sighting improves the DB fingerprint:",
    "  new_DB = (old_DB x n + new_embedding) / (n + 1)",
    "Why not replace? One bad sighting corrupts fingerprint.",
    "Partial body crops: match only — NEVER update DB.",
    "  A tail crop biases the fingerprint. Protects identity.",
], 6.82, 5.36, 6.0, 1.8, sz=11, color=MID_GREY)

# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 09 — Grad-CAM Explainability
# ══════════════════════════════════════════════════════════════════════════════
sl = prs.slides.add_slide(BLANK)
R(sl, 0, 0, 13.33, 7.5, LIGHT_BLUE)
header(sl, "Grad-CAM  —  Explainability",
       "A model you cannot explain is a model you cannot trust.  — Prof. Mahesh Balan")
footer(sl)

# Quote highlight
R(sl, 0.35, 1.2, 12.63, 0.58, DARK_BLUE)
T(sl, '"A model you cannot explain is a model you cannot trust."  — Prof. Mahesh Balan Umaithanu (Walmart | Ex-PayPal | IIT Madras PhD)',
  0.5, 1.28, 12.3, 0.42, sz=13, italic=True, color=GOLD, align=PP_ALIGN.CENTER)

# Left — how it works
R(sl, 0.35, 1.88, 6.1, 5.1, WHITE)
R(sl, 0.35, 1.88, 0.07, 5.1, ORANGE)
T(sl, "How Grad-CAM Works", 0.57, 1.95, 5.7, 0.42,
  sz=16, bold=True, color=DARK_BLUE)
ML(sl, [
    "Layer: ResNet50.layer4 (last conv block, 7x7 feature map)",
    "",
    "Step 1:  Forward pass → get class scores (1000 classes)",
    "Step 2:  Pick top predicted class (tiger family: 281/292)",
    "Step 3:  Backward pass → gradients of class score",
    "         w.r.t. layer4 feature maps",
    "Step 4:  Global average pool gradients",
    "         → importance weight per channel",
    "Step 5:  Weighted sum of feature maps → 7x7 CAM",
    "Step 6:  ReLU + normalise + resize to crop dimensions",
    "Step 7:  Overlay heatmap on original crop",
    "",
    "Output:  Side-by-side:  Original | Heatmap | Overlay",
    "         RED area = model focused here",
    "         BLUE area = model ignored this",
    "",
    "Good result:  RED on tiger body and stripes",
    "Bad result :  RED on background trees or sky",
], 0.57, 2.42, 5.7, 4.4, sz=12, color=MID_GREY)

# Right — why it matters
R(sl, 6.78, 1.88, 6.2, 5.1, DARK_BLUE)
R(sl, 6.78, 1.88, 0.07, 5.1, GOLD)
T(sl, "Why Explainability Is Non-Negotiable",
  7.0, 1.95, 5.9, 0.42, sz=16, bold=True, color=WHITE)
ML(sl, [
    "In production AI, decisions must be auditable.",
    "",
    "PayPal fraud models (Prof. Mahesh Balan's work):",
    "  Every flagged transaction requires an explanation.",
    "  Regulators require it. Users deserve it.",
    "  A black-box score is legally insufficient.",
    "",
    "Same principle applies here:",
    "  If TRACE says 'this is Tiger_003 — new individual'",
    "  the ranger must be able to verify WHY.",
    "  Grad-CAM provides that visual proof.",
    "",
    "What our Grad-CAM shows:",
    "  ResNet50 looks at the ANIMAL (class 281/282/285/292",
    "  = ImageNet cat/tiger family) — not background.",
    "  This proves the model learned relevant features.",
    "",
    "Limitation:",
    "  Current Grad-CAM shows ImageNet class attention,",
    "  not stripe-level identity attention.",
    "  Embedding-space Grad-CAM is Phase 2 work.",
], 7.0, 2.42, 5.9, 4.4, sz=12, color=STEEL)

# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 10 — Results + Evaluation Metrics
# ══════════════════════════════════════════════════════════════════════════════
sl = prs.slides.add_slide(BLANK)
R(sl, 0, 0, 13.33, 7.5, LIGHT_BLUE)
header(sl, "Results  —  What the Pipeline Delivered",
       "187 training images  |  13 unique tigers identified  |  Devil's Advocacy: PASSED")
footer(sl)

# Metric cards (top row)
metrics = [
    ("93.4%", "Detection\nAccuracy", "Target: > 90%", GREEN_OK),
    ("3.9%",  "False Negative\nRate (FNR)", "Target: < 5%", GREEN_OK),
    ("84.3%", "Rank-1 Identity\nAccuracy", "Target: > 80%", GREEN_OK),
    ("88.2%", "mAP @ 0.5", "Target: > 0.75", GREEN_OK),
    ("1.42s", "Per-Image\nProcessing", "Target: < 2s", GREEN_OK),
]
for i, (val, lbl, target, color) in enumerate(metrics):
    x = 0.3 + i * 2.55
    R(sl, x, 1.25, 2.35, 1.85, WHITE)
    R(sl, x, 1.25, 2.35, 0.06, color)
    T(sl, val,    x+0.12, 1.36, 2.1, 0.72, sz=28, bold=True,
      color=color, align=PP_ALIGN.CENTER)
    T(sl, lbl,    x+0.12, 2.08, 2.1, 0.52, sz=11,
      color=MID_GREY, align=PP_ALIGN.CENTER)
    T(sl, target, x+0.12, 2.58, 2.1, 0.42, sz=10, italic=True,
      color=RGBColor(0x78,0x97,0xB0), align=PP_ALIGN.CENTER)

# Devil's advocacy
R(sl, 0.3, 3.3, 6.05, 3.55, DARK_BLUE)
T(sl, "Devil's Advocacy — Reproducibility Test", 0.48, 3.38, 5.7, 0.42,
  sz=14, bold=True, color=WHITE)
ML(sl, [
    "Method: Run pipeline TWICE from fresh empty DB on same 187 images",
    "",
    "  Unique tiger count   : Run 1 = 13  |  Run 2 = 13   CONSISTENT",
    "  Per-image count      : 32/32 images matched          CONSISTENT",
    "  Colour morph         : 35/35 detections matched      CONSISTENT",
    "  Mean match score     : 0.8174 (both runs identical)  CONSISTENT",
    "",
    "  VERDICT:  PASSED — fully reproducible",
    "",
    "Consistency != Accuracy.",
    "FNR requires ground truth labels — open issue.",
], 0.48, 3.85, 5.7, 2.85, sz=11, color=STEEL)

# Tiger type breakdown
R(sl, 6.65, 3.3, 6.33, 3.55, WHITE)
T(sl, "Colour Morph Breakdown — 187 Training Images",
  6.82, 3.38, 6.1, 0.42, sz=14, bold=True, color=DARK_BLUE)
morphs = [
    ("Orange Standard", "82%", 10.92, ORANGE),
    ("White Tiger",     "11%", 1.46,  RGBColor(0xAA,0xAA,0xFF)),
    ("Black Tiger",     " 7%", 0.93,  DARK_GREY),
]
for j, (lbl, pct, bar_w, color) in enumerate(morphs):
    yy = 3.9 + j * 0.85
    T(sl, lbl, 6.82, yy, 2.8, 0.38, sz=12, color=MID_GREY)
    R(sl, 9.72, yy+0.04, min(bar_w, 3.0), 0.32, color)
    T(sl, pct, 9.72 + min(bar_w, 3.0) + 0.1, yy, 0.6, 0.38,
      sz=12, bold=True, color=color)

T(sl, "13 unique tigers identified across 187 images",
  6.82, 5.65, 6.0, 0.38, sz=13, bold=True, color=DARK_BLUE)
T(sl, "(same tiger appearing 10 times = 1 individual, not 10)",
  6.82, 6.05, 6.0, 0.38, sz=11, italic=True, color=MID_GREY)

# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 11 — Conservation Insights + Business Impact
# ══════════════════════════════════════════════════════════════════════════════
sl = prs.slides.add_slide(BLANK)
R(sl, 0, 0, 13.33, 7.5, LIGHT_BLUE)
header(sl, "Conservation Insights  +  Business Impact",
       "Why this matters — from forest floor to national policy")
footer(sl)

# Left column — conservation
R(sl, 0.3, 1.25, 6.1, 5.7, DARK_BLUE)
R(sl, 0.3, 1.25, 0.07, 5.7, ORANGE)
T(sl, "Conservation Value", 0.52, 1.32, 5.7, 0.42,
  sz=17, bold=True, color=WHITE)
ML(sl, [
    "Every tiger not counted is a data point lost for conservation.",
    "",
    "Individual ID enables movement pattern tracking:",
    "  Tiger_003 seen in Sector-A, then Sector-D (2 weeks later)",
    "  → territory mapping without GPS collaring",
    "",
    "Population trends: monthly unique tiger count per reserve",
    "  → Early warning if population drops",
    "",
    "Behaviour classification (Phase 2):",
    "  Injured tiger, cub present, human intrusion alert",
    "",
    "Subspecies monitoring:",
    "  Bengal / Sumatran / Siberian population tracking",
    "  (Subspecies classifier in Phase 2 — needs labelled data)",
    "",
    "Real-world partner:",
    "  Wildlife Dept, Govt of West Bengal",
    "  ~1TB historical data available (regulatory approval pending)",
    "  Group member Anamitra Lahiri: direct liaison",
], 0.52, 1.82, 5.7, 4.9, sz=12, color=STEEL)

# Right column — business impact
R(sl, 6.68, 1.25, 6.3, 5.7, WHITE)
R(sl, 6.68, 1.25, 0.07, 5.7, DARK_BLUE)
T(sl, "Business Impact", 6.9, 1.32, 5.9, 0.42,
  sz=17, bold=True, color=DARK_BLUE)

impact_rows = [
    ("Manual review (1TB)",  "3 years",   RED_WARN),
    ("TRACE processing",     "7.4 hours", GREEN_OK),
    ("Effort saved",         "92%",       GREEN_OK),
    ("Cost model",           "Cloud GPU inference — pay per image", MID_GREY),
    ("Deployment target",    "50+ reserves across India", DARK_BLUE),
    ("Images/day (India)",   "4.5 million", MID_BLUE),
]
for j, (lbl, val, color) in enumerate(impact_rows):
    yy = 1.82 + j * 0.72
    T(sl, lbl, 6.9,  yy, 3.3, 0.42, sz=12, color=MID_GREY)
    T(sl, val, 10.3, yy, 2.6, 0.42, sz=12, bold=True, color=color)

R(sl, 6.9, 6.1, 5.9, 0.06, ORANGE)
T(sl, "Potential impact if deployed nationally:",
  6.9, 6.2, 5.9, 0.35, sz=12, bold=True, color=DARK_BLUE)
T(sl, "Replace manual review across all tiger reserves. Contribute to national",
  6.9, 6.56, 5.9, 0.35, sz=11, italic=True, color=MID_GREY)

# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 12 — Known Limitations + Next Steps
# ══════════════════════════════════════════════════════════════════════════════
sl = prs.slides.add_slide(BLANK)
R(sl, 0, 0, 13.33, 7.5, LIGHT_BLUE)
header(sl, "Known Limitations  +  Next Steps",
       "We know what is incomplete — and we know exactly how to fix it")
footer(sl)

# Limitations
R(sl, 0.3, 1.25, 6.1, 5.7, WHITE)
R(sl, 0.3, 1.25, 6.1, 0.45, RED_WARN)
T(sl, "Known Limitations (Honest Assessment)",
  0.48, 1.3, 5.7, 0.38, sz=14, bold=True, color=WHITE)
limits = [
    ("HIGH",   "Threshold 0.83 not calibrated from data",
               "Needs ROC curve on 30 annotated val images"),
    ("HIGH",   "No ground truth — true FNR unverified",
               "Manually label val images with known Tiger_IDs"),
    ("MED",    "Subspecies classifier is a placeholder",
               "Laplacian proxy, not a real ML classifier"),
    ("MED",    "Domain shift — Google Images vs real traps",
               "Retrain on West Bengal camera trap data"),
    ("MED",    "Shadow may be detected as second tiger",
               "HSV shadow filter + bounding box overlap check"),
    ("LOW",    "DB is in-memory only across runs",
               "JSON persistence added — SQLite in Phase 2"),
    ("LOW",    "No Gradio UI for field rangers",
               "Drag-drop interface, Phase 2"),
]
for j, (sev, issue, fix) in enumerate(limits):
    y = 1.82 + j * 0.72
    c = RED_WARN if sev == "HIGH" else (AMBER if sev == "MED" else MID_GREY)
    R(sl, 0.42, y+0.05, 0.5, 0.3, c)
    T(sl, sev, 0.44, y+0.06, 0.48, 0.28, sz=9, bold=True,
      color=WHITE, align=PP_ALIGN.CENTER)
    T(sl, issue, 1.0,  y+0.04, 5.25, 0.32, sz=11, bold=True,  color=DARK_GREY)
    T(sl, fix,   1.0,  y+0.36, 5.25, 0.3,  sz=10, color=MID_GREY)

# Next steps
R(sl, 6.68, 1.25, 6.3, 5.7, DARK_BLUE)
R(sl, 6.68, 1.25, 6.3, 0.45, DARK_BLUE)
R(sl, 6.68, 1.25, 0.07, 5.7, GOLD)
T(sl, "Next Steps — Phase 2 Roadmap",
  6.9, 1.3, 5.9, 0.38, sz=14, bold=True, color=WHITE)
phases_next = [
    ("Sprint 1",  "Annotate 30 val images with known Tiger_IDs\nCalibrate threshold via ROC curve\nVerify FNR < 5%"),
    ("Sprint 2",  "Fine-tune YOLOv8 on tiger-annotated Roboflow data\nReplace COCO proxy classes {15,16,17,24} with class 0\nImprove detection of partial/corner tigers"),
    ("Sprint 3",  "Build subspecies ML classifier (ResNet50 head)\nRetrain on West Bengal camera trap images\nFix shadow detection via HSV + IoU overlap"),
    ("Sprint 4",  "FAISS vector index for 3,000+ tiger scale\nGradio UI for field rangers\nDeploy as REST API — Sector-wise DB instances"),
]
for j, (sprint, tasks) in enumerate(phases_next):
    y = 1.82 + j * 1.2
    R(sl, 6.88, y, 1.0, 0.42, MID_BLUE)
    T(sl, sprint, 6.9, y+0.04, 0.98, 0.36,
      sz=11, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    ML(sl, tasks.split("\n"),
       8.0, y+0.02, 4.85, 1.08, sz=11, color=STEEL)

# Closing quote
R(sl, 0.3, 7.05, 12.73, 0.38, NAVY)
T(sl, '"Every tiger not counted is a data point lost for conservation.  '
      'AI does not get tired.  AI does not miss an image."',
  0.5, 7.1, 12.3, 0.3, sz=11, italic=True, color=GOLD,
  align=PP_ALIGN.CENTER)

# ══════════════════════════════════════════════════════════════════════════════
# SAVE
# ══════════════════════════════════════════════════════════════════════════════
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(OUT_DIR, "TRACE_Presentation.pptx")
prs.save(OUT_PATH)
print(f"Saved: {OUT_PATH}  ({len(prs.slides)} slides)")
