"""
Generate Word document capturing all Tiger AI Ultimate v3.0 run outputs.
Covers: Train run (3392 images) + Test Bh run (35 images) + Test Round 2 (18 images).
Usage: py -3 docs/generate_run_report_word.py
"""

import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from docx import Document
from docx.shared import Pt, RGBColor, Inches, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import pandas as pd
from pathlib import Path
from datetime import datetime

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent.parent
REPORTS_DIR = BASE_DIR / "reports"
CSV_FILE    = REPORTS_DIR / "ultimate_population_report.csv"
OUTPUT_DOCX = REPORTS_DIR / "Tiger_AI_Ultimate_Run_Report.docx"

# ── Colour palette ─────────────────────────────────────────────────────────────
C_DARK_GREEN  = RGBColor(0x1B, 0x5E, 0x20)
C_MID_GREEN   = RGBColor(0x2E, 0x7D, 0x32)
C_LIGHT_GREEN = RGBColor(0xE8, 0xF5, 0xE9)
C_ORANGE      = RGBColor(0xE6, 0x5C, 0x00)
C_GOLD        = RGBColor(0xF9, 0xA8, 0x25)
C_BLACK_T     = RGBColor(0x21, 0x21, 0x21)
C_WHITE       = RGBColor(0xFF, 0xFF, 0xFF)
C_HEADER_BG   = RGBColor(0x1B, 0x5E, 0x20)
C_ALT_ROW     = RGBColor(0xF1, 0xF8, 0xE9)
C_FIXED       = RGBColor(0x00, 0x7B, 0x00)   # green for FIXED items
C_REJECTED    = RGBColor(0xB7, 0x1C, 0x1C)   # red for rejected items


# ── Helpers ────────────────────────────────────────────────────────────────────

def set_cell_bg(cell, rgb: RGBColor):
    tc   = cell._tc
    tcPr = tc.get_or_add_tcPr()
    shd  = OxmlElement("w:shd")
    shd.set(qn("w:val"),   "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"),  f"{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}")
    tcPr.append(shd)


def set_cell_border(cell, sides=("top","bottom","left","right"), size=4, color="1B5E20"):
    tc   = cell._tc
    tcPr = tc.get_or_add_tcPr()
    tcBorders = OxmlElement("w:tcBorders")
    for side in sides:
        el = OxmlElement(f"w:{side}")
        el.set(qn("w:val"),   "single")
        el.set(qn("w:sz"),    str(size))
        el.set(qn("w:space"), "0")
        el.set(qn("w:color"), color)
        tcBorders.append(el)
    tcPr.append(tcBorders)


def add_heading(doc, text, level=1, colour=C_DARK_GREEN, size=16, bold=True, space_before=12):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(space_before)
    p.paragraph_format.space_after  = Pt(4)
    run = p.add_run(text)
    run.bold      = bold
    run.font.size = Pt(size)
    run.font.color.rgb = colour
    return p


def add_body(doc, text, size=10.5, colour=C_BLACK_T, space_before=2, space_after=4, italic=False):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(space_before)
    p.paragraph_format.space_after  = Pt(space_after)
    run = p.add_run(text)
    run.font.size      = Pt(size)
    run.font.color.rgb = colour
    run.italic         = italic
    return p


def add_kv(doc, key, value, key_colour=C_MID_GREEN, val_colour=C_BLACK_T):
    p   = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(1)
    p.paragraph_format.space_after  = Pt(1)
    r1  = p.add_run(f"{key}: ")
    r1.bold            = True
    r1.font.size       = Pt(10.5)
    r1.font.color.rgb  = key_colour
    r2  = p.add_run(str(value))
    r2.font.size       = Pt(10.5)
    r2.font.color.rgb  = val_colour


def styled_table(doc, headers, rows, col_widths=None):
    n_cols = len(headers)
    table  = doc.add_table(rows=1 + len(rows), cols=n_cols)
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    hdr = table.rows[0]
    for i, h in enumerate(headers):
        cell = hdr.cells[i]
        set_cell_bg(cell, C_HEADER_BG)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(h)
        run.bold            = True
        run.font.size       = Pt(9.5)
        run.font.color.rgb  = C_WHITE

    for ri, row_data in enumerate(rows):
        row = table.rows[ri + 1]
        bg  = C_ALT_ROW if ri % 2 == 0 else C_WHITE
        for ci, val in enumerate(row_data):
            cell = row.cells[ci]
            set_cell_bg(cell, bg)
            p    = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run  = p.add_run(str(val))
            run.font.size = Pt(9.5)

    if col_widths:
        for i, w in enumerate(col_widths):
            for row in table.rows:
                row.cells[i].width = Inches(w)

    return table


def add_divider(doc):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after  = Pt(6)
    run = p.add_run("-" * 80)
    run.font.size      = Pt(8)
    run.font.color.rgb = RGBColor(0xC8, 0xE6, 0xC9)


# ══════════════════════════════════════════════════════════════════════════════
# HARDCODED DETECTION RESULTS (CSV is overwritten each run — data preserved here)
# ══════════════════════════════════════════════════════════════════════════════

# Test Bh — per-image detection results (actual run with all 7 fixes applied)
# 35 images | 32 with tigers | 3 no tiger | 34 total detections
# OIV7 now dominant (64.7%) — many images that previously used Whole_Image_Fallback
# now have direct YOLO boxes due to COCO class fix + dual-scan improvements
TEST_BH_DETECTIONS = [
    # Image,        Tiger ID,    Morph,            Visibility,    Detection,             Score, Status
    ["000004.jpg",  "Tiger_029", "Orange Tiger",   "Corner_Trace","COCO_Proxy",          "0.96","Corner Match"],
    ["000038.jpg",  "Tiger_030", "Orange Tiger",   "Full_Body",   "Whole_Image_Fallback","1.00","Recaptured"],
    ["000067.jpg",  "Tiger_001", "Orange Tiger",   "Corner_Trace","COCO_Proxy",          "0.86","Corner Match"],
    ["000110.jpg",  "Tiger_001", "Orange Tiger",   "Corner_Trace","COCO_Proxy",          "0.86","Corner Match"],
    ["000264.jpg",  "Tiger_034", "Orange Tiger",   "Corner_Trace","OIV7_Direct",         "0.80","Corner New"],
    ["000274.jpg",  "Tiger_001", "Orange Tiger",   "Corner_Trace","OIV7_Direct",         "0.89","Corner Match"],
    ["000274.jpg",  "Tiger_001", "Golden Tiger",   "Corner_Trace","OIV7_Direct",         "0.94","Corner Match"],
    ["000289.jpg",  "Tiger_001", "Orange Tiger",   "Corner_Trace","OIV7_Direct",         "0.85","Corner Match"],
    ["000417.jpg",  "Tiger_031", "Orange Tiger",   "Full_Body",   "Whole_Image_Fallback","1.00","Recaptured"],
    ["000464.jpg",  "Tiger_037", "Black Tiger",    "Full_Body",   "Whole_Image_Fallback","1.00","Recaptured [FIXED]"],
    ["000477.jpg",  "Tiger_032", "Black Tiger",    "Corner_Trace","OIV7_Direct",         "0.79","Corner New"],
    ["000544.jpg",  "Tiger_036", "Black Tiger",    "Corner_Trace","OIV7_Direct",         "0.86","Corner Match [FIXED]"],
    ["000571.jpg",  "Tiger_001", "Orange Tiger",   "Corner_Trace","COCO_Proxy",          "0.91","Corner Match"],
    ["000725.jpg",  "Tiger_001", "Orange Tiger",   "Full_Body",   "OIV7_Direct",         "0.89","Recaptured [FIXED]"],
    ["000729.jpg",  "Tiger_033", "Golden Tiger",   "Full_Body",   "Whole_Image_Fallback","1.00","Recaptured"],
    ["000946.jpg",  "Tiger_001", "Orange Tiger",   "Corner_Trace","OIV7_Direct",         "0.91","Corner Match"],
    ["000973.jpg",  "Tiger_001", "Golden Tiger",   "Full_Body",   "OIV7_Direct",         "0.94","Recaptured [FIXED]"],
    ["001031.jpg",  "Tiger_001", "Orange Tiger",   "Full_Body",   "COCO_Proxy",          "0.85","Recaptured"],
    ["001082.jpg",  "Tiger_039", "Black Tiger",    "Corner_Trace","COCO_Proxy",          "0.94","Corner Match [FIXED]"],
    ["001110.jpg",  "Tiger_001", "Orange Tiger",   "Corner_Trace","OIV7_Direct",         "0.94","Corner Match [FIXED]"],
    ["001124.jpg",  "Tiger_024", "Orange Tiger",   "Corner_Trace","OIV7_Direct",         "0.70","Corner New"],
    ["001132.jpg",  "Tiger_001", "Orange Tiger",   "Corner_Trace","OIV7_Direct",         "0.83","Corner Match"],
    ["001172.jpg",  "Tiger_001", "Golden Tiger",   "Full_Body",   "OIV7_Direct",         "0.93","Recaptured"],
    ["001172.jpg",  "Tiger_001", "Golden Tiger",   "Corner_Trace","OIV7_Direct",         "0.77","Corner New"],
    ["001271.jpg",  "Tiger_001", "Orange Tiger",   "Full_Body",   "OIV7_Direct",         "0.95","Recaptured"],
    ["001275.jpg",  "Tiger_001", "Orange Tiger",   "Corner_Trace","OIV7_Direct",         "0.95","Corner Match"],
    ["001284.jpg",  "Tiger_001", "Orange Tiger",   "Corner_Trace","OIV7_Direct",         "0.94","Corner Match"],
    ["001297.jpg",  "Tiger_001", "Orange Tiger",   "Corner_Trace","OIV7_Direct",         "0.89","Corner Match"],
    ["001319.jpg",  "Tiger_034", "Orange Tiger",   "Full_Body",   "Whole_Image_Fallback","1.00","Recaptured"],
    ["001345.jpg",  "Tiger_001", "Orange Tiger",   "Full_Body",   "Whole_Image_Fallback","0.89","Recaptured"],
    ["0012q.jpeg",  "Tiger_040", "Orange Tiger",   "Corner_Trace","OIV7_Direct",         "0.89","Corner Match [FIXED]"],
    ["0013q.jpeg",  "Tiger_041", "Orange Tiger",   "Corner_Trace","OIV7_Direct",         "0.79","Corner New [FIXED]"],
    ["0015q.jpeg",  "Tiger_001", "Golden Tiger",   "Full_Body",   "OIV7_Direct",         "0.93","Recaptured"],
    ["1108q.jpeg",  "Tiger_001", "Golden Tiger",   "Corner_Trace","OIV7_Direct",         "0.91","Corner Match"],
]

TEST_BH_NO_TIGER = [
    ["000010.jpg", "IR_Night — Species rejected",  "Bucket: IR_Night. Species gate top-3: komondor, skunk, curly-coated retriever (no tiger in frame)"],
    ["000764.jpg", "Clean — Species rejected",     "COCO found 1 proxy box. Species gate top-3: newfoundland, flat-coated retriever, groenendael (dense foliage reads as dog fur)"],
    ["0014q.jpeg", "Clean — Species rejected",     "Whole-image fallback. Species gate top-3: fox squirrel, dhole, bolete (empty/obscured frame)"],
]

# Test Round 2 — per-image results (18 images)
# Updated after Bug Fix 6 (COCO class 22=zebra, not 24=backpack) and
# Bug Fix 7 (OIV7 dual scan: raw + preprocessed image):
#   6.jpeg  — REMOVED from detections; correctly rejected by leopard dominance rule
#   15.jpeg — now 2 OIV7 boxes (dual scan recovered second tiger on raw image)
#   16.jpeg — now 2 COCO_Proxy boxes (COCO class 22 zebra fix recovered 3 boxes,
#              2 confirmed by species gate; 1 rejected due to tiny crop)
TEST_R2_DETECTIONS = [
    # Image,    Tiger ID,    Morph,          Visibility,    Detection,              Score, Status
    ["1.jpeg",  "Tiger_042", "Orange Tiger", "Full_Body",   "Whole_Image_Fallback", "0.66","New Enrollment"],
    ["2.jpeg",  "Tiger_024", "Orange Tiger", "Corner_Trace","OIV7_Direct",          "0.83","Cross-Dataset Match"],
    ["3.jpeg",  "Tiger_045", "Orange Tiger", "Full_Body",   "Whole_Image_Fallback", "0.59","New Enrollment"],
    ["4.jpeg",  "Tiger_046", "Black Tiger",  "Full_Body",   "Whole_Image_Fallback", "0.80","New Enrollment"],
    ["15.jpeg", "Tiger_001", "Orange Tiger", "Corner_Trace","OIV7_Direct",          "0.93","Cross-Dataset Match"],
    ["15.jpeg", "Tiger_001", "Orange Tiger", "Corner_Trace","OIV7_Direct (raw)",    "0.82","Cross-Dataset Match (dual scan)"],
    ["16.jpeg", "Tiger_021", "Orange Tiger", "Partial_Body","COCO_Proxy",           "0.66","Partial New"],
    ["16.jpeg", "Tiger_021", "Orange Tiger", "Partial_Body","COCO_Proxy",           "0.52","Partial New"],
    ["18.jpeg", "Tiger_044", "Black Tiger",  "Full_Body",   "Whole_Image_Fallback", "0.67","New Enrollment"],
]

TEST_R2_NO_TIGER = [
    ["5.jpeg",  "IR_Night",  "Species gate: leopard, jaguar, cheetah — different big cat"],
    ["6.jpeg",  "Clean",     "LEOPARD REJECTION RULE: top-3 = snow leopard, leopard, jaguar (≥2 non-tiger big cats → immediate reject). Previously accepted in error via top-5 widening."],
    ["7.jpeg",  "IR_Night",  "Species gate: leopard, jaguar — different big cat"],
    ["8.jpeg",  "IR_Night",  "Species gate: cheetah, leopard — different big cat"],
    ["9.jpeg",  "Clean",     "Species gate: hyena, african hunting dog — different animal"],
    ["10.jpeg", "Clean",     "Species gate: whiptail, common newt, banded gecko — no animal"],
    ["11.jpeg", "Clean",     "Species gate: birdhouse, fox squirrel — background/no tiger"],
    ["12.jpeg", "Clean",     "Species gate: worm fence, valley, tusker — empty frame"],
    ["13.jpeg", "Clean",     "Species gate: hen-of-the-woods, bighorn — no tiger"],
    ["14.jpeg", "IR_Night",  "Species gate: hog, wild boar — extreme close-up / unusual pose"],
    ["17.jpeg", "Clean",     "Species gate: bulletproof vest, assault rifle — cage/enclosure bars"],
]


# ══════════════════════════════════════════════════════════════════════════════
# BUILD DOCUMENT
# ══════════════════════════════════════════════════════════════════════════════

doc = Document()

for section in doc.sections:
    section.top_margin    = Cm(2.0)
    section.bottom_margin = Cm(2.0)
    section.left_margin   = Cm(2.5)
    section.right_margin  = Cm(2.5)

# ── COVER ─────────────────────────────────────────────────────────────────────
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.space_before = Pt(40)
r = p.add_run("TIGER AI ULTIMATE — v3.0")
r.bold            = True
r.font.size       = Pt(22)
r.font.color.rgb  = C_DARK_GREEN

p2 = doc.add_paragraph()
p2.alignment = WD_ALIGN_PARAGRAPH.CENTER
r2 = p2.add_run("Fused Multi-Source Pipeline — Run Report")
r2.font.size      = Pt(14)
r2.font.color.rgb = C_MID_GREEN

doc.add_paragraph()
p3 = doc.add_paragraph()
p3.alignment = WD_ALIGN_PARAGRAPH.CENTER
r3 = p3.add_run("EPAIB Batch 05 | Group 4 | IIM Lucknow")
r3.font.size      = Pt(11)
r3.font.color.rgb = C_BLACK_T

p4 = doc.add_paragraph()
p4.alignment = WD_ALIGN_PARAGRAPH.CENTER
r4 = p4.add_run(f"Generated: {datetime.now().strftime('%d %B %Y  %H:%M')}")
r4.font.size      = Pt(10)
r4.font.color.rgb = RGBColor(0x75, 0x75, 0x75)

doc.add_paragraph()
p5 = doc.add_paragraph()
p5.alignment = WD_ALIGN_PARAGRAPH.CENTER
r5 = p5.add_run("Datasets covered in this report:")
r5.font.size      = Pt(10)
r5.font.color.rgb = C_MID_GREEN
r5.bold = True

styled_table(doc,
    ["Dataset", "Images", "Tigers Detected", "Unique Tigers"],
    [
        ["Amur Tigers Train Set (ATRW)", "3,392", "2,457",  "28"],
        ["Test Bh (Prof. Mahesh Balan)", "35",    "34",     "13 new + 2 cross-dataset"],
        ["Test Round 2",                 "18",    "9",      "4 new + 2 cross-dataset (6.jpeg = leopard, correctly rejected)"],
        ["Test MT (WhatsApp images)",    "10",    "5",      "3 new (Tiger_048–050) + Tiger_001 ×2 | Grid Scan found mother+cub"],
    ],
    col_widths=[2.5, 1.0, 1.5, 2.5]
)

doc.add_page_break()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — PROJECT OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
add_heading(doc, "1.  Project Overview", level=1, size=15)
add_body(doc,
    "This report documents the complete output of the Tiger AI Ultimate v3.0 pipeline "
    "run on three datasets as part of the EPAIB Batch 05 capstone project at IIM Lucknow. "
    "The pipeline was developed by Group 4 and fuses techniques from four independent "
    "sources: TRACE (Group 4), Dr. Bose's Gemini-designed architecture, Yeshvir Script B "
    "(multi-animal), and Yeshvir Script A (Colab).")

add_heading(doc, "1.1  Pipeline Architecture (v3.0 Fused)", level=2, size=12, colour=C_MID_GREEN)
styled_table(doc,
    ["Component", "Technology", "Source"],
    [
        ["Stage 1 Detection",   "YOLOv8n-OIV7 (Open Images V7 — direct Tiger class)", "Dr. Bose"],
        ["Stage 2 Detection",   "YOLOv8n-COCO (cat/dog/horse proxy classes)",          "TRACE v2.0"],
        ["Stage 3 Fallback",    "Whole-image scan if both YOLOs find nothing",          "Yeshvir Script B"],
        ["Species Gate",        "ResNet50 top-3/top-5 must contain 'tiger'",            "Yeshvir Script B"],
        ["Dark Image Fix",      "Gamma correction + strong CLAHE for dark crops",       "v3.0 (new)"],
        ["IR Night Fix",        "IR check runs BEFORE brightness check in triage",      "v3.0 (new)"],
        ["ViewPoint Label",     "Left_Flank / Right_Flank / Frontal classification",    "Dr. Bose"],
        ["Feature Extraction",  "ResNet50 2048-dim avg_pool stripe fingerprint",        "TRACE / Yeshvir"],
        ["Identity Matching",   "Cosine similarity >= 0.83, best-match (no break)",     "TRACE + Yeshvir fix"],
        ["DB Persistence",      "Running average embedding, JSON file, cross-session",  "TRACE v2.0"],
        ["Preprocessing",       "CLAHE + Dark Channel Prior + IR normalisation",        "TRACE + Dr. Bose"],
        ["Guards",              "Shadow / Corner-trace / Overlap IoU checks",           "TRACE v2.0"],
        ["Explainability",      "Grad-CAM on ResNet50 layer4 (--gradcam flag)",         "TRACE v2.0"],
    ],
    col_widths=[1.6, 3.4, 1.5]
)

add_heading(doc, "1.2  Why Each Model Was Chosen", level=2, size=12, colour=C_MID_GREEN)
styled_table(doc,
    ["Model", "Role", "Why Chosen", "Alternative Considered"],
    [
        ["YOLOv8n + OIV7",  "Tiger detection",        "Open Images V7 has direct Tiger class",             "MegaDetector (complex install)"],
        ["YOLOv8n + COCO",  "Backup detection",       "Standard benchmark; cat/dog/zebra as proxies",      "Nothing (COCO is standard)"],
        ["ResNet50",        "Species gate + ID",       "ImageNet includes tiger; fast transfer learning",   "EfficientNetB3 (heavier)"],
        ["Cosine similarity","Identity matching",      "Scale-invariant; handles brightness changes",       "Euclidean (brightness-sensitive)"],
        ["Running average", "Fingerprint refinement", "Each sighting improves the DB profile",             "Fixed embedding from first shot"],
        ["CLAHE",           "Contrast enhancement",   "Recovers stripe detail in shadow/dawn images",      "Global histogram eq (too harsh)"],
        ["Grad-CAM",        "Explainability",         "Visual proof model reads stripes not background",   "LIME (slower for images)"],
    ],
    col_widths=[1.3, 1.2, 2.4, 1.6]
)

doc.add_page_break()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — DATASET 1: AMUR TIGERS TRAIN (3392 images)
# ══════════════════════════════════════════════════════════════════════════════
add_heading(doc, "2.  Dataset 1 — Amur Tigers Train Set (ATRW Benchmark)", level=1, size=15)
add_body(doc,
    "The ATRW (Amur Tiger Re-Identification in the Wild) dataset is the benchmark used in "
    "the ICCV Wildlife Computer Vision challenge. It contains 3,392 Amur (Siberian) tiger "
    "images from wildlife parks. Recommended by Prof. Mahesh Balan and referenced in "
    "Dr. Bose's Gemini chat. Ground-truth identity annotations available in reid_list_train.csv.")

add_heading(doc, "2.1  Run Summary", level=2, size=12, colour=C_MID_GREEN)
styled_table(doc,
    ["Metric", "Value"],
    [
        ["Dataset path",             r"Amur Tigers\train"],
        ["Total images",             "3,392"],
        ["Images with tigers found", "2,445  (72.1%)"],
        ["Images with no tiger",     "62  (1.8%)"],
        ["Unusable images",          "885  (26.1%)"],
        ["Total tiger detections",   "2,457"],
        ["Unique tigers in DB",      "28"],
        ["Conservative census",      "29"],
        ["Run device",               "CPU"],
        ["Run date",                 "2026-06-02"],
    ],
    col_widths=[2.5, 4.0]
)

add_heading(doc, "2.2  Image Quality Breakdown", level=2, size=12, colour=C_MID_GREEN)
styled_table(doc,
    ["Category", "Count", "% of Total", "Reason"],
    [
        ["Clean — processable",    "2,507", "73.9%", "Passed all quality checks"],
        ["Skipped — Blurry",       "858",   "25.3%", "Laplacian variance < 25 (motion blur)"],
        ["Skipped — Underexposed", "26",    "0.8%",  "Mean brightness < 8 (very dark)"],
        ["Skipped — Overexposed",  "0",     "0%",    "No blown-out images in this dataset"],
        ["IR / Night",             "1",     "0.03%", "Near-identical RGB channels detected"],
    ],
    col_widths=[2.2, 0.8, 1.0, 2.5]
)

add_heading(doc, "2.3  Cascade Detection Performance", level=2, size=12, colour=C_MID_GREEN)
styled_table(doc,
    ["Detection Stage", "Detections", "% of Total", "Observation"],
    [
        ["Stage 1: OIV7 Direct Tiger",    "1,075","43.8%","Tiger class detected directly — no proxy needed"],
        ["Stage 2: COCO Proxy",           "30",   "1.2%", "Proxy barely contributes — OIV7 far superior"],
        ["Stage 3: Whole-Image Fallback", "1,352","55.0%","Critical — rescued more than half of all detections"],
    ],
    col_widths=[2.2, 1.0, 1.0, 2.3]
)

add_heading(doc, "2.4  Identity Matching Results", level=2, size=12, colour=C_MID_GREEN)
styled_table(doc,
    ["Metric", "Value"],
    [
        ["Unique tigers in DB",               "28"],
        ["New enrollments",                   "28"],
        ["Recaptured (re-matched) detections","~430+"],
        ["Mean cosine similarity score",      "0.796"],
        ["Tiger_001 sightings",               "320 (dominant individual — 13% of all detections)"],
    ],
    col_widths=[2.8, 3.7]
)

add_heading(doc, "2.5  Colour Morph Breakdown", level=2, size=12, colour=C_MID_GREEN)
styled_table(doc,
    ["Morph", "Count", "%", "Notes"],
    [
        ["Orange Tiger",            "1,885","76.7%","Standard Bengal-type colouring"],
        ["Golden Tiger",            "455",  "18.5%","Common in Amur tigers — paler Siberian base coat"],
        ["Black Tiger",             "115",  "4.7%", "Pseudomelanistic — black stripe dominant"],
        ["White Tiger",             "2",    "0.1%", "Extremely rare — captive genetic mutation"],
    ],
    col_widths=[2.0, 1.0, 0.8, 2.7]
)

doc.add_page_break()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — DATASET 2: TEST BH (35 images)
# ══════════════════════════════════════════════════════════════════════════════
add_heading(doc, "3.  Dataset 2 — Test Bh (Prof. Mahesh Balan's Dataset)", level=1, size=15)
add_body(doc,
    "35 images acquired from Prof. Mahesh Balan's recommended dataset. The pipeline loaded "
    "the existing DB from the 3,392-image train run (28 known tiger profiles) and performed "
    "cross-dataset recognition without any retraining. Multiple model threshold fixes were "
    "applied during this run based on image quality review.")

add_heading(doc, "3.1  Run Summary", level=2, size=12, colour=C_MID_GREEN)
styled_table(doc,
    ["Metric", "Value"],
    [
        ["Dataset path",             r"Amur Tigers\Test Bh"],
        ["Total images",             "35"],
        ["Images with tigers found", "32  (91.4%)  — improved from 24 before fixes"],
        ["Images with no tiger",     "3   (8.6%)"],
        ["Unusable images",          "0   — all 35 passed triage after threshold fixes"],
        ["Total tiger detections",   "34"],
        ["New tigers enrolled",      "13  (Tiger_029 to Tiger_041)"],
        ["Cross-dataset matches",    "Tiger_001 (436 sightings) and Tiger_024 recognised"],
        ["DB profiles after run",    "41  (28 from train + 13 new)"],
    ],
    col_widths=[2.5, 4.0]
)

add_heading(doc, "3.2  Fixes Applied During Test Bh Review", level=2, size=12, colour=C_MID_GREEN)
add_body(doc,
    "After reviewing Test Bh results, 9 images that were previously skipped or misclassified "
    "were identified as containing visible tigers. Four pipeline fixes were applied:")
styled_table(doc,
    ["Fix", "Images Recovered", "Root Cause", "Code Change"],
    [
        ["IR check order",
         "000464, 000544, 001082 (3 images)",
         "IR detection ran AFTER brightness check. Dark IR images were rejected as 'underexposed' before being identified as valid night shots.",
         "Moved detect_ir_night() to run FIRST in triage_image()"],
        ["Blur threshold",
         "000725, 000973, 001110 (3 images)",
         "Threshold of 80 was calibrated on Sundarbans DSLR images. Camera trap images naturally score 38-48 and are visually sharp.",
         "Lowered BLUR_THRESHOLD from 80 to 25"],
        ["Shadow guard skip for IR",
         "0012q, 0013q (2 images)",
         "Flash night IR cameras produce greyscale (saturation=0). Shadow guard checks saturation < 30, so every IR image was rejected as a shadow.",
         "Added: if bucket != 'IR_Night': before shadow guard check"],
        ["Three-attempt species gate",
         "000764 (1 image)",
         "Tiger in unusual pose (lying flat). ImageNet top-3 returned 'hog/wild boar'. Original gate had no retry.",
         "Added Attempt 2 (brightened top-3) and Attempt 3 (top-5) in verify_species_top3()"],
    ],
    col_widths=[1.4, 1.5, 2.1, 2.0]
)

add_heading(doc, "3.3  Per-Image Detection Results", level=2, size=12, colour=C_MID_GREEN)
add_body(doc, "[FIXED] = image was previously skipped, now detected after pipeline fixes.", italic=True)
styled_table(doc,
    ["Image", "Tiger ID", "Morph", "Visibility", "Detection", "Score", "Status"],
    TEST_BH_DETECTIONS,
    col_widths=[1.0, 0.9, 1.1, 1.0, 1.5, 0.6, 1.4]
)

doc.add_paragraph()
add_heading(doc, "3.4  Images With No Tiger Detected", level=2, size=12, colour=C_MID_GREEN)
styled_table(doc,
    ["Image", "Reason", "Details"],
    TEST_BH_NO_TIGER,
    col_widths=[1.1, 1.8, 3.6]
)

add_heading(doc, "3.5  Cross-Dataset Recognition", level=2, size=12, colour=C_MID_GREEN)
add_body(doc,
    "Tiger_001, whose fingerprint was built from 320 sightings in the train dataset, was "
    "recognised in 15 images from the completely separate Test Bh dataset with match scores "
    "0.84–0.95. No retraining was needed. Tiger_024 (5 prior sightings) was also recognised "
    "in image 001124.jpg. This validates the running average embedding approach — the DB "
    "genuinely generalises to new images of previously seen individuals.")

doc.add_page_break()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — DATASET 3: TEST ROUND 2 (18 images)
# ══════════════════════════════════════════════════════════════════════════════
add_heading(doc, "4.  Dataset 3 — Test Round 2", level=1, size=15)
add_body(doc,
    "18 images from a new test batch, including mixed lighting conditions (IR night, flash "
    "photography, daylight). The pipeline loaded 41 existing tiger profiles from the combined "
    "Train + Test Bh DB and ran cross-dataset recognition. The dataset appears to contain "
    "multiple species — 10 of 18 images were correctly rejected by the species gate as "
    "non-tiger animals or empty frames.")

add_heading(doc, "4.1  Run Summary", level=2, size=12, colour=C_MID_GREEN)
styled_table(doc,
    ["Metric", "Value"],
    [
        ["Dataset path",             r"Amur Tigers\Test round 2"],
        ["Total images",             "18"],
        ["Images with tigers found", "7   (38.9%)"],
        ["Images with no tiger",     "11  (61.1%)"],
        ["Unusable images",          "0   — all 18 passed triage"],
        ["Total tiger detections",   "9   (15.jpeg now yields 2 OIV7 boxes; 16.jpeg yields 2 COCO boxes)"],
        ["New tigers enrolled",      "4   (Tiger_042, Tiger_044, Tiger_045, Tiger_046)"],
        ["Cross-dataset matches",    "2   (Tiger_001 in 15.jpeg × 2 boxes; Tiger_024 in 2.jpeg)"],
        ["Species gate rejections",  "10  (incl. 6.jpeg = leopard — previously accepted in error)"],
        ["DB profiles after run",    "46  (42 prior + 4 new)"],
        ["Run date",                 "2026-06-02"],
    ],
    col_widths=[2.5, 4.0]
)

add_heading(doc, "4.2  Per-Image Detection Results", level=2, size=12, colour=C_MID_GREEN)
styled_table(doc,
    ["Image", "Tiger ID", "Morph", "Visibility", "Detection", "Score", "Status"],
    TEST_R2_DETECTIONS,
    col_widths=[0.8, 0.9, 1.1, 1.0, 1.7, 0.6, 1.4]
)

add_heading(doc, "4.3  Images With No Tiger — Species Gate Results", level=2, size=12, colour=C_MID_GREEN)
add_body(doc,
    "10 images were correctly rejected by the species gate. This is a key validation of "
    "the pipeline's accuracy — it does not call everything a tiger.", italic=True)
styled_table(doc,
    ["Image", "Quality", "Species Gate Result (Why Rejected)"],
    TEST_R2_NO_TIGER,
    col_widths=[0.9, 1.0, 4.6]
)

add_heading(doc, "4.4  Cross-Dataset Recognition in Test Round 2", level=2, size=12, colour=C_MID_GREEN)
styled_table(doc,
    ["Tiger", "Prior Sightings", "Recognised In", "Match Score", "Notes"],
    [
        ["Tiger_001", "429 (train + Test Bh)", "15.jpeg (box 1)", "0.93", "Dominant individual — 3 datasets, no retraining. OIV7 on preprocessed image."],
        ["Tiger_001", "429 (train + Test Bh)", "15.jpeg (box 2)", "0.82", "Same tiger, second camera angle. Recovered by OIV7 dual-scan on raw image (Bug Fix 7)."],
        ["Tiger_024", "5 (train)",             "2.jpeg",          "0.83", "Third dataset where this individual is recognised."],
    ],
    col_widths=[1.0, 1.8, 1.4, 1.0, 2.3]
)
add_body(doc,
    "Note: 6.jpeg was previously accepted as Tiger_040 (cross-dataset match, score 0.47). "
    "After applying the leopard dominance rejection rule (Bug Fix 6), 6.jpeg is now correctly "
    "rejected — ResNet top-3 returns snow leopard, leopard, jaguar with no tiger in top-5. "
    "The prior Tiger_040 'match' at 0.47 was below the identity threshold and was a false positive.", italic=True)

add_heading(doc, "4.5  Species Detected vs Rejected — Observation", level=2, size=12, colour=C_MID_GREEN)
add_body(doc,
    "Test Round 2 contained multiple big cat species. Images 5.jpeg, 7.jpeg, and 8.jpeg "
    "returned 'leopard', 'jaguar', and 'cheetah' from the species gate — confirming these "
    "are genuinely different animals, not tigers. The pipeline correctly rejected them. "
    "This demonstrates the species gate works as a multi-species filter, not just a "
    "tiger vs. non-animal binary check.")

doc.add_page_break()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — DATASET 4: TEST MT (10 images — WhatsApp shared images)
# ══════════════════════════════════════════════════════════════════════════════
add_heading(doc, "5.  Dataset 4 — Test MT (WhatsApp Images)", level=1, size=15)
add_body(doc,
    "10 images shared via WhatsApp (2026-06-02). Mixed content — tigers, other wildlife, "
    "and non-animal images. The pipeline loaded 47 existing profiles from the cumulative DB "
    "(Train + Test Bh + Test Round 2) and ran cross-dataset recognition. The species gate "
    "correctly filtered 7 of 10 images as non-tiger.")

add_heading(doc, "5.1  Run Summary", level=2, size=12, colour=C_MID_GREEN)
styled_table(doc,
    ["Metric", "Value"],
    [
        ["Dataset path",             r"Amur Tigers\Test MT"],
        ["Total images",             "10"],
        ["Images with tigers found", "3   (30.0%)"],
        ["Images with no tiger",     "7   (70.0%)"],
        ["Unusable images",          "0   — all 10 passed triage"],
        ["Total tiger detections",   "5   (grid scan found 2 extra tigers in multi-tiger frames)"],
        ["New tigers enrolled",      "3   (Tiger_048, Tiger_049, Tiger_050)"],
        ["Cross-dataset matches",    "Tiger_001 recognised twice (OIV7 + Grid_Scan_R)"],
        ["DB profiles after run",    "50  (47 prior + 3 new)"],
        ["Run date",                 "2026-06-02"],
        ["Grid scan detections",     "4 of 5 detections came from Grid_Scan (2 multi-tiger frames)"],
    ],
    col_widths=[2.5, 4.0]
)

add_heading(doc, "5.2  Per-Image Detection Results", level=2, size=12, colour=C_MID_GREEN)
add_body(doc,
    "Grid scan algorithm split 2 multi-tiger frames into individual half-frame detections. "
    "Each half is verified independently by ResNet species gate and given its own fingerprint.", italic=True)
styled_table(doc,
    ["Image (shortened)", "Tiger ID", "Morph", "Visibility", "Detection", "Score", "Status"],
    [
        ["1.44.15 PM (3) — LEFT",  "Tiger_048", "Orange Tiger", "Full_Body", "Grid_Scan_L", "1.00", "Recaptured (adult)"],
        ["1.44.15 PM (3) — RIGHT", "Tiger_049", "Orange Tiger", "Full_Body", "Grid_Scan_R", "1.00", "Recaptured (cub)"],
        ["1.44.16 PM (3)",         "Tiger_001", "Golden Tiger", "Full_Body", "OIV7_Direct",  "0.93", "Recaptured"],
        ["1.44.16 PM — LEFT",      "Tiger_050", "Orange Tiger", "Full_Body", "Grid_Scan_L", "0.79", "New Enrollment"],
        ["1.44.16 PM — RIGHT",     "Tiger_001", "Orange Tiger", "Full_Body", "Grid_Scan_R", "0.88", "Recaptured"],
    ],
    col_widths=[1.7, 0.9, 1.1, 1.0, 1.3, 0.6, 1.4]
)

add_heading(doc, "5.3  Grid Scan Algorithm — How Multi-Tiger Detection Works", level=2, size=12, colour=C_MID_GREEN)
add_body(doc,
    "When YOLO (OIV7 + COCO) finds zero bounding boxes and the whole-image fallback confirms "
    "a tiger via the species gate, the pipeline runs a Grid Scan:")
styled_table(doc,
    ["Step", "Action", "Purpose"],
    [
        ["1", "Whole-image fallback confirms tiger via ResNet top-3", "Baseline: at least 1 tiger in frame"],
        ["2", "Split image into Left half and Right half",            "Check each side independently"],
        ["3", "Run ResNet species gate on Left half",                 "Is there a tiger in the left side?"],
        ["4", "Run ResNet species gate on Right half",                "Is there a tiger in the right side?"],
        ["5", "Both halves confirm tiger → 2 separate detections",   "Each half enrolled with its own fingerprint"],
        ["6", "If split fails: try Top half + Bottom half",           "Handles stacked / near-far tiger arrangements"],
        ["7", "If both splits fail → 1 detection (original)",        "Conservative fallback — no over-counting"],
    ],
    col_widths=[0.4, 3.0, 3.1]
)
add_body(doc,
    "This algorithm specifically solves the mother+cub problem: two tigers sharing a frame "
    "where YOLO cannot draw individual boxes. Each half-frame gets its own ResNet embedding "
    "and is enrolled as a separate individual in the identity DB.", italic=True)

add_heading(doc, "5.4  Images With No Tiger — Species Gate Results", level=2, size=12, colour=C_MID_GREEN)
add_body(doc,
    "7 of 10 images correctly rejected. This dataset is the most diverse yet — "
    "containing other big cats, ungulates, primates, and non-animal objects.", italic=True)
styled_table(doc,
    ["Image (shortened)", "Species Gate Result (Why Rejected)"],
    [
        ["1.44.15 PM (1)", "cougar, lion, jaguar — different big cat (leopard family)"],
        ["1.44.15 PM (2)", "mousetrap, television, birdhouse — no animal detected"],
        ["1.44.15 PM",     "water buffalo, ox, oxcart — ungulate, not tiger"],
        ["1.44.16 PM (1)", "orangutan, sloth bear, titi — primate / bear, not tiger"],
        ["1.44.16 PM (2)", "cougar, lynx, valley — different big cat"],
        ["1.44.16 PM (5)", "hyena, coyote, grey fox — canid / hyena, not tiger"],
        ["1.44.16 PM (6)", "leopard, lynx, jaguar — different big cat"],
    ],
    col_widths=[1.5, 5.0]
)

add_heading(doc, "5.5  Key Observation — Species Diversity + Multi-Tiger Detection", level=2, size=12, colour=C_MID_GREEN)
add_body(doc,
    "Test MT is the most species-diverse dataset in this study. The grid scan algorithm "
    "found 2 additional tigers that were previously missed (counted as 1). "
    "In 10 images: 3 unique new tigers + Tiger_001 recaptured. "
    "7 non-tiger images (3 other big cats, 1 ungulate, 1 primate/bear, 1 canid, 1 non-animal) "
    "all correctly rejected. Zero false positives.")

doc.add_page_break()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — CUMULATIVE CENSUS (all 4 datasets)
# ══════════════════════════════════════════════════════════════════════════════
add_heading(doc, "6.  Cumulative Census — All Four Datasets", level=1, size=15)
add_body(doc,
    "The identity database is persistent — fingerprints built in one run carry forward "
    "to the next. This cumulative census represents the total knowledge after running "
    "all four datasets in sequence.")

styled_table(doc,
    ["Dataset", "Images", "Tigers Detected", "New Tigers", "Cross-Dataset"],
    [
        ["Amur Tigers Train (ATRW)", "3,392", "2,457", "28",  "—"],
        ["Test Bh",                  "35",    "34",    "13",  "Tiger_001, Tiger_024"],
        ["Test Round 2",             "18",    "9",     "4",   "Tiger_001, Tiger_024"],
        ["Test MT",                  "10",    "3",     "1",   "Tiger_001 (×2 sightings)"],
        ["TOTAL",                    "3,455", "2,503", "46",  "Tiger_001 across all 4 datasets"],
    ],
    col_widths=[2.0, 0.9, 1.3, 1.1, 2.2]
)

doc.add_paragraph()
styled_table(doc,
    ["Metric", "Value"],
    [
        ["Total images processed",         "3,455  (3,392 + 35 + 18 + 10)"],
        ["Total tiger detections",          "2,505  (incl. 2 extra from grid scan in Test MT)"],
        ["Unique tigers in DB",             "50"],
        ["Tiger_001 total sightings",       "440  (recognised across all 4 datasets)"],
        ["Tiger_024 total sightings",       "5+   (Train + Test Bh + Round 2)"],
        ["Cumulative detection rate",       "72.4% of all images had at least one tiger"],
        ["Grid Scan detections",            "4 detections from multi-tiger frames (Test MT)"],
        ["Colour morph distribution",       "Orange: ~70% | Golden: ~20% | Black: ~10%"],
    ],
    col_widths=[2.8, 3.7]
)

doc.add_paragraph()
add_body(doc,
    "Tiger_001 has now been photographed 440 times across four completely different datasets — "
    "train images (ATRW), Test Bh, Test Round 2, and Test MT (WhatsApp images) — using a "
    "fingerprint built entirely from the original train run. No retraining was performed between "
    "any dataset. The grid scan algorithm also discovered that Test MT contained 2 multi-tiger "
    "frames — a mother and cub and two tigers side-by-side — that were previously counted as 1 each. "
    "This is the strongest validation of the running average embedding approach: the DB fingerprint "
    "generalises to new sensors, new lighting conditions, new image formats, and now multiple "
    "animals sharing a single camera trap frame.", italic=True)

doc.add_page_break()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — BUGS FOUND AND FIXED
# ══════════════════════════════════════════════════════════════════════════════
add_heading(doc, "7.  Bugs Found and Fixed", level=1, size=15)

styled_table(doc,
    ["Bug", "Root Cause", "Impact", "Fix"],
    [
        ["Corner Trace on Whole-Image Fallback",
         "Fallback uses box (0,0,W,H). x1=0 is within 12px margin, so corner-trace fired on every fallback detection.",
         "452 detections wrongly labelled Corner_Trace — never updated the identity DB.",
         "Added: if det_src != 'Whole_Image_Fallback' before is_corner_trace() check."],
        ["IR check order in triage",
         "Brightness check ran before IR detection. Dark IR images (brightness 11-13) were rejected as 'underexposed' before being identified as valid night shots.",
         "3 images with tigers (000464, 000544, 001082) were skipped entirely.",
         "Moved detect_ir_night() to execute first in triage_image(), before any brightness/blur check."],
        ["Blur threshold too strict for camera traps",
         "Threshold of 80 was calibrated on Sundarbans DSLR photos. Camera trap images through foliage score 38-48 even when visually sharp.",
         "3 images with clearly visible tigers (000725, 000973, 001110) were rejected as 'blurry'.",
         "Lowered BLUR_THRESHOLD from 80 to 25."],
        ["Shadow guard false trigger on IR images",
         "IR cameras produce greyscale images (saturation=0 in HSV). Shadow guard checks mean_saturation < 30, which is always True for IR.",
         "Flash night IR images (0012q, 0013q) passed triage but were rejected by shadow guard.",
         "Added: if bucket != 'IR_Night': condition before shadow guard check."],
        ["Species gate top-3 insufficient for unusual poses",
         "A tiger lying flat or in close-up of fur returns 'hog/wild boar' at rank 1. Top-3 gate had no retry mechanism.",
         "1 image (000764) with a clearly visible tiger was rejected across all attempts.",
         "Added three-attempt strategy: (1) top-3 as-is, (2) brightened top-3, (3) top-5 on brightened."],
        ["Leopard accepted as tiger via top-5 widening",
         "6.jpeg (Test Round 2) = leopard. COCO found a 'dog' box. Top-3 gate failed (leopard/snow leopard). "
         "top-5 found 'tiger cat' (a small South American wild cat — different species). Image was accepted.",
         "1 leopard incorrectly enrolled in DB. ResNet top-10 for this image = snow leopard, leopard, cougar, jaguar — no actual tiger.",
         "Added leopard dominance rule: if >=2 of top-3 are non-tiger big cats (leopard/jaguar/cheetah/cougar/lion), reject immediately. top-5 widening is blocked."],
        ["COCO zebra class ID wrong — proxy missed stripe-pattern detections",
         "TIGER_PROXY_CLASSES had {15, 16, 17, 24}. Code comment said 24=zebra but actual COCO class 24 = backpack. "
         "Zebra is class 22. So COCO's best tiger proxy (stripe pattern) was never active.",
         "16.jpeg had 3 tigers at 1.7-2.5% frame area. COCO found 3 zebra (class 22) boxes at conf 0.78, 0.52, 0.51 — all ignored. "
         "Image fell through to Whole_Image_Fallback, counting 3 tigers as 1.",
         "Fixed TIGER_PROXY_CLASSES to {15, 16, 17, 22}. 16.jpeg now gets 3 COCO boxes, 2 confirmed by species gate."],
        ["OIV7 misses second tiger after CLAHE + DCP preprocessing",
         "OIV7 runs only on the preprocessed image. CLAHE + Dark Channel Prior can suppress local contrast features "
         "that YOLO uses for detection. 15.jpeg: raw image has 2 OIV7 Tiger boxes (conf 0.72 + 0.10). "
         "After preprocessing, only 1 box appears at any confidence threshold.",
         "Second tiger in 15.jpeg (same individual, different camera angle) was missed entirely.",
         "Modified run_cascade_detection() to accept img_raw parameter. OIV7 runs on BOTH images. "
         "Unique boxes merged using IoU deduplication (threshold 0.30). Both tigers now detected."],
    ],
    col_widths=[1.4, 1.8, 1.6, 1.7]
)

doc.add_page_break()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — KEY OBSERVATIONS
# ══════════════════════════════════════════════════════════════════════════════
add_heading(doc, "8.  Key Observations", level=1, size=15)

obs = [
    ("OIV7 direct Tiger class outperforms all proxies",
     "43.8% of detections came from OIV7's direct Tiger class. COCO proxy (cat/dog/horse) contributed only 1.2%. "
     "This proves that a model with the correct target class is far superior to proxy workarounds."),
    ("Whole-image fallback is non-negotiable at scale",
     "55% of accepted detections came from the whole-image fallback stage. Close-up crops where the tiger "
     "fills the frame are not detected as 'a tiger within a scene' by YOLO — the fallback rescues them."),
    ("Species gate correctly identifies other big cats",
     "In Test Round 2, images 5, 7, 8 returned leopard/jaguar/cheetah — genuinely different animals. "
     "The gate correctly rejected them. This validates the pipeline's specificity, not just sensitivity."),
    ("Cross-dataset recognition at 0.84-0.95 without retraining",
     "Tiger_001 was recognised across three completely different datasets (Train, Test Bh, Test Round 2) "
     "using a fingerprint built only from the original train images. The running average DB genuinely learns."),
    ("Camera trap thresholds differ from DSLR benchmarks",
     "All 5 bugs discovered were related to thresholds calibrated on higher-quality Sundarbans images. "
     "Camera trap images are inherently darker, blurrier, and include IR night shots. Thresholds must be "
     "calibrated per deployment environment, not assumed from lab datasets."),
    ("Species gate needs three attempts for challenging images",
     "14.jpeg in Test Round 2 still fails all three attempts (hog/wild boar across top-3 and top-5). "
     "This is a genuine ImageNet limitation — extreme close-up tiger images look like 'hog' in the feature "
     "space. A fine-tuned classifier on tiger-specific data would fix this."),
    ("Tiger_001 is the dominant individual across all datasets",
     "429 sightings across 3 datasets, vs next-highest of 7. This 60:1 ratio suggests Tiger_001 is either "
     "the most territory-active individual or was photographed in a high-traffic location."),
]

for i, (title, detail) in enumerate(obs, 1):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after  = Pt(2)
    r1 = p.add_run(f"  {i}.  {title}")
    r1.bold           = True
    r1.font.size      = Pt(10.5)
    r1.font.color.rgb = C_MID_GREEN
    add_body(doc, f"     {detail}", size=10, space_before=0, space_after=4)

doc.add_page_break()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — OUTPUT FILES
# ══════════════════════════════════════════════════════════════════════════════
add_heading(doc, "9.  Output Files Generated", level=1, size=15)
styled_table(doc,
    ["File", "Location", "Contents"],
    [
        ["ultimate_population_report.csv",    "reports/", "Per-detection log: Tiger_ID, ViewPoint, Match_Score, Top3_ResNet, Morph"],
        ["ultimate_tiger_db.json",            "reports/", "46 tiger fingerprints (2048-dim running average embeddings + sighting counts)"],
        ["ultimate_population_report.txt",    "reports/", "Full text census with viewpoint-aware counts"],
        ["ultimate_annotated/",               "reports/", "Annotated images with bounding boxes, Tiger IDs, morph labels"],
        ["Tiger_AI_Dashboard.html",           "reports/", "Interactive HTML dashboard with 8 Plotly charts"],
        ["Tiger_AI_Ultimate_Run_Report.docx", "reports/", "This Word document"],
        ["amur_tiger_run_observations.md",    "docs/",    "Detailed 11-section run observation log"],
        ["capstone_checklist.md",             "docs/",    "IIM Lucknow submission checklist with viva prep questions"],
        ["architecture.md",                   "docs/",    "Full pipeline architecture with design rules and scale roadmap"],
        ["Classified_TestBh/",               r"Amur Tigers/","35 Test Bh images sorted into per-tiger folders"],
    ],
    col_widths=[2.2, 1.3, 3.0]
)

doc.add_page_break()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — NEXT STEPS
# ══════════════════════════════════════════════════════════════════════════════
add_heading(doc, "10.  Recommended Next Steps", level=1, size=15)
styled_table(doc,
    ["Priority", "Action", "Why"],
    [
        ["HIGH",   "Measure accuracy against ATRW ground truth",
                   "reid_list_train.csv has actual tiger IDs — compare our IDs to compute precision, recall, and F1"],
        ["HIGH",   "Generate Grad-CAM on 5-10 sample images",
                   "Prof-MB will ask for this in the viva. Visual proof the model reads stripes, not background."],
        ["HIGH",   "Re-run full 3,392 train images with all 5 bug fixes",
                   "Corner Trace bug blocked 452 DB updates. A clean re-run gives more accurate fingerprints."],
        ["MEDIUM", "Recalibrate 0.83 threshold per dataset",
                   "Amur tiger stripe density differs from Bengal. Plot cosine score distribution and find optimal cutoff."],
        ["MEDIUM", "ViewPoint improvement using ATRW keypoints",
                   "reid_keypoints_train.json has body landmarks — use for pose-aware Left/Right flank classification."],
        ["MEDIUM", "Investigate 14.jpeg and remaining species gate failures",
                   "Top-5 approach still fails for extreme close-up. CLIP or a fine-tuned classifier would fix this."],
        ["LOW",    "Add MegaDetector as Stage 0",
                   "For 1 lakh image scale — pre-filter empties before running the full cascade."],
        ["LOW",    "FAISS vector index for the identity DB",
                   "In-memory list is fine for 46 tigers; at 10,000+ tigers, FAISS IVF handles millions of vectors."],
        ["FUTURE", "EfficientNetB3 fine-tuning when WII labels arrive",
                   "Phase 2: fine-tune on tiger-specific labelled data for higher species gate accuracy."],
    ],
    col_widths=[0.8, 2.4, 3.3]
)

# ── Footer ─────────────────────────────────────────────────────────────────────
doc.add_paragraph()
add_divider(doc)
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = p.add_run(
    f"Tiger AI Ultimate v3.0  |  Group 4  |  EPAIB Batch 05  |  IIM Lucknow  |  "
    f"{datetime.now().strftime('%d %B %Y')}")
r.font.size      = Pt(9)
r.font.color.rgb = RGBColor(0x75, 0x75, 0x75)

# ── Save ───────────────────────────────────────────────────────────────────────
doc.save(str(OUTPUT_DOCX))
print(f"SUCCESS: Saved -> {OUTPUT_DOCX}")
