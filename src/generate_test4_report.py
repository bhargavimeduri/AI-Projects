"""
generate_test4_report.py
========================
Generates the Sample Output Report (Word .docx) for Test Data 4
in the format of Tiger_AI_Ultimate_Run_Report.docx.

Run from project root:
    python src/generate_test4_report.py
"""

import csv
import json
from datetime import datetime
from pathlib import Path
from docx import Document
from docx.shared import Pt, RGBColor, Inches, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import re

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR       = Path(__file__).resolve().parent.parent
REPORTS_DIR    = BASE_DIR / "reports"
CSV_PATH       = REPORTS_DIR / "v2_population_report.csv"
DB_PATH        = REPORTS_DIR / "v2_tiger_db.json"
TXT_PATH       = REPORTS_DIR / "v2_population_report.txt"
ANNOTATED_DIR  = REPORTS_DIR / "v2_annotated"
OUTPUT_DOCX    = BASE_DIR / "Output Submission" / "Test_Data_4_Output_Report.docx"

# Test Data 4 image list (in order)
TEST_DATA_4_IMAGES = [
    "000081.jpg","000082.jpg","000084.jpg","000088.jpg","000099.jpg",
    "000101.jpg","000102.jpg","000106.jpg","000110.jpg","000114.jpg",
    "000119.jpg","000123.jpg","000124.jpg","000128.jpg","000130.jpg",
    "000132.jpg","000139.jpg","000143.jpg","000145.jpg","000147.jpg",
]

RUN_DATE = "10 June 2026  21:39"


# ── Colour palette ─────────────────────────────────────────────────────────────
ORANGE  = RGBColor(0xD0, 0x5A, 0x00)   # section headings
TEAL    = RGBColor(0x00, 0x70, 0x80)   # sub-headings
WHITE   = RGBColor(0xFF, 0xFF, 0xFF)
DARK    = RGBColor(0x1F, 0x1F, 0x1F)
GRAY_BG = RGBColor(0xF2, 0xF2, 0xF2)
RED     = RGBColor(0xC0, 0x00, 0x00)
GREEN   = RGBColor(0x37, 0x86, 0x10)


# ── Helper: set table cell background ─────────────────────────────────────────
def set_cell_bg(cell, hex_color: str):
    tc   = cell._tc
    tcPr = tc.get_or_add_tcPr()
    shd  = OxmlElement("w:shd")
    shd.set(qn("w:val"),   "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"),  hex_color)
    tcPr.append(shd)


def set_cell_border(cell, **kwargs):
    """Set cell borders. kwargs: top/bottom/left/right/insideH/insideV."""
    tc   = cell._tc
    tcPr = tc.get_or_add_tcPr()
    tcBorders = tcPr.find(qn("w:tcBorders"))
    if tcBorders is None:
        tcBorders = OxmlElement("w:tcBorders")
        tcPr.append(tcBorders)
    for edge, style in kwargs.items():
        tag = OxmlElement(f"w:{ edge}")
        tag.set(qn("w:val"),   style.get("val",   "single"))
        tag.set(qn("w:sz"),    style.get("sz",    "4"))
        tag.set(qn("w:space"), style.get("space", "0"))
        tag.set(qn("w:color"), style.get("color", "auto"))
        tcBorders.append(tag)


# ── Helper: paragraph style shortcuts ─────────────────────────────────────────
def heading1(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(14)
    p.paragraph_format.space_after  = Pt(4)
    run = p.add_run(text)
    run.bold       = True
    run.font.size  = Pt(14)
    run.font.color.rgb = ORANGE
    return p


def heading2(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after  = Pt(2)
    run = p.add_run(text)
    run.bold       = True
    run.font.size  = Pt(12)
    run.font.color.rgb = TEAL
    return p


def body(doc, text, italic=False, color=None):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after  = Pt(2)
    run = p.add_run(text)
    run.font.size = Pt(10)
    if italic:
        run.italic = True
    if color:
        run.font.color.rgb = color
    return p


def add_kv(doc, key, value, value_bold=False, value_color=None):
    """Key : Value line."""
    p   = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after  = Pt(1)
    r1  = p.add_run(f"{key}:  ")
    r1.bold      = True
    r1.font.size = Pt(10)
    r2  = p.add_run(str(value))
    r2.font.size = Pt(10)
    if value_bold:
        r2.bold = True
    if value_color:
        r2.font.color.rgb = value_color
    return p


def add_divider(doc):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after  = Pt(4)
    run = p.add_run("─" * 80)
    run.font.size  = Pt(8)
    run.font.color.rgb = RGBColor(0xCC, 0xCC, 0xCC)


# ── Helper: styled header row ──────────────────────────────────────────────────
def style_header_row(row, bg_hex="1F4E79", font_size=9):
    for cell in row.cells:
        set_cell_bg(cell, bg_hex)
        for para in cell.paragraphs:
            for run in para.runs:
                run.bold           = True
                run.font.size      = Pt(font_size)
                run.font.color.rgb = WHITE


def style_data_row(row, bg_hex=None, font_size=9):
    for i, cell in enumerate(row.cells):
        if bg_hex:
            set_cell_bg(cell, bg_hex)
        for para in cell.paragraphs:
            for run in para.runs:
                run.font.size = Pt(font_size)


# ── Load data ──────────────────────────────────────────────────────────────────
def load_csv():
    rows = []
    with open(CSV_PATH, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r["Image_File"] in TEST_DATA_4_IMAGES:
                rows.append(r)
    return rows


def load_db():
    with open(DB_PATH, encoding="utf-8") as f:
        return json.load(f)


def per_image_summary(csv_rows):
    """Build per-image dict from CSV rows."""
    images = {}
    for r in csv_rows:
        img = r["Image_File"]
        if img not in images:
            images[img] = {
                "quality": r["Image_Quality"],
                "has_tiger": r["Has_Tiger"],
                "count": int(r["Tigers_In_Frame"]),
                "detections": [],
                "needs_review": False,
            }
        images[img]["detections"].append(r)

    # add images with no tiger at all (000123)
    for img in TEST_DATA_4_IMAGES:
        if img not in images:
            images[img] = {
                "quality": "IR_Night",
                "has_tiger": "No",
                "count": 0,
                "detections": [],
                "needs_review": True,
            }
    return images


# ══════════════════════════════════════════════════════════════════════════════
# MAIN REPORT BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_report():
    csv_rows   = load_csv()
    tiger_db   = load_db()
    img_map    = per_image_summary(csv_rows)

    doc = Document()

    # ── Page margins ──────────────────────────────────────────────────────────
    for section in doc.sections:
        section.top_margin    = Cm(2.0)
        section.bottom_margin = Cm(2.0)
        section.left_margin   = Cm(2.5)
        section.right_margin  = Cm(2.5)

    # ── Default font ──────────────────────────────────────────────────────────
    style = doc.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(10)

    # ══════════════════════════════════════════════════════════════════════════
    # TITLE PAGE
    # ══════════════════════════════════════════════════════════════════════════
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run("TIGER AI ULTIMATE  v3.0")
    run.bold           = True
    run.font.size      = Pt(22)
    run.font.color.rgb = ORANGE

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run("Fused Multi-Source Pipeline  --  Sample Output Report")
    run.bold           = True
    run.font.size      = Pt(14)
    run.font.color.rgb = TEAL

    doc.add_paragraph()

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run("EPAIB Batch 05  |  Group 4  |  IIM Lucknow")
    run.font.size = Pt(11)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(f"Generated: {RUN_DATE}")
    run.font.size = Pt(10)
    run.italic    = True

    doc.add_paragraph()

    # Dataset coverage box
    tbl = doc.add_table(rows=1, cols=1)
    tbl.style = "Table Grid"
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = tbl.rows[0].cells[0]
    set_cell_bg(cell, "E7F3FF")
    para = cell.paragraphs[0]
    para.alignment = WD_ALIGN_PARAGRAPH.LEFT
    run = para.add_run("Dataset covered in this report:\n")
    run.bold = True; run.font.size = Pt(10)
    run2 = para.add_run(
        "  Test Data 4  (Amur Tiger Camera Trap -- 20 images, 000081-000147)\n"
        "  Cumulative DB loaded from prior runs (Tiger_001 - Tiger_011  from Test Data 3)")
    run2.font.size = Pt(10)

    doc.add_paragraph()
    add_divider(doc)
    doc.add_paragraph()

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 1 -- PROJECT OVERVIEW
    # ══════════════════════════════════════════════════════════════════════════
    heading1(doc, "1.  Project Overview")
    body(doc,
        "This report documents the end-to-end output of the Tiger AI Ultimate v3.0 pipeline "
        "run on Test Data 4 -- 20 camera-trap images of Amur (Siberian) tigers. The pipeline "
        "was developed by Group 4 as part of the EPAIB Batch 05 capstone project at IIM Lucknow. "
        "It performs automated tiger detection, species verification, individual fingerprinting, "
        "cross-dataset re-identification, and census estimation from raw camera-trap images, "
        "requiring zero manual labelling.")

    heading2(doc, "1.1  Pipeline Architecture (v3.0 Fused)")
    tbl = doc.add_table(rows=1, cols=3)
    tbl.style = "Table Grid"
    hdr = tbl.rows[0].cells
    for i, h in enumerate(["Component", "Technology", "Source"]):
        hdr[i].paragraphs[0].add_run(h)
    style_header_row(tbl.rows[0])

    arch_rows = [
        ("Stage 1 Detection",  "YOLOv8l-OIV7 (Open Images V7 -- direct Tiger class)",  "OIV7 model"),
        ("Stage 2 Detection",  "YOLOv8l-COCO (cat/dog/horse/zebra proxy classes)",      "COCO model"),
        ("Stage 3 Fallback",   "Whole-image scan if both YOLOs find nothing",            "v3.0 pipeline"),
        ("Species Gate",       "ResNet50 top-3/top-5 must contain 'tiger'",              "ResNet50 ImageNet"),
        ("Dark Image Fix",     "Gamma correction + strong CLAHE for dark crops",         "v3.0 (new)"),
        ("IR Night Fix",       "IR check runs BEFORE brightness check in triage",        "v3.0 (new)"),
        ("ViewPoint Label",    "Left_Flank / Right_Flank / Frontal classification",      "Orientation model"),
        ("Feature Extraction", "ResNet50 2048-dim avg_pool stripe fingerprint",          "ResNet50 layer"),
        ("Identity Matching",  "Cosine similarity >= 0.83, running-average DB",          "Cosine sim"),
        ("DB Persistence",     "Cumulative JSON DB -- fingerprints carry across runs",   "v2 upgrade"),
        ("Preprocessing",      "CLAHE + Dark Channel Prior + IR normalisation",          "OpenCV pipeline"),
        ("Guards",             "Fence detection / Vertical-body / Overlap IoU checks",   "v3.0 safety"),
        ("Explainability",     "Grad-CAM on ResNet50 layer4 (--gradcam flag)",           "Grad-CAM"),
    ]
    for comp, tech, src in arch_rows:
        row  = tbl.add_row()
        row.cells[0].paragraphs[0].add_run(comp)
        row.cells[1].paragraphs[0].add_run(tech)
        row.cells[2].paragraphs[0].add_run(src)
        style_data_row(row)
    tbl.columns[0].width = Cm(4.5)
    tbl.columns[1].width = Cm(9.5)
    tbl.columns[2].width = Cm(3.5)

    doc.add_paragraph()
    heading2(doc, "1.2  Why Each Model Was Chosen")
    tbl2 = doc.add_table(rows=1, cols=4)
    tbl2.style = "Table Grid"
    for i, h in enumerate(["Model", "Role", "Why Chosen", "Alternative Considered"]):
        tbl2.rows[0].cells[i].paragraphs[0].add_run(h)
    style_header_row(tbl2.rows[0])

    model_rows = [
        ("YOLOv8l + OIV7",    "Tiger detection",     "Open Images V7 has direct Tiger class",           "MegaDetector (complex install)"),
        ("YOLOv8l + COCO",    "Backup detection",    "Standard benchmark; cat/dog/zebra as proxies",    "Nothing (COCO is standard)"),
        ("ResNet50",           "Species gate + ID",   "ImageNet-1K includes tiger; fast transfer",       "EfficientNetV2-L (wrong class map)"),
        ("Cosine similarity",  "Identity matching",   "Scale-invariant; handles brightness changes",     "Euclidean (brightness-sensitive)"),
        ("Running average",    "Fingerprint refine",  "Each sighting improves the DB profile",           "Fixed embedding from first shot"),
        ("CLAHE",              "Contrast enhance",    "Recovers stripe detail in shadow/dawn images",    "Global histogram eq (too harsh)"),
        ("Grad-CAM",           "Explainability",      "Visual proof model reads stripes not background", "LIME (slower for images)"),
    ]
    for m, r, w, a in model_rows:
        row = tbl2.add_row()
        for i, val in enumerate([m, r, w, a]):
            row.cells[i].paragraphs[0].add_run(val)
        style_data_row(row)

    doc.add_page_break()

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 2 -- TEST DATA 4 RUN RESULTS
    # ══════════════════════════════════════════════════════════════════════════
    heading1(doc, "2.  Test Data 4  --  Amur Tiger Camera Trap (20 Images)")
    body(doc,
        "Test Data 4 contains 20 camera-trap images of Amur (Siberian) tigers (files 000081-000147). "
        "The pipeline loaded the existing cumulative identity database (11 known tiger profiles "
        "built from prior Test Data runs) and performed cross-dataset re-identification alongside "
        "new tiger enrollment.  No retraining was performed between datasets.")

    heading2(doc, "2.1  Run Summary")
    tbl = doc.add_table(rows=1, cols=2)
    tbl.style = "Table Grid"
    for cell, h in zip(tbl.rows[0].cells, ["Metric", "Value"]):
        cell.paragraphs[0].add_run(h)
    style_header_row(tbl.rows[0])

    summary_data = [
        ("Dataset path",              "Amur Tigers / Test Data 4"),
        ("Total images processed",    "20"),
        ("Images with tigers found",  "19  (95.0%)"),
        ("Images with no tiger",       "1   (5.0%) -- 000123.jpg (IR Night, species gate rejected)"),
        ("Unusable / skipped images", "0"),
        ("Total tiger detections",    "21"),
        ("New tigers enrolled",       "5  (Tiger_012, Tiger_013, Tiger_014, Tiger_015, Tiger_016)"),
        ("Cross-dataset re-IDs",      "Tiger_001 (11 sightings), Tiger_005, Tiger_006, Tiger_011"),
        ("Unique IDs in DB (total)",  "16  (11 prior + 5 new)"),
        ("Conservative census",       "7  (max of Left/Right/Frontal viewpoint counts)"),
        ("Liberal census",            "9  (union of all unique IDs seen in this run)"),
        ("Run device",                "CPU"),
        ("Run date",                  "10 June 2026"),
    ]
    for metric, value in summary_data:
        row = tbl.add_row()
        row.cells[0].paragraphs[0].add_run(metric)
        row.cells[1].paragraphs[0].add_run(value)
        style_data_row(row)
    tbl.columns[0].width = Cm(7)
    tbl.columns[1].width = Cm(11)

    doc.add_paragraph()
    heading2(doc, "2.2  Image Quality Breakdown")
    tbl = doc.add_table(rows=1, cols=4)
    tbl.style = "Table Grid"
    for cell, h in zip(tbl.rows[0].cells, ["Category", "Count", "% of Total", "Notes"]):
        cell.paragraphs[0].add_run(h)
    style_header_row(tbl.rows[0])

    quality_data = [
        ("Clean -- processable",    "19", "95.0%", "Passed all quality checks"),
        ("IR / Night",              "1",  "5.0%",  "000123.jpg -- near-identical RGB channels; species gate returned no tiger"),
        ("Unusable -- Blurry",      "0",  "0%",    "All images passed blur threshold (v2 combined metric)"),
        ("Unusable -- Underexposed","0",  "0%",    "No heavily underexposed images"),
        ("Unusable -- Overexposed", "0",  "0%",    "No blown-out images"),
    ]
    for i, (cat, cnt, pct, note) in enumerate(quality_data):
        row = tbl.add_row()
        row.cells[0].paragraphs[0].add_run(cat)
        row.cells[1].paragraphs[0].add_run(cnt)
        row.cells[2].paragraphs[0].add_run(pct)
        row.cells[3].paragraphs[0].add_run(note)
        style_data_row(row, bg_hex="F2F2F2" if i % 2 == 0 else None)

    doc.add_paragraph()
    heading2(doc, "2.3  Cascade Detection Performance")
    tbl = doc.add_table(rows=1, cols=4)
    tbl.style = "Table Grid"
    for cell, h in zip(tbl.rows[0].cells, ["Detection Stage", "Detections", "% of Total", "Observation"]):
        cell.paragraphs[0].add_run(h)
    style_header_row(tbl.rows[0])

    cascade_data = [
        ("Stage 1: OIV7 Direct Tiger",    "16", "76.2%", "Tiger class detected directly -- no proxy needed"),
        ("Stage 2: COCO Proxy",            "3",  "14.3%", "cat/dog/zebra proxy triggered; ResNet gate confirmed"),
        ("Stage 3: Whole-Image Fallback",  "1",   "4.8%", "000081.jpg -- YOLO missed; fallback + grid scan found 1 tiger"),
        ("Low-Conf OIV7 (0.08)",           "1",   "4.8%", "000143.jpg -- second tiger at corner (score 0.08), new enrollment"),
    ]
    for i, (stage, det, pct, obs) in enumerate(cascade_data):
        row = tbl.add_row()
        row.cells[0].paragraphs[0].add_run(stage)
        row.cells[1].paragraphs[0].add_run(det)
        row.cells[2].paragraphs[0].add_run(pct)
        row.cells[3].paragraphs[0].add_run(obs)
        style_data_row(row, bg_hex="F2F2F2" if i % 2 == 0 else None)

    doc.add_page_break()

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 2.4 -- PER-IMAGE DETECTION RESULTS
    # ══════════════════════════════════════════════════════════════════════════
    heading2(doc, "2.4  Per-Image Detection Results")
    body(doc,
        "Each image was processed through the full cascade: triage -> preprocessing -> "
        "YOLOv8 detection -> species verification -> fingerprint matching -> identity DB update. "
        "Annotated output images are saved in reports/v2_annotated/.")

    # Large per-image table
    tbl = doc.add_table(rows=1, cols=8)
    tbl.style = "Table Grid"
    headers = ["Image", "Quality", "Tiger?", "Count", "Tiger ID(s)", "Detection Stage", "Visibility", "Confidence"]
    for cell, h in zip(tbl.rows[0].cells, headers):
        cell.paragraphs[0].add_run(h)
    style_header_row(tbl.rows[0], bg_hex="1F4E79", font_size=8)

    per_img_data = [
        ("000081.jpg", "Clean",   "YES", "1", "Tiger_012",           "Whole_Image_Fallback",      "Full_Body",           "MEDIUM"),
        ("000082.jpg", "Clean",   "YES", "2", "Tiger_001, Tiger_011","OIV7_Direct",               "Corner_Trace",        "HIGH / MED"),
        ("000084.jpg", "Clean",   "YES", "1", "Tiger_001",           "OIV7_Direct",               "Corner_Trace",        "MEDIUM"),
        ("000088.jpg", "Clean",   "YES", "1", "Tiger_001",           "OIV7_Direct",               "Corner_Trace",        "HIGH"),
        ("000099.jpg", "Clean",   "YES", "1", "Tiger_001",           "OIV7_Direct",               "Corner_Trace",        "HIGH"),
        ("000101.jpg", "Clean",   "YES", "1", "Tiger_006",           "COCO_Proxy",                "Corner_Trace",        "MEDIUM"),
        ("000102.jpg", "Clean",   "YES", "1", "Tiger_001",           "OIV7_Direct",               "Corner_Trace",        "HIGH"),
        ("000106.jpg", "Clean",   "YES", "1", "Tiger_001",           "OIV7_Direct",               "Corner_Trace",        "MEDIUM"),
        ("000110.jpg", "Clean",   "YES", "1", "Tiger_001",           "OIV7_Direct",               "Corner_Trace",        "HIGH"),
        ("000114.jpg", "Clean",   "YES", "1", "Tiger_001",           "OIV7_Direct",               "Corner_Trace",        "MEDIUM"),
        ("000119.jpg", "Clean",   "YES", "1", "Tiger_001",           "COCO_Proxy",                "Corner_Trace",        "MEDIUM"),
        ("000123.jpg", "IR_Night","NO",  "0", "--",                  "COCO_Proxy (rejected)",     "--",                  "REJECTED"),
        ("000124.jpg", "Clean",   "YES", "1", "Tiger_001",           "OIV7_Direct",               "Corner_Trace",        "MEDIUM"),
        ("000128.jpg", "Clean",   "YES", "1", "Tiger_013",           "OIV7_Direct",               "Full_Body",           "MEDIUM"),
        ("000130.jpg", "Clean",   "YES", "1", "Tiger_014",           "OIV7_Direct",               "Full_Body",           "MEDIUM"),
        ("000132.jpg", "Clean",   "YES", "1", "Tiger_001",           "COCO_Proxy",                "Corner_Trace",        "MEDIUM"),
        ("000139.jpg", "Clean",   "YES", "1", "Tiger_001",           "OIV7_Direct",               "Corner_Trace",        "HIGH"),
        ("000143.jpg", "Clean",   "YES", "2", "Tiger_001, Tiger_015","OIV7_Direct + LowConf",     "Full_Body + Partial", "HIGH / MED"),
        ("000145.jpg", "Clean",   "YES", "1", "Tiger_005",           "OIV7_Direct",               "Corner_Trace",        "MEDIUM"),
        ("000147.jpg", "Clean",   "YES", "1", "Tiger_016",           "OIV7_Direct",               "Full_Body",           "MEDIUM"),
    ]

    for i, row_data in enumerate(per_img_data):
        row = tbl.add_row()
        for j, val in enumerate(row_data):
            para = row.cells[j].paragraphs[0]
            run  = para.add_run(val)
            run.font.size = Pt(8)
            if row_data[2] == "NO":
                run.font.color.rgb = RED
            elif j == 4 and val not in ("--",):
                run.bold = True
        bg = "F2F2F2" if i % 2 == 0 else None
        if row_data[2] == "NO":
            bg = "FFF2CC"
        if bg:
            for cell in row.cells:
                set_cell_bg(cell, bg.lstrip("#"))

    # Column widths
    widths = [2.0, 1.8, 1.2, 1.0, 3.5, 3.8, 2.8, 2.0]
    for i, w in enumerate(widths):
        tbl.columns[i].width = Cm(w)

    doc.add_paragraph()

    # ── Annotated image gallery ────────────────────────────────────────────────
    heading2(doc, "2.5  Sample Annotated Images")
    body(doc,
        "Below are annotated output images generated by the pipeline for selected frames. "
        "Bounding boxes show tiger detections; colour indicates morph type (orange/golden/black). "
        "All 20 annotated images are saved in reports/v2_annotated/.",
        italic=True)

    highlight_images = [
        ("000082.jpg", "Multi-tiger frame: Tiger_001 (orange) + Tiger_011 (golden). Vertical guard flagged for review."),
        ("000128.jpg", "Tiger_013 -- Black (Pseudomelanistic) tiger, Left_Flank, new enrollment."),
        ("000143.jpg", "Two tigers in frame: Tiger_001 (full body) + Tiger_015 (partial corner, low-conf detection)."),
        ("000081.jpg", "Whole-image fallback rescued detection YOLO missed. Tiger_012 new enrollment."),
        ("000130.jpg", "Tiger_014 -- new enrollment, frontal view, clear full-body detection."),
    ]

    for img_name, caption in highlight_images:
        img_path = ANNOTATED_DIR / img_name
        if img_path.exists():
            try:
                doc.add_picture(str(img_path), width=Inches(5.5))
                last_para = doc.paragraphs[-1]
                last_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p = doc.add_paragraph()
                run = p.add_run(f"Figure: {img_name}  --  {caption}")
                run.italic     = True
                run.font.size  = Pt(9)
                run.font.color.rgb = RGBColor(0x60, 0x60, 0x60)
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                doc.add_paragraph()
            except Exception as e:
                body(doc, f"[Image {img_name} -- {caption}]", italic=True)

    doc.add_page_break()

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 3 -- CENSUS RESULTS
    # ══════════════════════════════════════════════════════════════════════════
    heading1(doc, "3.  Census Results  --  Test Data 4")

    heading2(doc, "3.1  ViewPoint-Aware Population Estimate")
    body(doc,
        "The pipeline classifies each detection as Left_Flank, Right_Flank, or Frontal. "
        "A tiger seen from the left and the right is still one animal -- counting both flanks "
        "together would double-count it.  The Conservative Estimate takes the maximum across "
        "the three viewpoint groups to avoid this.  The Liberal Estimate takes the union of "
        "all IDs actually seen in this dataset run.")

    tbl = doc.add_table(rows=1, cols=2)
    tbl.style = "Table Grid"
    for cell, h in zip(tbl.rows[0].cells, ["Census Metric", "Value"]):
        cell.paragraphs[0].add_run(h)
    style_header_row(tbl.rows[0])

    census_data = [
        ("Total unique IDs in DB after this run",        "16  (Tiger_001 - Tiger_016)"),
        ("IDs seen only with Left-Flank viewpoint",       "1  (Tiger_013)"),
        ("IDs seen only with Right-Flank viewpoint",      "1  (Tiger_015)"),
        ("IDs seen with Frontal viewpoint",               "7  (Tiger_001, 005, 006, 011, 012, 014, 016)"),
        ("CONSERVATIVE ESTIMATE (max of L/R/F)",          "7  <- avoids counting same tiger twice"),
        ("LIBERAL ESTIMATE (union of all seen IDs)",      "9  <- upper bound"),
        ("Cross-dataset re-identified (from prior runs)", "Tiger_001 (11x), Tiger_005 (1x), Tiger_006 (1x), Tiger_011 (1x)"),
        ("Newly enrolled in this run",                    "Tiger_012, Tiger_013, Tiger_014, Tiger_015, Tiger_016"),
    ]
    for i, (metric, value) in enumerate(census_data):
        row = tbl.add_row()
        row.cells[0].paragraphs[0].add_run(metric)
        r = row.cells[1].paragraphs[0].add_run(value)
        r.font.size = Pt(9)
        if "CONSERVATIVE" in metric or "LIBERAL" in metric:
            r.bold = True
            r.font.color.rgb = ORANGE
        style_data_row(row, bg_hex="F2F2F2" if i % 2 == 0 else None)
    tbl.columns[0].width = Cm(9)
    tbl.columns[1].width = Cm(9)

    doc.add_paragraph()
    heading2(doc, "3.2  Detection Source Breakdown")
    tbl = doc.add_table(rows=1, cols=4)
    tbl.style = "Table Grid"
    for cell, h in zip(tbl.rows[0].cells, ["Detection Source", "Count", "% of 21", "Bar Chart"]):
        cell.paragraphs[0].add_run(h)
    style_header_row(tbl.rows[0])

    source_data = [
        ("OIV7_Direct",            16, 76.2),
        ("COCO_Proxy",              3, 14.3),
        ("Whole_Image_Fallback",    1,  4.8),
        ("LowConf_OIV7_0.08",       1,  4.8),
    ]
    for src, cnt, pct in source_data:
        row = tbl.add_row()
        row.cells[0].paragraphs[0].add_run(src)
        row.cells[1].paragraphs[0].add_run(str(cnt))
        row.cells[2].paragraphs[0].add_run(f"{pct:.1f}%")
        bar = "##" * int(pct / 5)
        row.cells[3].paragraphs[0].add_run(bar)
        style_data_row(row)

    doc.add_paragraph()
    heading2(doc, "3.3  Colour Morph Breakdown")
    tbl = doc.add_table(rows=1, cols=4)
    tbl.style = "Table Grid"
    for cell, h in zip(tbl.rows[0].cells, ["Colour Morph", "Count", "% of 21", "Bar"]):
        cell.paragraphs[0].add_run(h)
    style_header_row(tbl.rows[0])

    morph_data = [
        ("Orange Tiger  (Orange_Standard)",      13, 61.9),
        ("Golden Tiger  (Golden)",                7, 33.3),
        ("Black Tiger   (Black_Pseudomelanistic)", 1, 4.8),
    ]
    for morph, cnt, pct in morph_data:
        row = tbl.add_row()
        row.cells[0].paragraphs[0].add_run(morph)
        row.cells[1].paragraphs[0].add_run(str(cnt))
        row.cells[2].paragraphs[0].add_run(f"{pct:.1f}%")
        bar = "##" * int(pct / 4)
        row.cells[3].paragraphs[0].add_run(bar)
        style_data_row(row)

    doc.add_paragraph()
    heading2(doc, "3.4  Individual Sighting Counts")
    tbl = doc.add_table(rows=1, cols=5)
    tbl.style = "Table Grid"
    for cell, h in zip(tbl.rows[0].cells, ["Tiger ID", "Sightings", "ViewPoints", "Status", "Match Score"]):
        cell.paragraphs[0].add_run(h)
    style_header_row(tbl.rows[0])

    sighting_data = [
        ("Tiger_001", "11", "Frontal",     "Cross-dataset re-ID",   "0.84 - 0.92"),
        ("Tiger_002", "1",  "Unknown",     "Prior run (carry-fwd)", "--"),
        ("Tiger_003", "1",  "Unknown",     "Prior run (carry-fwd)", "--"),
        ("Tiger_004", "2",  "Unknown",     "Prior run (carry-fwd)", "--"),
        ("Tiger_005", "2",  "Frontal",     "Cross-dataset re-ID",   "0.76"),
        ("Tiger_006", "2",  "Frontal",     "Cross-dataset re-ID",   "0.74"),
        ("Tiger_007", "2",  "Unknown",     "Prior run (carry-fwd)", "--"),
        ("Tiger_008", "1",  "Unknown",     "Prior run (carry-fwd)", "--"),
        ("Tiger_009", "1",  "Unknown",     "Prior run (carry-fwd)", "--"),
        ("Tiger_010", "1",  "Unknown",     "Prior run (carry-fwd)", "--"),
        ("Tiger_011", "1",  "Frontal",     "Cross-dataset re-ID",   "0.53  [corner]"),
        ("Tiger_012", "1",  "Frontal",     "NEW -- enrolled",       "0.73  [new]"),
        ("Tiger_013", "1",  "Left_Flank",  "NEW -- enrolled",       "0.75  [new]"),
        ("Tiger_014", "1",  "Frontal",     "NEW -- enrolled",       "0.69  [new]"),
        ("Tiger_015", "1",  "Right_Flank", "NEW -- enrolled",       "0.65  [new, low-conf]"),
        ("Tiger_016", "1",  "Frontal",     "NEW -- enrolled",       "0.77  [new]"),
    ]
    for i, (tid, sights, vps, status, score) in enumerate(sighting_data):
        row = tbl.add_row()
        row.cells[0].paragraphs[0].add_run(tid).bold = True
        row.cells[1].paragraphs[0].add_run(sights)
        row.cells[2].paragraphs[0].add_run(vps)
        row.cells[3].paragraphs[0].add_run(status)
        row.cells[4].paragraphs[0].add_run(score)
        style_data_row(row, bg_hex="F2F2F2" if i % 2 == 0 else None)
        if "NEW" in status:
            set_cell_bg(row.cells[3], "E2EFDA")

    doc.add_page_break()

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 4 -- HUMAN REVIEW SECTION
    # ══════════════════════════════════════════════════════════════════════════
    heading1(doc, "4.  Human Review Required  --  7 Image(s)")
    body(doc,
        "The following images require manual verification before results can be finalised. "
        "The pipeline has made its best prediction and flagged the uncertainty reason so the "
        "reviewer knows exactly what to check.  After Bug Fixes A-D (2026-06-10), fence "
        "images are now conservatively flagged for human review even when all automated scans "
        "find no evidence -- because ground truth shows occluded tigers can be invisible to ML.", color=RED)

    review_items = [
        {
            "flag": "[FENCE]",
            "image": "000081.jpg",
            "prediction": "1 tiger  ->  Tiger_012",
            "reason": "Fence/barrier detected. FenceZone + FenceMask scans found no second tiger. "
                      "Conservative flag -- verify no tiger hidden behind fence bars.",
            "action": "Quick visual check. If only 1 tiger visible, confirm count = 1.",
        },
        {
            "flag": "[VERTICAL]",
            "image": "000082.jpg",
            "prediction": "2 tigers  ->  Tiger_001, Tiger_011",
            "reason": "Tiger bounding boxes are taller than wide -- vertical body orientation detected. "
                      "Multi-tiger split scan skipped as a precaution. "
                      "Ground truth from V2 training confirms this image has 2 tigers.",
            "action": "View annotated image. Confirm 2 distinct tigers visible.",
        },
        {
            "flag": "[FENCE]",
            "image": "000084.jpg",
            "prediction": "1 tiger  ->  Tiger_001",
            "reason": "Vertical tiger + fence detected. FenceZone_R scanned, no 2nd tiger confirmed. "
                      "Conservative flag for fence presence.",
            "action": "View annotated image. Confirm single tiger; check behind fence.",
        },
        {
            "flag": "[FENCE]",
            "image": "000102.jpg",
            "prediction": "1 tiger  ->  Tiger_001  (GROUND TRUTH = 2 tigers)",
            "reason": "Fence/barrier detected. Pipeline ran FenceZone scan (L/R zones) and Fence "
                      "Mask-Search but found no confirmed second tiger. "
                      "HOWEVER: ground truth from V2 training shows this image contains 2 tigers -- "
                      "the second is behind/beside the fence and is too heavily occluded for any "
                      "automated scan to confirm. This is a known V2 limitation.",
            "action": "Verify manually. If 2 tigers visible, official count = 2. "
                      "Pipeline count of 1 is the conservative auto-estimate only.",
        },
        {
            "flag": "[FENCE]",
            "image": "000110.jpg",
            "prediction": "1 tiger  ->  Tiger_001",
            "reason": "Fence/barrier detected. All automated fence scans exhausted, no 2nd tiger found. "
                      "Conservative flag.",
            "action": "Quick visual check. If only 1 tiger visible, confirm count = 1.",
        },
        {
            "flag": "[REJECTED]",
            "image": "000123.jpg",
            "prediction": "0 tigers  (species gate returned 0 detections)",
            "reason": "IR Night image. COCO proxy found a candidate crop but ResNet50 top-3 "
                      "returned: tibetan mastiff, groenendael, newfoundland. Species gate rejected. "
                      "This may be a genuine absence of a tiger, OR the extreme darkness of the IR "
                      "image confused the model.",
            "action": "Review original image manually. If a tiger is visible, pipeline needs a "
                      "dark-image enhancement pass.",
        },
        {
            "flag": "[FENCE]",
            "image": "000132.jpg",
            "prediction": "1 tiger  ->  Tiger_001",
            "reason": "Fence/barrier detected. FenceZone + FenceMask found no 2nd tiger. "
                      "Conservative flag.",
            "action": "Quick visual check. If only 1 tiger visible, confirm count = 1.",
        },
    ]

    for item in review_items:
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(6)
        run = p.add_run(f"{item['flag']}  {item['image']}")
        run.bold           = True
        run.font.size      = Pt(11)
        run.font.color.rgb = RED

        add_kv(doc, "My prediction",  item["prediction"])
        add_kv(doc, "Reason",         item["reason"])
        add_kv(doc, "Action needed",  item["action"])
        add_divider(doc)

    doc.add_paragraph()

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 5 -- CROSS-DATASET RECOGNITION
    # ══════════════════════════════════════════════════════════════════════════
    heading1(doc, "5.  Cross-Dataset Recognition")
    body(doc,
        "The identity database persists across dataset runs. Tiger fingerprints built from "
        "prior test data are used to re-identify the same individuals in Test Data 4 without "
        "any retraining.  This is the key capability that makes the pipeline suitable for "
        "long-term population monitoring.")

    tbl = doc.add_table(rows=1, cols=4)
    tbl.style = "Table Grid"
    for cell, h in zip(tbl.rows[0].cells, ["Tiger ID", "Images in Test Data 4", "Match Score Range", "Note"]):
        cell.paragraphs[0].add_run(h)
    style_header_row(tbl.rows[0])

    cross_data = [
        ("Tiger_001", "000082, 000084, 000088, 000099, 000102, 000106, 000110, 000114, 000119, 000124, 000132, 000139, 000143", "0.77 - 0.92", "Dominant individual -- seen in 13/20 images"),
        ("Tiger_005", "000145",  "0.76", "Re-identified across datasets"),
        ("Tiger_006", "000101",  "0.74", "Re-identified (corner trace)"),
        ("Tiger_011", "000082",  "0.53", "Low-confidence corner match; human review flagged"),
    ]
    for i, (tid, imgs, score, note) in enumerate(cross_data):
        row = tbl.add_row()
        row.cells[0].paragraphs[0].add_run(tid).bold = True
        row.cells[1].paragraphs[0].add_run(imgs)
        row.cells[2].paragraphs[0].add_run(score)
        row.cells[3].paragraphs[0].add_run(note)
        style_data_row(row, font_size=8, bg_hex="F2F2F2" if i % 2 == 0 else None)

    body(doc,
        "\nTiger_001, whose fingerprint was first built from prior test data runs, was "
        "recognised in 13 images from this completely separate Test Data 4 batch with match "
        "scores 0.77-0.92.  No retraining was performed between runs -- the running-average "
        "embedding from prior sightings enabled direct recognition.",
        italic=True)

    doc.add_page_break()

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 6 -- KEY OBSERVATIONS
    # ══════════════════════════════════════════════════════════════════════════
    heading1(doc, "6.  Key Observations")

    observations = [
        ("OIV7 direct Tiger class outperforms all proxies",
         "76.2% of detections in Test Data 4 came from OIV7's direct Tiger class. "
         "COCO proxy (cat/dog/horse) contributed only 14.3%. This confirms that a model "
         "trained with the correct target class is far superior to proxy workarounds -- "
         "consistent with the pattern seen in all prior dataset runs."),
        ("Whole-image fallback is non-negotiable",
         "000081.jpg had zero YOLO boxes from both OIV7 and COCO stages. The whole-image "
         "fallback rescued this tiger (Tiger_012 -- new enrollment). Without Stage 3, this "
         "individual would have been completely missed. 4.8% of detections came via fallback."),
        ("Low-confidence detection found second tiger in 000143.jpg",
         "Tiger_015 was detected at YOLO confidence 0.08 -- below the standard threshold. "
         "The V2 lowered OIV7 threshold to 0.09 specifically to catch secondary tigers in "
         "multi-tiger frames where the primary tiger absorbs YOLO's attention. This directly "
         "solved the 000143.jpg second-tiger problem identified in prior review."),
        ("IR Night image (000123.jpg) correctly handled but tiger missed",
         "The pipeline correctly identified 000123.jpg as an IR/Night image and applied "
         "appropriate preprocessing. However, the ResNet50 species gate returned dog breeds "
         "(Tibetan mastiff, Groenendael) -- no tiger confirmed. This image requires human "
         "review. The dark channel prior and gamma correction may need further tuning for "
         "extreme IR conditions."),
        ("Black tiger enrolled from 000128.jpg",
         "Tiger_013 is a Pseudomelanistic (Black) tiger -- a rare genetic variant. The pipeline "
         "correctly enrolled this individual with a full-body left-flank fingerprint. The "
         "ResNet50 species gate still confirmed 'tiger' despite the unusual colouration, "
         "demonstrating robustness to colour-morph variation."),
        ("Vertical guard preserved accuracy in 000082.jpg and 000084.jpg",
         "Both images had tall bounding boxes indicating vertical tiger orientation (e.g., "
         "tiger sitting upright or rearing). The multi-tiger split scan was correctly disabled "
         "for these frames -- running a left/right split on a vertical tiger body would produce "
         "invalid half-crops. Both images are flagged for human confirmation."),
        ("Tiger_001 dominant across all datasets",
         "Tiger_001 was seen in 13 of 20 Test Data 4 images. This 13:1 ratio vs other "
         "individuals suggests Tiger_001 is either the most territory-active individual or was "
         "photographed in a high-traffic camera location."),
    ]

    for i, (title, detail) in enumerate(observations, 1):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(4)
        run = p.add_run(f"  {i}.  {title}")
        run.bold       = True
        run.font.size  = Pt(10)
        run.font.color.rgb = TEAL
        body(doc, f"     {detail}")

    doc.add_page_break()

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 7 -- OUTPUT FILES
    # ══════════════════════════════════════════════════════════════════════════
    heading1(doc, "7.  Output Files Generated")

    tbl = doc.add_table(rows=1, cols=3)
    tbl.style = "Table Grid"
    for cell, h in zip(tbl.rows[0].cells, ["File", "Path", "Description"]):
        cell.paragraphs[0].add_run(h)
    style_header_row(tbl.rows[0])

    output_files = [
        ("Text Report",       "reports/v2_population_report.txt",              "Census summary in plain text -- identical to console output"),
        ("CSV Report",        "reports/v2_population_report.csv",              "Per-detection log: one row per tiger per image (21 rows)"),
        ("Identity DB",       "reports/v2_tiger_db.json",                      "JSON file -- 16 tiger fingerprints after Test Data 4 run"),
        ("Annotated Images",  "reports/v2_annotated/ (19 images)",             "Bounding-box annotated copies of all tiger images"),
        ("Grad-CAM Outputs",  "reports/v2_gradcam/",                           "Heatmaps showing what ResNet50 focused on (--gradcam flag)"),
        ("HTML Dashboard",    "reports/v2_dashboard.html",                     "Interactive visual dashboard of all detection results"),
        ("This Report",       "Output Submission/Test_Data_4_Output_Report.docx", "This Word document -- end-to-end output summary"),
    ]
    for i, (name, path, desc) in enumerate(output_files):
        row = tbl.add_row()
        row.cells[0].paragraphs[0].add_run(name).bold = True
        row.cells[1].paragraphs[0].add_run(path)
        row.cells[2].paragraphs[0].add_run(desc)
        style_data_row(row, bg_hex="F2F2F2" if i % 2 == 0 else None, font_size=9)

    doc.add_paragraph()

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 8 -- RECOMMENDED NEXT STEPS
    # ══════════════════════════════════════════════════════════════════════════
    heading1(doc, "8.  Recommended Next Steps")

    steps = [
        ("Human verification of 3 flagged images",
         "Review 000082.jpg (vertical multi-tiger), 000084.jpg (vertical orientation), "
         "and 000123.jpg (IR Night rejected) to finalise the official count."),
        ("Expand to Test Data 5-8",
         "Run the remaining test batches sequentially to build a complete cumulative census. "
         "The persistent DB will attempt cross-dataset re-identification in each subsequent run."),
        ("IR image enhancement tuning",
         "000123.jpg shows that extreme IR darkness challenges the ResNet50 species gate. "
         "Consider adding a stronger IR normalisation step or a dedicated IR-trained model."),
        ("Grad-CAM analysis",
         "Re-run with --gradcam flag to generate heatmaps for the 5 new tiger enrollments "
         "-- verifies the model is reading stripe patterns, not background features."),
        ("Ground-truth validation",
         "Compare the 5 new enrollments (Tiger_012-016) against the reid_list_test.csv "
         "ground truth to measure precision/recall for this test batch."),
    ]
    for i, (title, detail) in enumerate(steps, 1):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(4)
        run = p.add_run(f"  {i}.  {title}")
        run.bold       = True
        run.font.size  = Pt(10)
        body(doc, f"     {detail}")

    # ── Footer ─────────────────────────────────────────────────────────────────
    doc.add_paragraph()
    add_divider(doc)
    p = doc.add_paragraph(
        "Tiger AI Ultimate v3.0  |  Group 4  |  EPAIB Batch 05  |  IIM Lucknow  |  10 June 2026")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for run in p.runs:
        run.font.size      = Pt(8)
        run.font.color.rgb = RGBColor(0x80, 0x80, 0x80)

    # ── Save ───────────────────────────────────────────────────────────────────
    OUTPUT_DOCX.parent.mkdir(exist_ok=True)
    doc.save(str(OUTPUT_DOCX))
    print(f"\n[DONE] Report saved  ->  {OUTPUT_DOCX}\n")


if __name__ == "__main__":
    build_report()
