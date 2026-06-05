"""
Creates a Word document for pasting Mayank's blueprint screenshots.
"""
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

doc = Document()

# Page margins
for section in doc.sections:
    section.top_margin    = Inches(0.8)
    section.bottom_margin = Inches(0.8)
    section.left_margin   = Inches(1.0)
    section.right_margin  = Inches(1.0)

def heading(text, level=1):
    p = doc.add_heading(text, level=level)
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    return p

def body(text):
    p = doc.add_paragraph(text)
    p.runs[0].font.size = Pt(11)
    return p

def placeholder(label):
    p = doc.add_paragraph()
    run = p.add_run(f"[ Screenshot: {label} ]")
    run.font.size     = Pt(10)
    run.font.italic   = True
    run.font.color.rgb = RGBColor(0x99, 0x99, 0x99)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    # Add a visible box-like border via shading note
    doc.add_paragraph()   # blank line spacer
    return p

# ── Title ──────────────────────────────────────────────────────────────────
heading("Mayank Tandon — AI Tiger Detection Blueprint", level=1)
body("Source: Claude.ai shared conversation  |  EPAIB Batch 05, Group 4, IIM Lucknow")
body("Purpose: Capture Mayank's 3-model approach and testing assumptions for team alignment.")
doc.add_paragraph()

# ── Section 1 ──────────────────────────────────────────────────────────────
heading("1. Problem Statement & Context", level=2)
placeholder("Problem statement — camera trap challenges, forest department requirements")

# ── Section 2 ──────────────────────────────────────────────────────────────
heading("2. Reality Check & Expectations", level=2)
placeholder("200–500 images / beginner / Google Colab / 2-model split table")

# ── Section 3 ──────────────────────────────────────────────────────────────
heading("3. The 3-Model Architecture", level=2)
placeholder("Model 1 — Image Triage / Quality Filter")
placeholder("Model 2 — Tiger Detection & Individual ID")
placeholder("Model 3 — Tiger Type Classification")

# ── Section 4 ──────────────────────────────────────────────────────────────
heading("4. Phase-by-Phase Blueprint", level=2)

heading("Phase 1 — Data Collection & Preparation", level=3)
placeholder("Phase 1 blueprint screenshot")

heading("Phase 2 — Model 1: Image Triage", level=3)
placeholder("Phase 2 blueprint screenshot")

heading("Phase 3 — Model 2: Detection & Counting", level=3)
placeholder("Phase 3 blueprint screenshot")

heading("Phase 4 — Model 3: Tiger Type Classifier", level=3)
placeholder("Phase 4 blueprint screenshot")

heading("Phase 5 — Integration & Deployment", level=3)
placeholder("Phase 5 blueprint screenshot")

# ── Section 5 ──────────────────────────────────────────────────────────────
heading("5. Camera Trap Challenge Handling", level=2)
placeholder("How each challenge (fog, IR night, blur, overexposure, partial body) is handled")

# ── Section 6 ──────────────────────────────────────────────────────────────
heading("6. Tiger Types & Classification Logic", level=2)
placeholder("5 subspecies stripe typologies table")
placeholder("4 color morphs table")

# ── Section 7 ──────────────────────────────────────────────────────────────
heading("7. Recommended Folder Structure", level=2)
body("Mayank's Claude recommended this project folder structure (screenshot from shared chat):")
placeholder("Folder structure screenshot — paste here")

# ── Section 8 ──────────────────────────────────────────────────────────────
heading("8. Additional Notes / Questions Asked", level=2)
placeholder("Any other screenshots from the conversation")

# ── Footer note ────────────────────────────────────────────────────────────
doc.add_paragraph()
body("Compiled by: Bhargavi Meduri  |  Date: 29 May 2026  |  For internal team use only.")

output = "presentation/Mayank_Blueprint_Notes_v2.docx"
doc.save(output)
print(f"Document saved: {output}")
