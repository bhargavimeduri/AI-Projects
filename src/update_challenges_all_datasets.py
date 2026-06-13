"""
Comprehensive update to Challenges_Issues_and_Resolutions_V2.docx
- Replace all [Image Placeholder] lines with actual image names + dataset
- Add NEW Appendix B: Complete Dataset-wise Challenge Registry (ALL 19 datasets)
- Add NEW Appendix C: Master Bug Fix Registry with image references
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")
from docx import Document
from docx.shared import Pt, RGBColor, Inches, Cm
from docx.oxml.ns import qn
from docx.enum.table import WD_TABLE_ALIGNMENT
from lxml import etree

INPUT = r"C:\Users\91630\OneDrive\Desktop\Bhargavi AIB\epaib-batch05-group4\Output Submission\Challenges_Issues_and_Resolutions_V2.docx"
OUTPUT = INPUT

doc = Document(INPUT)
body = doc.element.body

# ─────────────────────────────────────────────────────────────
# STEP 1: Replace any remaining [Image Placeholder] lines
# ─────────────────────────────────────────────────────────────
image_map = {
    "Tiger walking near water body with its reflection visible":
        "Reference Image: 002655.jpg | Dataset: Test Data 13 | Tiger near water with visible reflection, suppressed by flip-similarity detection.",
    "Two or three tigers in the same frame, all with full or mostly visible bodies":
        "Reference Image: 000082.jpg | Dataset: Test Data 4 | Two tigers fully visible. V1 missed 2nd tiger (YOLOv8n too small). V2 detects both.",
    "One tiger in the foreground (full body visible) and a second tiger partially hidden":
        "Reference Image: 000909.jpg | Dataset: Test Data 9 | Foreground tiger + partially visible 2nd tiger (Corner_Trace). 3rd tiger too occluded for any model.",
    "Two tigers on either side of a fence":
        "Reference Image: 000341.jpg | Dataset: Test Data 6 | Also: 000946.jpg (Test Data 9), 000102.jpg (Test Data 4). Fence-separated tigers flagged for human review.",
    "Tiger partially in frame with only head, tail, legs, or a body segment visible":
        "Reference Image: 002673.jpg | Dataset: Test Data 13 | Rear-view tiger detected via COCO proxy fallback with species gate bypass.",
    "Tiger photographed at night using IR flash":
        "Reference Image: 000956.jpg | Dataset: Test Data 9 | Also: 000123.jpg (Test Data 4), 002657.jpg (Test Data 13). IR species gate bypass (Fix F).",
    "Tiger in fog, haze, or backlit conditions":
        "Reference Image: 002677.jpg | Dataset: Test Data 13 | Also: 000143.jpg (Test Data 4, initially rejected as blurry). Combined blur metric fixed this.",
    "Tiger in motion creating directional blur, or rain streaks":
        "Reference Image: 002677.jpg | Dataset: Test Data 13 | Also: 000182.jpg (Test Data 5). Combined Laplacian + Gradient metric tolerates motion blur.",
    "Camera-trap image triggered by wind, vegetation movement, or other animals but no tiger":
        "Reference Images: Multiple empty frames across all test datasets. 3-stage cascade returns tiger_count=0 correctly.",
    "Camera-trap image containing non-tiger animals (deer, elephant, wild boar, monkey, bear)":
        "Reference Images: 1.44.16 PM (1).jpeg (Test Bh, orangutan/sloth bear), 1.44.16 PM (5).jpeg (Test Bh, hyena/coyote). Species gate correctly rejects.",
    "Camera-trap image containing a felid species visually similar to a tiger":
        "Reference Image: 007000.jpeg | Dataset: Test Data 3 | Lion image. ResNet50 top-3: lion, cougar, tiger. Big-cat dominance check correctly rejects.",
    "Tigers in various poses, ages, and orientations":
        "Reference Image: 002701.jpg | Dataset: Test Data 13 (vertical tiger) | Also: 007005.jpeg (Test Data 3, 3 tiger cubs), 000084.jpg (Test Data 4, vertical).",
    "Camera-trap images with non-standard dimensions, orientation, or resolution":
        "Reference Image: 002687.jpg | Dataset: Test Data 13 (129x134px small) | Also: 000084.jpg (Test Data 4, portrait orientation), 000491.jpg (Test Data 8, 314x215px).",
    "Tiger walking alongside water body with its reflection visible":
        "Reference Image: 007004.jpeg | Dataset: Test Data 3 | Muddy brown puddle reflection. detect_water_presence failed (not blue). Fixed with flip_sim criterion.",
    "Tiger partially occluded by tall grass, bamboo, or tree branches":
        "Reference Image: 002711.jpg | Dataset: Test Data 13 | Also: 004840.jpg (Test MT3, tiger behind foreground tiger). Cascade + LowConf top-up detects partial.",
    "Tiger in deep shadow under tree canopy or at twilight":
        "Reference Image: 002711.jpg | Dataset: Test Data 13 | Dark enhancement (gamma 0.5 + CLAHE 8x8) reveals stripe patterns for detection.",
    "Single image containing multiple challenge types simultaneously":
        "Reference Image: 008001.jpeg | Dataset: Test Data 3 | Night + 2 adults + 1 cub (3 tigers). Multiple challenges: IR, multi-tiger, partial visibility.",
    "Same tiger appearing in multiple images across different camera positions":
        "Reference Image: Tiger_001 across 002656.jpg, 002657.jpg, 002663.jpg | Dataset: Test Data 13 | Match scores improve: 0.78 to 0.92 via running-average fingerprint.",
}

updated_count = 0
for i, p in enumerate(doc.paragraphs):
    if "[Image Placeholder:" in p.text:
        for key, replacement in image_map.items():
            if key in p.text:
                for run in p.runs:
                    run.text = ""
                p.runs[0].text = replacement
                p.runs[0].bold = True
                p.runs[0].font.color.rgb = RGBColor(0, 51, 153)
                p.runs[0].font.size = Pt(10)
                updated_count += 1
                break
    # Also fix any that were already updated but need the broader references
    elif p.text.startswith("Reference Image:") and "Dataset:" not in p.text:
        for key, replacement in image_map.items():
            # Match on image name
            for img_name in ["002655", "000082", "000909", "000341", "002673", "000956",
                             "002677", "002701", "002687", "002711", "007004"]:
                if img_name in p.text and img_name in replacement:
                    for run in p.runs:
                        run.text = ""
                    p.runs[0].text = replacement
                    p.runs[0].bold = True
                    p.runs[0].font.color.rgb = RGBColor(0, 51, 153)
                    p.runs[0].font.size = Pt(10)
                    updated_count += 1
                    break

print(f"Step 1: Updated {updated_count} image reference lines")


# ─────────────────────────────────────────────────────────────
# STEP 2: Find where Appendix section is and add new appendices
# ─────────────────────────────────────────────────────────────

def make_para(text, style_name):
    p = etree.SubElement(body, qn("w:p"))
    body.remove(p)
    pPr = etree.SubElement(p, qn("w:pPr"))
    pStyle = etree.SubElement(pPr, qn("w:pStyle"))
    style = doc.styles[style_name]
    pStyle.set(qn("w:val"), style.style_id)
    r = etree.SubElement(p, qn("w:r"))
    t = etree.SubElement(r, qn("w:t"))
    t.text = text
    t.set(qn("xml:space"), "preserve")
    return p

# Find the last paragraph in the document
last_para = doc.paragraphs[-1]._element

# Add Appendix B: Complete Dataset-wise Challenge Registry
appendix_b_content = [
    ("Appendix B: Complete Dataset-Wise Challenge Registry", "Heading 1"),
    ("This appendix documents every challenge encountered across ALL 19 test datasets tested during V1 and V2 development, from the earliest runs to final validation.", "Normal"),

    ("B.1  Amur Tiger Training Data (3,392 images)", "Heading 2"),
    ("The ATRW benchmark dataset was the first large-scale validation. Key findings:", "Normal"),
    ("25.9% of images rejected as blurry (Laplacian < 80 threshold, later lowered to 10). 54% of detections required Whole-Image Fallback because OIV7 missed close-up crops where tiger fills entire frame. Corner Trace bug discovered: all 452 fallback detections incorrectly labelled as Corner_Trace because fallback box starts at (0,0) which is always within 12px of border. Fix: exclude Whole_Image_Fallback from corner trace check. Species gate correctly rejected 15 non-tiger crops including zoo cage bars (labelled as 'prison', 'bulletproof vest') and background-only frames.", "Normal"),

    ("B.2  Test Bh (35 images — Bhargavi/Prof dataset)", "Heading 2"),
    ("Cross-dataset generalisation test using Bhargavi's collected images. Challenges:", "Normal"),
    ("Images scored low on Laplacian blur (38-48 vs threshold 80). Dark IR images (brightness 11-13) were rejected before IR detection could run. Non-tiger animals present: orangutan, sloth bear, hyena, coyote — all correctly rejected by species gate. Result after fixes: 34/35 detected, 13 new tigers + 2 cross-dataset recaptures.", "Normal"),
    ("Key images: 1.44.16 PM (1).jpeg — orangutan/sloth bear (species gate reject). 1.44.16 PM (5).jpeg — hyena/coyote (species gate reject). 1.44.16 PM (3).jpeg — single tiger detected correctly.", "Normal"),

    ("B.3  Test Data 1 (10 images)", "Heading 2"),
    ("First water reflection bug discovered here.", "Normal"),
    ("006000.jpeg: 1 tiger + water reflection counted as 2 tigers. Grid scan L/R split detected reflection as new individual (Tiger_079). Root cause: reflection passes species gate (stripe patterns still visible when flipped) and fingerprint similarity ~0.45-0.60 is below GRID_SAME_TIGER_THRESH of 0.68. Fix: detect_water_presence() + relative flip_sim > direct_sim criterion. Initial absolute threshold 0.70 was too high (flip_sim was 0.529). Switched to relative criterion.", "Normal"),

    ("B.4  Test Data 2", "Heading 2"),
    ("005052.jpg: Pipeline predicted 2 tigers (Tiger_001 + Tiger_081 in bottom portion). Flagged for human confirmation — uncertain multi-tiger detection. This was an early case that led to the Confidence Flag Rule.", "Normal"),

    ("B.5  Test Data 3 (13 images)", "Heading 2"),
    ("Multiple critical bugs discovered in this dataset:", "Normal"),
    ("007000.jpeg: LION image. ResNet50 top-3 = ['lion', 'cougar', 'tiger']. Pipeline initially accepted because 'tiger' appeared at position 3 BEFORE the big-cat dominance check ran. Fix: check top-1 against NON_TIGER_BIG_CATS list BEFORE tiger acceptance.", "Normal"),
    ("007004.jpeg: 1 tiger + water reflection in muddy brown puddle on left side. Remainder_L counted reflection as new tiger. detect_water_presence returned False because puddle was brown (not blue — failed blue-pixel check). Fix: removed detect_water_presence gate from remainder scan; replaced with flip_sim > direct_sim AND flip_sim >= 0.55.", "Normal"),
    ("007005.jpeg: 3 tiger cubs detected as 4. Three similar-looking cubs all matched Tiger_001 (an adult from earlier images) due to high cosine similarity. Challenge: cubs matching adult fingerprint causes ID confusion.", "Normal"),
    ("008001.jpeg: Night image with 2 adults + 1 cub (3 tigers). Pipeline counted only 1. Remainder_R blocked a genuine second tiger with sim=0.88 (same-tiger guard triggered on different tiger). Challenge: under-counting in night/multi-tiger scenes.", "Normal"),

    ("B.6  Test Data 4 (20 images, 000081-000147)", "Heading 2"),
    ("V1 to V2 upgrade validation dataset. Key challenges:", "Normal"),
    ("000082.jpg: 2 tigers fully visible. V1 (YOLOv8n) missed 2nd tiger entirely. V2 (YOLOv8l) correctly detected both. This image motivated the model upgrade from nano to large.", "Normal"),
    ("000084.jpg: Vertical/portrait orientation tiger. Grid scan incorrectly split into L/R halves (designed for landscape). V1 counted 2 tigers (false split). Fix: orientation-aware grid — portrait images use T/B split, landscape uses L/R.", "Normal"),
    ("000102.jpg: 2 tigers — one behind fence. V2 detects fence but cannot confirm 2nd tiger through fence bars. Always flagged [FENCE] for human review. Geometrically unresolvable from single 2D frame.", "Normal"),
    ("000123.jpg: IR Night image. Species gate returned dog breeds (Tibetan mastiff, Groenendael, Japanese spaniel). Near-identical RGB channels. Fix F: bypass species gate when IR + YOLO conf >= 0.15.", "Normal"),
    ("000143.jpg: Rejected as blurry by V1 (Laplacian 22.6 < threshold 25). Image was visually clear. Fix: combined Laplacian + Gradient Magnitude metric, threshold lowered from 25 to 10.", "Normal"),

    ("B.7  Test Data 5 (20 images)", "Heading 2"),
    ("000182.jpg: V1 wrongly rejected as blurry (Laplacian variance 22.6 < 25). V2 accepted correctly with combined blur metric. All three V2 algorithm upgrades validated here: larger model (YOLOv8l), better blur detection, orientation-aware grid logic. Result: 20/20 correct (100%).", "Normal"),

    ("B.8  Test Data 6 (20 images)", "Heading 2"),
    ("000341.jpg: 2 tigers (one in front, one behind fence). Pipeline predicted 1. Fence occlusion too severe for auto-detection — routed to human review with [FENCE-SIGNAL] flag.", "Normal"),
    ("000363.jpg: 1 tiger but pipeline predicted 2 (false positive). Tiger_003 enrolled with match score 0.44 despite vertical guard being active. Fix: hard-reject new enrollments with match_score < 0.50 under vertical guard. Filename mismatch bug also found: ground truth dict used assumed filenames that did not match actual files, causing reported accuracy to drop to 25%.", "Normal"),

    ("B.9  Test Data 7 (20 images, 000397-000442)", "Heading 2"),
    ("000428.jpg: Missed fence tiger — same class of problem as 000341 (second tiger physically occluded behind fence). Flagged for human review. 000422.jpg: Tiger standing vertically, vertical guard triggered correctly. Initially 5 images had spurious fence flags — after Bug Fix 18 (smart flagging), reduced to 0. V2 accuracy: 19/20 (95%).", "Normal"),

    ("B.10  Test Data 8 (20 images, 000448-000498)", "Heading 2"),
    ("000466.jpg: V1 predicted 2 tigers (false positive from corner overlap). Actual = 1 tiger. Fix: Corner Overlap Guard correctly blocks in V2.", "Normal"),
    ("000491.jpg: V1 and V2 predicted 0 tigers. Actual = 1. Image very small (314x215 pixels). YOLO found zero boxes, Whole_Image_Fallback triggered but ResNet50 top-3 returned ox/ibex/bison — species gate misclassification on tiny image. Now flagged [REJECTED] for human review.", "Normal"),

    ("B.11  Test Data 9 (20 images, 000900-000972)", "Heading 2"),
    ("000903.jpg / 000916.jpg: Same tiger detected by 2 YOLO boxes — counted as 2 (over-counting). Root cause: tiger_count = valid_count used raw box count. Fix E: tiger_count = len(set(tiger_ids)).", "Normal"),
    ("000909.jpg: 3 tigers present, V2 detected 2. Vertical guard blocked Remainder + Margin scans entirely. Fix G: unblock with stricter thresholds (coverage 0.75, similarity 0.75). 3rd tiger too occluded = YOLO model limit.", "Normal"),
    ("000946.jpg: 2 tigers with fence, counted as 1. Fence detected but auto-scans exhausted. Fix D: fence + 1 tiger + scans fail = needs_confirmation=True.", "Normal"),
    ("000956.jpg: IR Night image. Species gate returned ram/retriever/ox on greyscale. Fix F: IR bypass when YOLO conf >= 0.15, accept with LOW confidence flag.", "Normal"),

    ("B.12  Test Data 10 (20 images)", "Heading 2"),
    ("000976.jpg: Standing tiger upper-body guard issue — Remainder_T enrolled head/back as second tiger. Bug Fix 15: block Remainder_T when existing box covers >40% of image height.", "Normal"),
    ("000985.jpg: Same tiger, 2 YOLO boxes with different IDs (body part fingerprints diverged, match 0.60). Bug Fix 16: same-image cross-check merges if cosine similarity >= 0.50. Result: 20/20 correct (100%).", "Normal"),

    ("B.13  Test Data 11 (20 images)", "Heading 2"),
    ("FenceZone false positive: vegetation/fence created ambiguous view, re-detecting same tiger. Bug Fix 17: FenceZone detections with similarity 0.65-0.82 blocked from enrollment. 002702.jpg: Smart review flag validation — Bug Fix 18: only flag when actual detection uncertainty exists. Result: 20/20 correct (100%).", "Normal"),

    ("B.14  Test Data 12 (20 images)", "Heading 2"),
    ("002618.jpg: 2 tigers (main + partial at top-right behind fence). Pipeline detected 1, flagged [FENCE] for review.", "Normal"),
    ("002600.jpg: Tiger crossing fence, only back+tail visible. YOLO found nothing, species gate rejected rear view. Bug Fix 19: fence + whole-image fallback species bypass with LOW confidence.", "Normal"),
    ("002627.jpg: 1 sitting tiger marked as 2. Same-tiger sim 0.81 slipped through 0.82 threshold. Also: 76px narrow fence strip passed via top-5 widening. Bug Fix 20: lower REMAINDER_SAME_TIGER_THRESH to 0.80. Bug Fix 20b: reject narrow margin strips passing only via top-5.", "Normal"),

    ("B.15  Test Data 13 (20 images — Final Validation)", "Heading 2"),
    ("002655.jpg: Water reflection — suppressed correctly. 002657.jpg: IR night — species bypass working. 002673.jpg: Rear-view partial tiger — detected via COCO proxy. 002677.jpg: Dark/dusk conditions — CLAHE enhancement working. 002687.jpg: Very small image (129x134px) — Whole-Image Fallback. 002701.jpg: Vertical tiger — FLAGGED for human review. 002711.jpg: Vegetation occlusion at dusk — dark enhancement applied. Result: 20/20 correct (100%), 8 unique tigers, 0 false positives.", "Normal"),

    ("B.16  Test MT (10 WhatsApp images)", "Heading 2"),
    ("Most species-diverse dataset: tigers, leopards, water buffalo, orangutan, hyena. 7 non-tigers all correctly rejected. Zero false positives. 3 unique new tigers identified. Grid scan algorithm found 2 additional tigers previously missed.", "Normal"),

    ("B.17  Test MT2 (10 images, a.jpeg to j.jpeg)", "Heading 2"),
    ("e.jpeg: African wildcat. ResNet50 top-3: ['egyptian cat', 'tabby', 'tiger cat']. Old code accepted 'tiger cat' as tiger — FALSE POSITIVE. Tiger cat (Leopardus tigrinus) is a South American wildcat, NOT a tiger. Bug Fix 10: permanent exclusion rule for tiger cat, tiger shark, tiger beetle. CRITICAL rule — never to be reverted.", "Normal"),

    ("B.18  Test MT3 (multi-tiger stress test)", "Heading 2"),
    ("004840.jpg: 2 tigers (foreground + background). Foreground tiger YOLO box covers 73-88% of all halves — half-scan cannot reach second tiger. Bug Fix 13: Low-Confidence OIV7 Top-Up — second YOLO pass at conf=0.05 on raw image (not preprocessed). Result: 2 tigers detected (Tiger_001 + Tiger_073 at conf=0.08).", "Normal"),
    ("004880.jpg: 2 tigers. Initially detected 3 (Remainder_R max_sim=0.81, just below 0.82 threshold). Then detected 1 (under-counting). Fix: Remainder scan cross-half guard (REMAINDER_CROSS_HALF_THRESH = 0.62). Cross-half sim 0.65 > 0.62 correctly blocked duplicate.", "Normal"),

    ("B.19  Test MT4", "Heading 2"),
    ("Validation run confirming all prior fixes hold. No new bugs discovered. Pipeline processed all images correctly with fixes from MT, MT2, and MT3 in place.", "Normal"),

    ("B.20  Test Round 2 (18 images)", "Heading 2"),
    ("Final validation round alongside Test Bh to confirm pipeline robustness across different image sources. All fixes from earlier rounds validated.", "Normal"),

    # ─────────────────────────────────────────────────────────────
    # Appendix C: Master Bug Fix Registry
    # ─────────────────────────────────────────────────────────────
    ("Appendix C: Master Bug Fix Registry", "Heading 1"),
    ("Complete chronological list of all bug fixes applied during V1-to-V2 development, with the specific images and datasets that exposed each bug.", "Normal"),
]

# Insert all appendix content at the end
prev = last_para
for text, style in appendix_b_content:
    new_p = make_para(text, style)
    prev.addnext(new_p)
    prev = new_p

print("Step 2: Added Appendix B (Dataset-Wise Challenge Registry) and Appendix C header")


# ─────────────────────────────────────────────────────────────
# STEP 3: Add Bug Fix table as actual table
# ─────────────────────────────────────────────────────────────
# Add the table after the last paragraph we inserted

bug_fixes = [
    ("Bug Fix 2", "Dark IR images rejected before IR check", "Test Data 4 (000123.jpg)", "IR detection runs BEFORE brightness check"),
    ("Bug Fix 6", "COCO class 22=zebra misidentified as backpack", "Test MT (6.jpeg)", "Fixed class ID mapping"),
    ("Bug Fix 7", "OIV7 missed detections on preprocessed images", "Test MT (various)", "Dual scan: raw + preprocessed image"),
    ("Bug Fix 8", "Grid Scan for multi-tiger frames", "Test MT (various)", "Split frame L/R and T/B, verify each half"),
    ("Bug Fix 9b", "Grid scan over-counting on single tiger", "Test MT (various)", "Similarity threshold 0.68 between halves"),
    ("Bug Fix 9c", "Grid scan double-enrollment via T/B after L/R", "Test Bh (1.44.16 PM)", "grid_handled=True flag prevents T/B after L/R"),
    ("Bug Fix 10", "tiger cat label accepted as tiger (FALSE POSITIVE)", "Test MT2 (e.jpeg)", "Permanent exclusion: tiger cat, tiger shark, tiger beetle"),
    ("Bug Fix 11", "Remainder scan for missed tigers post-YOLO", "Test MT3 (various)", "Split image into 4 halves (L/R/T/B), check each"),
    ("Bug Fix 12", "Remainder scan cross-half guard", "Test MT3 (004880.jpg)", "Compare fingerprints between halves, block if similar"),
    ("Bug Fix 13", "Low-conf 2nd tiger missed by NMS suppression", "Test MT3 (004840.jpg)", "2nd YOLO pass at conf=0.05 on raw image"),
    ("Bug Fix 14", "Margin scan for tigers outside accepted boxes", "Test MT3 (various)", "Extract strips outside box, run species gate"),
    ("Bug Fix 15", "Standing tiger Remainder_T false positive", "Test Data 10 (000976.jpg)", "Block Remainder_T when box covers >40% height"),
    ("Bug Fix 16", "Same tiger 2 boxes counted as 2", "Test Data 10 (000985.jpg)", "Cross-check: merge if fingerprint sim >= 0.50"),
    ("Bug Fix 17", "FenceZone re-detecting same tiger", "Test Data 11 (various)", "Block enrollment when sim 0.65-0.82"),
    ("Bug Fix 18", "Noisy false review flags", "Test Data 11 (002702.jpg)", "Smart flags: only when actual uncertainty exists"),
    ("Bug Fix 19", "Rear-view tiger rejected by species gate", "Test Data 12 (002600.jpg)", "Fence + fallback species bypass, LOW confidence"),
    ("Bug Fix 20", "Lying tiger same-tiger sim 0.81 < threshold 0.82", "Test Data 12 (002627.jpg)", "Lower REMAINDER_SAME_TIGER_THRESH to 0.80"),
    ("Bug Fix 20b", "Narrow margin strip false positive via top-5", "Test Data 12 (002627.jpg)", "Reject narrow strips passing only via top-5"),
    ("Fix A", "Vertical guard blocked fence scan entirely", "Test Data 4 (000082.jpg)", "Unblock fence scan when vertical active"),
    ("Fix B", "Vertical guard blocked LowConf scan", "Test Data 4/9 (000082.jpg)", "Unblock with stricter IoU 0.30"),
    ("Fix C", "Hard-block changed to soft-flag", "Test Data 9 (various)", "Soft-flag LOW confidence instead of hard-reject"),
    ("Fix D", "No fence flag when all scans fail", "Test Data 4 (000102.jpg)", "Always flag for human review when fence detected"),
    ("Fix E", "Over-counting: raw box count not unique IDs", "Test Data 9 (000903/000916)", "tiger_count = len(set(tiger_ids))"),
    ("Fix F", "IR Night species gate failure", "Test Data 9 (000956.jpg)", "Bypass species gate when IR + YOLO >= 0.15"),
    ("Fix G", "Vertical guard blocked Remainder + Margin", "Test Data 9 (000909.jpg)", "Unblock with stricter thresholds (cov 0.75, sim 0.75)"),
    ("Water Fix 1", "Grid scan water reflection over-counting", "Test Data 1 (006000.jpeg)", "detect_water_presence + flip_sim > direct_sim"),
    ("Water Fix 2", "Brown puddle reflection not detected as water", "Test Data 3 (007004.jpeg)", "Removed water gate; use flip_sim > direct_sim AND >= 0.55"),
    ("Corner Trace Fix", "Fallback detections all labelled Corner_Trace", "Amur Train dataset", "Exclude Whole_Image_Fallback from corner trace check"),
    ("Lion Reject Fix", "Lion image accepted (tiger at position 3)", "Test Data 3 (007000.jpeg)", "Check top-1 against NON_TIGER_BIG_CATS before acceptance"),
    ("Blur Fix", "Valid images rejected as blurry", "Test Data 4 (000143.jpg), TD5 (000182.jpg)", "Combined Laplacian + Gradient metric, threshold 25 to 10"),
    ("Orientation Fix", "Grid scan L/R split on portrait image", "Test Data 4 (000084.jpg)", "Orientation-aware grid: portrait uses T/B, landscape uses L/R"),
]

# Create the table
table = doc.add_table(rows=1, cols=4)
table.alignment = WD_TABLE_ALIGNMENT.CENTER

# Header row
headers = ["Fix ID", "Problem", "Image / Dataset", "Resolution"]
for j, h in enumerate(headers):
    cell = table.rows[0].cells[j]
    cell.text = h
    for para in cell.paragraphs:
        for run in para.runs:
            run.bold = True
            run.font.size = Pt(9)

# Data rows
for fix_id, problem, image_ds, resolution in bug_fixes:
    row = table.add_row()
    row.cells[0].text = fix_id
    row.cells[1].text = problem
    row.cells[2].text = image_ds
    row.cells[3].text = resolution
    for cell in row.cells:
        for para in cell.paragraphs:
            for run in para.runs:
                run.font.size = Pt(8)

# Move table to end (after Appendix C header)
# The table was added at end of body by default, which is correct
print(f"Step 3: Added Master Bug Fix table with {len(bug_fixes)} entries")

# ─────────────────────────────────────────────────────────────
# STEP 4: Update TOC entries to include new appendices
# ─────────────────────────────────────────────────────────────
for i, p in enumerate(doc.paragraphs):
    if p.text.strip() == "10. Appendix":
        # Add new TOC lines after this
        toc_additions = [
            "    Appendix A: Image-to-Scenario Mapping",
            "    Appendix B: Complete Dataset-Wise Challenge Registry",
            "    Appendix C: Master Bug Fix Registry (31 fixes)",
        ]
        ref = p._element
        for line in reversed(toc_additions):
            new_p = make_para(line, "Normal")
            ref.addnext(new_p)
        print("Step 4: Updated Table of Contents with new appendices")
        break

# Save
doc.save(OUTPUT)
print(f"\nSaved to: {OUTPUT}")

# Final verification
doc2 = Document(OUTPUT)
total_paras = len(doc2.paragraphs)
headings = [p.text for p in doc2.paragraphs if p.style.name.startswith("Heading")]
print(f"Total paragraphs: {total_paras}")
print(f"Total headings: {len(headings)}")
for h in headings:
    print(f"  - {h}")
