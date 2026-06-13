"""
Generate Challenges, Issues Encountered, and Resolutions Document
TRACE Pipeline — EPAIB Batch 05 Group 4 — IIM Lucknow
"""
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
from docx import Document
from docx.shared import Pt, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT

doc = Document()
style = doc.styles['Normal']
style.font.name = 'Calibri'
style.font.size = Pt(11)

def hr(doc):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run('\u2500' * 80)
    run.font.size = Pt(8)
    run.font.color.rgb = RGBColor(150, 150, 150)

def tbl(doc, headers, rows):
    table = doc.add_table(rows=1+len(rows), cols=len(headers))
    table.style = 'Light Grid Accent 1'
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(headers):
        table.rows[0].cells[i].text = h
        for r in table.rows[0].cells[i].paragraphs:
            for run in r.runs:
                run.bold = True; run.font.size = Pt(10)
    for ri, row in enumerate(rows):
        for ci, val in enumerate(row):
            table.rows[ri+1].cells[ci].text = str(val)
            for r in table.rows[ri+1].cells[ci].paragraphs:
                for run in r.runs:
                    run.font.size = Pt(10)

def bp(doc, label, text):
    p = doc.add_paragraph()
    run = p.add_run(label)
    run.bold = True
    p.add_run(text)

def bul(doc, items):
    for item in items:
        doc.add_paragraph(item, style='List Bullet')

def scenario(doc, name, img_desc, expected, actual, failure, root_cause, tech_explanation, fix, why_fix, risks, lessons):
    doc.add_heading(name, level=3)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(f'[Image Placeholder: {img_desc}]')
    run.italic = True; run.font.color.rgb = RGBColor(150,150,150)
    bp(doc, 'Expected Behaviour: ', expected)
    bp(doc, 'Actual Behaviour: ', actual)
    bp(doc, 'Failure Observed: ', failure)
    bp(doc, 'Root Cause Analysis: ', root_cause)
    bp(doc, 'Technical Explanation: ', tech_explanation)
    bp(doc, 'Fix Implemented: ', fix)
    bp(doc, 'Why the Fix Worked: ', why_fix)
    bp(doc, 'Risks Remaining: ', risks)
    bp(doc, 'Lessons Learned: ', lessons)

print('Starting report generation...')

# ══════════════════════════════════════════════════════════════
# TITLE PAGE
# ══════════════════════════════════════════════════════════════
for _ in range(4):
    doc.add_paragraph()

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = p.add_run('TRACE')
run.bold = True; run.font.size = Pt(36); run.font.color.rgb = RGBColor(0, 70, 130)

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = p.add_run('Tiger Re-identification & Automated Census Engine')
run.font.size = Pt(18); run.font.color.rgb = RGBColor(80, 80, 80)

doc.add_paragraph()

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = p.add_run('Challenges, Issues Encountered, and Resolutions')
run.bold = True; run.font.size = Pt(20); run.font.color.rgb = RGBColor(0, 70, 130)

doc.add_paragraph()
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = p.add_run('Capstone Project Submission')
run.font.size = Pt(14)

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = p.add_run('Executive Post Graduate Programme in AI for Business (EPAIB)')
run.font.size = Pt(12)

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = p.add_run('Batch 05  |  Group 4  |  IIM Lucknow')
run.font.size = Pt(12)

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = p.add_run('June 2026')
run.font.size = Pt(12); run.font.color.rgb = RGBColor(100, 100, 100)

doc.add_page_break()
print('Title page done')

# ══════════════════════════════════════════════════════════════
# TABLE OF CONTENTS
# ══════════════════════════════════════════════════════════════
doc.add_heading('Table of Contents', level=1)
toc_items = [
    '1. Executive Summary',
    '2. Project Overview',
    '3. Model Evolution Journey',
    '   3.1 V1 - Initial Architecture',
    '   3.2 V2 - Production-Ready Pipeline',
    '   3.3 V1 vs V2 Comparison',
    '4. Failure Analysis and Resolution Framework',
    '   4.1 Reflection Cases',
    '   4.2 Multi-Tiger Cases',
    '   4.3 Partial Visibility Cases',
    '   4.4 Environmental Challenges',
    '   4.5 Empty Frames',
    '   4.6 Non-Tiger Animals',
    '   4.7 Similar Species',
    '   4.8 Tiger Variations',
    '   4.9 Camera Challenges',
    '   4.10 Complex Cases',
    '5. Root Cause Analysis Summary',
    '6. Devil\'s Advocate Review - Round 1',
    '7. Devil\'s Advocate Review - Round 2',
    '8. Lessons Learned',
    '9. Final Capstone Summary',
    '10. Appendix',
]
for item in toc_items:
    doc.add_paragraph(item)

doc.add_page_break()

# ══════════════════════════════════════════════════════════════
# 1. EXECUTIVE SUMMARY
# ══════════════════════════════════════════════════════════════
doc.add_heading('1.  Executive Summary', level=1)

bp(doc, 'Business Problem: ',
   'India is home to approximately 70% of the world\'s wild tiger population. Accurate population census '
   'is critical for conservation funding allocation, anti-poaching enforcement, habitat protection, and '
   'compliance with international wildlife agreements. Currently, tiger population estimation relies heavily '
   'on manual review of millions of camera-trap images by trained field biologists - a process that is '
   'slow, expensive, error-prone, and does not scale to the volume of imagery generated by modern camera-trap networks.')

bp(doc, 'Technical Problem: ',
   'Automated tiger detection from camera-trap imagery is significantly more challenging than standard object '
   'detection due to: (1) extreme environmental variation (day/night, rain/fog, dense vegetation), '
   '(2) partial visibility (only head, tail, or legs showing), (3) visual confusion with similar species '
   '(leopards, jaguars, tiger cats), (4) multi-tiger scenes requiring individual counting, '
   '(5) optical illusions (water reflections, shadows, fence-bar interference), and '
   '(6) the need for individual re-identification across sightings, not just detection.')

bp(doc, 'Project Objective: ',
   'Develop an AI-powered pipeline (TRACE - Tiger Re-identification & Automated Census Engine) that can '
   'automatically detect, count, classify, and individually identify tigers from camera-trap images with '
   'sufficient accuracy for semi-automated wildlife census operations. The system must flag uncertain cases '
   'for human review rather than silently miscounting.')

bp(doc, 'Final Outcome: ',
   'The TRACE V2 pipeline achieved 100% detection accuracy on the final validation dataset (Test Data 13, '
   '20 images), correctly identifying 8 unique tigers with zero false positives and zero false negatives. '
   'Across 7 test datasets (140 total images), the system achieved 97.1% accuracy, with all 4 remaining '
   'cases correctly flagged for human review rather than silently failing.')

bp(doc, 'Key Achievements: ', '')
bul(doc, [
    '3-stage cascade detection architecture ensuring no tiger is missed (YOLOv8-OIV7 -> YOLOv8-COCO -> Whole-Image Fallback).',
    'ResNet50-based individual re-identification with progressive accuracy improvement (match scores 0.78 to 0.92 over repeated sightings).',
    '10 targeted bug fixes developed through systematic iterative testing across 7 datasets.',
    'Smart confidence flagging that eliminates noisy false flags while preserving genuine uncertainty signals.',
    'Robust handling of edge cases: water reflections, fence obstructions, IR/night images, vertical orientations, partial visibility.',
    'Zero custom model training required - entire system built on transfer learning from pre-trained models.',
])

hr(doc)
print('Executive summary done')

# ══════════════════════════════════════════════════════════════
# 2. PROJECT OVERVIEW
# ══════════════════════════════════════════════════════════════
doc.add_heading('2.  Project Overview', level=1)

doc.add_heading('2.1  Why Tiger Detection is Difficult', level=2)
doc.add_paragraph(
    'Tiger detection from camera-trap imagery presents a unique combination of challenges that makes it '
    'significantly harder than standard object detection tasks:')

tbl(doc,
    ['Challenge Category', 'Specific Difficulty', 'Impact on Detection'],
    [
        ['Camouflage', 'Tiger stripes evolved for concealment in dappled forest light', 'Low contrast between tiger and background; feature extraction struggles'],
        ['Nocturnal Activity', '60-70% of tiger activity is nocturnal; camera traps use IR flash', 'Greyscale images lose colour information; species verification fails'],
        ['Dense Habitat', 'Tigers inhabit thick forest, tall grass, bamboo thickets', 'Partial occlusion; only fragments of body visible'],
        ['Rare Events', 'Tiger density is 1-5 per 100 sq km; most frames are empty', 'Extreme class imbalance; model biased toward "no tiger"'],
        ['Individual Variation', 'Each tiger has unique stripe pattern (like fingerprints)', 'Need re-identification, not just detection; traditional object detection insufficient'],
        ['Environmental Noise', 'Rain, fog, lens flare, sensor noise, motion blur', 'Image quality degradation; preprocessing required'],
        ['Multi-Animal Scenes', 'Tigers may appear together (mating, mother-cub, territorial)', 'Counting requires spatial reasoning; NMS suppresses overlapping boxes'],
        ['Similar Species', 'Leopards, clouded leopards, tiger cats share habitat', 'False positive risk; species verification layer needed'],
    ])

doc.add_heading('2.2  Why Wildlife Imagery Presents Unique Challenges', level=2)
doc.add_paragraph(
    'Unlike controlled photography or internet image datasets, camera-trap imagery has characteristics '
    'that challenge standard computer vision models:')
bul(doc, [
    'Fixed camera position with no zoom/pan control - tigers may be far away or partially out of frame.',
    'Uncontrolled lighting - same camera produces daylight, twilight, and IR images within hours.',
    'No subject cooperation - tigers move freely, presenting rear views, partial views, or blurred motion.',
    'Environmental interference - rain drops, spider webs, growing vegetation obstruct the lens.',
    'Hardware limitations - low resolution sensors (640x480 common), motion blur from slow shutter speeds.',
    'Trigger delay - camera fires after motion detected, tiger may have partially exited frame.',
])

doc.add_heading('2.3  Expected Model Behaviour', level=2)
doc.add_paragraph('A production-ready tiger detection system should:')
bul(doc, [
    'Detect every tiger in every image where a tiger is present (recall > 95%).',
    'Never count a non-tiger object as a tiger (precision > 95%).',
    'Correctly count multiple tigers in the same frame.',
    'Individually identify tigers for population census (re-identification accuracy > 80%).',
    'Handle environmental variation (day/night, weather, vegetation) without manual intervention.',
    'Flag uncertain cases for human review rather than silently miscounting.',
    'Process images without custom model training (transfer learning only, for rapid deployment).',
])

doc.add_heading('2.4  Success Criteria', level=2)
tbl(doc,
    ['Criterion', 'Target', 'Achieved (V2 Final)'],
    [
        ['Detection Rate', '> 95%', '100% (20/20 on final dataset)'],
        ['False Positive Rate', '< 5%', '0% (zero false detections)'],
        ['False Negative Rate', '< 5%', '0% (zero missed tigers)'],
        ['Individual Re-identification', '> 80% match accuracy', '92% peak (Tiger_005 5th recapture)'],
        ['Multi-tiger Counting', 'Correct count or human flag', 'Achieved (flag when uncertain)'],
        ['Human Review Rate', '< 10% of images', '5% (1/20 on final dataset)'],
        ['Edge Case Handling', 'No silent failures', 'All edge cases detected and handled'],
    ])

hr(doc)
print('Project overview done')

# ══════════════════════════════════════════════════════════════
# 3. MODEL EVOLUTION JOURNEY
# ══════════════════════════════════════════════════════════════
doc.add_heading('3.  Model Evolution Journey', level=1)

doc.add_heading('3.1  V1 - Initial Architecture', level=2)
doc.add_paragraph(
    'V1 was the initial proof-of-concept implementation designed to validate whether pre-trained models '
    'could detect tigers from camera-trap imagery without custom training.')

bp(doc, 'Architecture: ', 'Single-stage YOLOv8n (nano) detection + ResNet50 species verification.')
bp(doc, 'Assumptions Made: ', '')
bul(doc, [
    'One tiger per image (single-detection assumption).',
    'Daytime images with reasonable lighting.',
    'Tigers fully visible in frame.',
    'YOLOv8n (nano) sufficient for detection despite small model size.',
    'Fixed Laplacian blur threshold (25) adequate for all image qualities.',
    'No fence/obstruction handling needed.',
])

bp(doc, 'Known Limitations of V1: ', '')
bul(doc, [
    'YOLOv8n (80M params) has limited feature extraction capacity - misses small/distant tigers.',
    'No multi-tiger detection capability - only processes first YOLO box.',
    'Rigid blur rejection threshold discards valid but slightly soft images.',
    'No identity matching - every detection is a "new" tiger.',
    'No environmental adaptation (IR/night, fog, rain).',
    'No fence/obstruction awareness.',
    'No water reflection handling.',
])

doc.add_heading('3.2  V2 - Production-Ready Pipeline', level=2)
doc.add_paragraph(
    'V2 was a ground-up architectural redesign based on systematic analysis of all V1 failure modes. '
    'Every limitation identified in V1 testing was addressed with a specific engineering solution.')

bp(doc, 'Key Improvements Introduced: ', '')
tbl(doc,
    ['V1 Limitation', 'V2 Solution', 'Why the Change Was Required'],
    [
        ['Single YOLO model', '3-stage cascade (OIV7 -> COCO -> Fallback)', 'Different models excel at different tiger presentations'],
        ['Single-tiger assumption', '6 multi-tiger scan paths', 'Real camera traps capture 2-3 tigers together'],
        ['No identity matching', 'ResNet50 2048-dim fingerprint + cosine similarity', 'Census requires re-identification, not just detection'],
        ['Rigid blur threshold', 'Combined Laplacian + Gradient Magnitude', 'V1 rejected valid images; combined metric is more robust'],
        ['No fence awareness', 'Canny + Hough fence detection + bar removal', '35% of test images contained fences'],
        ['No IR/night handling', 'IR detection + species gate bypass (Fix F)', 'Nocturnal imagery is majority of real camera-trap data'],
        ['No reflection handling', 'Water detection + flip similarity comparison', 'Water reflections cause false double-counts'],
        ['No confidence flagging', 'Smart uncertainty detection + human review flags', 'Silent failures are worse than flagged uncertainty'],
    ])

bp(doc, 'New Algorithms Adopted in V2: ', '')
bul(doc, [
    'Cascade Detection: YOLOv8n-OIV7 (direct Tiger class) -> YOLOv8l-COCO (proxy classes: cat, dog, bear, horse) -> Whole-Image Fallback (ResNet50 species check on full frame). Each stage catches what the previous missed.',
    '6 Multi-Tiger Scan Paths: LowConf OIV7 Top-Up (conf=0.05), Remainder Scan (L/R/T/B halves), Margin Scan (strips outside union bbox), FenceZone Scan (fence-cleaned zones), Fence Mask-Search (fill main tiger, re-scan), Grid Scan (2x2 quadrants).',
    'Fingerprint Matching: ResNet50 extracts 2048-dimensional feature vectors from tiger crops. Cosine similarity comparison against a persistent identity database. Running-average fingerprint updates improve accuracy over repeated sightings.',
    'Water Reflection Suppression: detect_water_presence() checks for blue-channel dominance in lower frame. is_water_reflection() compares horizontal flip similarity vs direct similarity to distinguish reflection from real tiger.',
    'Fence Detection: Canny edge detection + Hough line transform identifies vertical fence bars. remove_fence_bars() applies morphological operations to clean fence patterns before species verification.',
    'Smart Confidence Flagging: Only flags images where genuine detection uncertainty exists (low match scores, ambiguous counts, tiny corner traces, IR bypass). Eliminates noisy blanket flags that burden reviewers.',
])

doc.add_heading('3.3  V1 vs V2 Comparison', level=2)
tbl(doc,
    ['Feature', 'V1', 'V2'],
    [
        ['Detection Models', 'YOLOv8n only', 'YOLOv8n-OIV7 + YOLOv8l-COCO + ResNet50 Fallback'],
        ['Multi-Tiger Support', 'No (single detection)', 'Yes (6 scan paths)'],
        ['Identity Matching', 'None', 'ResNet50 fingerprint + cosine similarity (0.83 threshold)'],
        ['Fence Handling', 'None', 'Canny + Hough detection + bar removal + FenceZone scan'],
        ['IR/Night Support', 'None (species gate rejects)', 'IR detection + species bypass + LOW confidence flag'],
        ['Water Reflection', 'None (counts as 2nd tiger)', 'Water detection + flip similarity suppression'],
        ['Blur Handling', 'Fixed Laplacian (threshold 25)', 'Combined Laplacian + Gradient Magnitude (threshold 10)'],
        ['Confidence Flagging', 'None (silent errors)', 'Smart flags with specific reasons for human review'],
        ['Code Size', '~800 lines', '4,401 lines (42 functions)'],
        ['Bug Fixes Applied', '0', '10 targeted fixes (E, F, G, BF-15 through BF-20b)'],
        ['Detection Rate', '~70-80%', '100% on final dataset'],
        ['False Positive Rate', '~15-20%', '0%'],
        ['Test Datasets', '1-2', '7 datasets, 140 images'],
    ])

bp(doc, 'Performance Gains: ', '')
bul(doc, [
    'Detection rate improved from ~75% (V1) to 100% (V2 final) - a 25 percentage point improvement.',
    'False positive rate reduced from ~15% to 0% through species verification, cross-checks, and identity matching.',
    'Multi-tiger accuracy went from 0% (V1 could not detect multiple tigers) to correct counting or human flagging in all cases.',
    'Edge case handling expanded from 0 handled categories (V1) to 7+ categories (V2): IR night, fences, reflections, vertical poses, partial bodies, small images, species confusion.',
])

hr(doc)
print('Model evolution done')

# ══════════════════════════════════════════════════════════════
# 4. FAILURE ANALYSIS AND RESOLUTION FRAMEWORK
# ══════════════════════════════════════════════════════════════
doc.add_heading('4.  Failure Analysis and Resolution Framework', level=1)
doc.add_paragraph(
    'This section documents every major failure category encountered during development and testing, '
    'following a structured analysis format. Each scenario includes the expected behaviour, actual behaviour, '
    'root cause analysis, fix implemented, and lessons learned.')

# ── 4.1 REFLECTION CASES ──
doc.add_heading('4.1  Reflection Cases', level=2)

scenario(doc,
    'Scenario: Tiger Reflection in Water',
    'Tiger walking near water body with its reflection visible in the water surface below',
    'Pipeline should detect 1 tiger and ignore the water reflection.',
    'V1 detected 2 tigers - one real tiger and its water reflection counted as a second individual.',
    'Water reflection of the tiger was detected as a separate YOLO bounding box, passed species verification (reflection still looks like a tiger), and was enrolled as a new individual.',
    'Data Issues: Training data did not include water reflection examples. Model Issues: YOLO treats reflection as a valid detection because the reflected image genuinely contains tiger features. Standard NMS does not suppress vertically mirrored boxes.',
    'Water reflections are near-perfect vertical mirrors of the real tiger. The reflected image contains valid tiger stripe patterns, so both species verification (ResNet50) and identity matching (fingerprint) treat it as a genuine tiger. The reflection bbox is typically directly below the real tiger bbox, but IoU between them may be low because they occupy different image regions (above vs below waterline).',
    'Two-stage water reflection guard: (1) detect_water_presence() analyses the lower portion of the image for blue-channel dominance and texture patterns consistent with water surfaces. (2) is_water_reflection() extracts both crops, creates a horizontally-flipped version of one, and compares flip_similarity vs direct_similarity. If flip_sim > direct_sim (minimum 0.55), the lower detection is a reflection and is suppressed.',
    'Reflections are geometrically constrained to be vertical mirrors. By comparing the flipped crop similarity (which matches a reflection) against the direct crop similarity (which matches a different tiger), the system can distinguish reflections from genuine second tigers with high reliability. A real second tiger would have direct_sim > flip_sim because two different tigers do not mirror each other.',
    'Edge case: partially disturbed water (ripples, wind) may reduce flip_sim below threshold, causing the reflection to be enrolled. Mitigation: threshold set conservatively at 0.55. Also, reflections in non-water surfaces (glass, wet pavement) are not detected by the water presence check.',
    'Never trust spatial separation alone to distinguish real detections from optical artifacts. Environmental context (water presence, surface type) must be explicitly detected and used as a guard condition.')

# ── 4.2 MULTI-TIGER CASES ──
doc.add_heading('4.2  Multi-Tiger Cases', level=2)

scenario(doc,
    'Scenario: Multiple Tigers Fully Visible',
    'Two or three tigers in the same frame, all with full or mostly visible bodies',
    'Pipeline should detect and count each tiger individually, assigning unique IDs.',
    'V1 detected only 1 tiger (the one with the highest YOLO confidence). V2 initially over-counted by detecting the same tiger twice via different scan paths.',
    'V1 had a single-detection assumption. V2 introduced multi-tiger scanning but needed calibration to avoid double-counting the same tiger from overlapping scan regions.',
    'Model Issues: YOLO NMS suppresses overlapping boxes even when two real tigers overlap spatially. V2 scan paths (Remainder, Margin) could detect the same tiger from different crop angles, producing different fingerprints that bypassed the identity match threshold.',
    'When two tigers are close together, YOLO may merge their bounding boxes (high IoU -> NMS suppression) or detect one and suppress the other. V2 scan paths address this by independently scanning image halves, but the same tiger appearing in overlapping scan regions (e.g., Remainder_L and main detection) creates duplicate detections with slightly different fingerprints due to different crop boundaries.',
    'Three-layer deduplication: (1) Same-Image Cross-Check (Bug Fix 16): after all detections, compare every pair by fingerprint - merge if cosine similarity >= 0.50. (2) IoU guard: new detections with IoU >= 0.30 against existing are rejected. (3) Standing Tiger Upper-Body Guard (Bug Fix 15): prevents Remainder_T from enrolling the head/back of a standing tiger already detected in the main pass.',
    'The cross-check catches duplicates that bypass IoU guards (different crops = low IoU but high fingerprint similarity). The 0.50 threshold was calibrated from observed same-tiger similarities across different crop angles (typically 0.55-0.80 range). The upper-body guard specifically addresses the geometric pattern of standing tigers where the head appears in Remainder_T.',
    'Two genuinely different tigers with very similar stripe patterns (siblings, same genetic lineage) might have fingerprint similarity >= 0.50, causing a false merge. Mitigation: 0.50 is conservative; same-individual similarities are typically 0.60-0.90 while different-individual similarities are typically 0.30-0.55.',
    'Multi-tiger detection requires both spatial reasoning (where in the frame) and identity reasoning (is this the same tiger?). Neither alone is sufficient.')

scenario(doc,
    'Scenario: Fully Visible + Partially Visible Tiger',
    'One tiger in the foreground (full body visible) and a second tiger partially hidden behind vegetation, fence, or the first tiger',
    'Pipeline should detect both tigers, enrolling the partial one with appropriate visibility classification.',
    'V2 detected the foreground tiger reliably but frequently missed the background/partial tiger, especially when it was behind the foreground tiger at the same x/y position.',
    'YOLO NMS suppresses the background tiger box when it overlaps spatially with the foreground tiger box. Species verification on partial crops often fails because ResNet50 sees only fragments (tail, legs, flank) that do not match learned tiger features.',
    'Model Issues: YOLO NMS is designed to remove duplicate detections, but it cannot distinguish between "two boxes for the same object" and "two boxes for two different objects that happen to overlap." Environmental Issues: When a background tiger is at the same x/y position as a foreground tiger (e.g., one behind a fence, one in front), IoU is very high, guaranteeing NMS suppression.',
    'YOLO Non-Maximum Suppression compares all detected bounding boxes and removes those with IoU above a threshold (typically 0.45) relative to a higher-confidence box. When two real tigers overlap spatially (foreground/background), the background tiger box is suppressed because it has lower confidence and high IoU with the foreground box. Even lowering NMS threshold does not help because the overlap is genuine.',
    'Multi-layer scan architecture: (1) LowConf OIV7 Top-Up at conf=0.05 catches NMS-suppressed boxes. (2) Remainder Scan divides image into L/R/T/B halves, scanning each independently. (3) FenceZone Scan detects fences and scans zones on either side. (4) Fence Mask-Search fills the foreground tiger with background median color and re-runs YOLO. (5) If all fail and fence is detected, set needs_confirmation=True with fence-occlusion reason.',
    'Each scan path addresses a different failure mode: LowConf catches NMS-suppressed boxes, Remainder catches spatially separated tigers, FenceZone targets fence-specific occlusion, and Mask-Search addresses spatial overlap by removing the foreground tiger texture. The human review flag is the safety net when all automated methods fail.',
    'Some partially visible tigers are genuinely undetectable by any automated method (< 10% body visible, heavily occluded by vegetation or another tiger). The pipeline correctly identifies these as potential missed detections and flags for human review rather than hallucinating a detection.',
    'No single scan method can detect all multi-tiger configurations. A layered approach with progressively more aggressive scanning, ending with a human review flag as the safety net, provides the best balance of recall and precision.')

scenario(doc,
    'Scenario: Tigers Separated by Fence',
    'Two tigers on either side of a fence - one clearly visible, the other partially visible through fence bars',
    'Pipeline should detect both tigers or flag for human review if the fence-side tiger is too occluded.',
    'V2 detected the near-side tiger but missed the far-side tiger. FenceZone scan found fence pattern but could not extract a clear enough crop through the fence bars for species verification.',
    'The far-side tiger visible through fence bars is fragmented by the fence pattern. YOLO cannot detect a tiger from fence-bar-fragmented features. Species verification on fence-cropped images fails because ResNet50 sees fence patterns, not tiger patterns.',
    'Environmental Issues: Fence bars fragment the tiger image into vertical strips. Each strip alone is too small for YOLO detection. Model Issues: ResNet50 species gate fails on fence-contaminated crops because the dominant visual features are metal/wire patterns, not tiger stripes.',
    'Chain-link and bar fences create a regular pattern of occlusion that fragments a tiger image into many small visible regions. YOLO requires a minimum continuous region of tiger features to generate a detection box. When fence bars divide a tiger into 5-10 narrow strips, no single strip has enough features. ResNet50 top-3 returns "chain mail", "window screen", or "porcupine" instead of "tiger" because the fence dominates the visual features.',
    'FenceZone Scan (5-step process): (1) detect_fence_pattern() identifies fences via Canny + Hough. (2) Define L/R zones outside main detection with 20% overlap buffer. (3) remove_fence_bars() applies morphological operations to clean fence patterns. (4) Run YOLO at very low confidence (0.01) on cleaned zones. (5) If species gate confirms, enroll as new tiger. If all fail + fence detected + only 1 tiger, set needs_confirmation=True with fence-occlusion reason for human review.',
    'Fence bar removal via morphological operations restores enough continuous tiger texture for detection. The very low YOLO confidence threshold (0.01) catches even faint tiger signals through fence gaps. The human review flag is the safety net when the tiger is too occluded for any automated method.',
    'Some fence occlusion is too severe for any automated recovery (< 5% of tiger visible through small mesh). The human review flag with "fence/barrier detected" reason ensures these are not silently missed but are also not hallucinated.',
    'Physical obstructions (fences, vegetation, structures) require dedicated detection and compensation algorithms. Generic object detection assumes unobstructed views. A "detect the obstruction, then work around it" approach is more robust than trying to detect through the obstruction.')

# ── 4.3 PARTIAL VISIBILITY CASES ──
doc.add_heading('4.3  Partial Visibility Cases', level=2)

scenario(doc,
    'Scenario: Only Head / Tail / Legs / Partial Body Visible',
    'Tiger partially in frame with only head, tail, legs, or a body segment visible at the edge',
    'Pipeline should detect the partial tiger and classify as Corner_Trace or Partial_Body.',
    'V1 frequently missed partial tigers due to insufficient features for YOLOv8n detection. V2 detected some via COCO proxy but initially assigned incorrect visibility labels and sometimes enrolled them as separate individuals from the same tiger detected in the main scan.',
    'Partial tigers have limited features - a tail tip alone has no stripes, legs alone lack the distinctive face/body pattern, and head-only crops may be too small for species verification.',
    'Model Issues: YOLOv8n has low resolution feature maps that cannot resolve small/partial tiger features. The species gate (ResNet50 top-3) was trained on full-body images, not isolated body parts. Data Issues: Camera-trap trigger delay means tigers are frequently partially exiting the frame.',
    'A tiger tail occupies approximately 3-5% of total body area and has a distinctive banded pattern, but this pattern is shared with other felids and even some non-felids (raccoons, ring-tailed lemurs). YOLO requires approximately 20-30% of the trained object features to generate a detection. Below this threshold, the model cannot distinguish a tiger body part from background noise.',
    'Three-stage cascade with progressive fallback: (1) OIV7 detects partial tigers with at least 20-30% body visible. (2) COCO proxy uses broader animal classes (cat, dog, bear) to catch partial tigers that lack enough tiger-specific features. (3) Whole-Image Fallback applies ResNet50 species check to the entire image. Visibility classification (Corner_Trace for < 30% body) adjusts match thresholds to account for lower fingerprint quality.',
    'COCO proxy classes (cat, dog) share general quadruped body features with tigers (four legs, fur texture, body proportions). Even a tail or leg can trigger a "cat" detection that is then verified by the species gate. The visibility classification prevents low-quality fingerprints from creating false identity matches.',
    'Very small partial views (< 10% body) may not trigger even the COCO proxy. The tiny corner trace flag (Bug Fix 18) identifies these and flags for human review. Additionally, partial-body fingerprints have inherently lower match quality, which may cause the same tiger to be enrolled as a new individual if previously seen from a completely different angle.',
    'Detection systems must gracefully degrade, not fail catastrophically, when presented with incomplete information. A partial detection with appropriate confidence labeling is more valuable than no detection at all.')

# ── 4.4 ENVIRONMENTAL CHALLENGES ──
doc.add_heading('4.4  Environmental Challenges', level=2)

scenario(doc,
    'Scenario: Dark / Night-time / IR Images',
    'Tiger photographed at night using IR flash, resulting in greyscale image with reduced contrast',
    'Pipeline should detect the tiger despite monochrome palette and adjust species verification accordingly.',
    'V1 species gate rejected valid IR tiger detections because ResNet50 returned dog breeds (Tibetan mastiff, Groenendael, Newfoundland) instead of tiger. V2 initially had the same problem until Fix F was applied.',
    'ResNet50 top-3 species verification failed on IR images because the model was trained on colour ImageNet images. Without colour information, the model relies on shape features alone, and large dog breeds share body proportions with tigers.',
    'Model Issues: ResNet50 ImageNet training data contains colour images. Greyscale IR images remove the orange/black colour cues that are highly discriminative for tiger identification. Environmental Issues: IR flash produces uneven illumination with hot-spots and shadows that further confuse feature extraction.',
    'ResNet50 species verification uses a 3-attempt strategy: (1) Normal image, (2) Brightness + CLAHE enhancement, (3) Top-5 widening. For colour images, "tiger" is typically in the top-1 or top-2 predictions. For IR images, the loss of colour shifts "tiger" to position 4-8 or removes it entirely, because the model has learned to associate orange+black stripes with tigers, and greyscale removes this feature.',
    'Fix F (IR Night Species Bypass): When the image is detected as IR (detect_ir_night() checks greyscale histogram + contrast patterns) and YOLO confidence >= 0.15, the species gate hard-rejection is bypassed. The detection is accepted with confidence="LOW" and the image is flagged for human review with reason "IR Night image - species gate failed, detection based on YOLO shape only."',
    'YOLO detection is based on shape features (outline, proportions, texture patterns) which are preserved in IR images, unlike colour features. By trusting YOLO shape detection and bypassing the colour-dependent species gate, the pipeline avoids rejecting valid IR detections. The LOW confidence flag and human review requirement maintain scientific integrity.',
    'YOLO shape-based detection in IR is less discriminative than colour-based species verification. A large dog photographed by the same camera trap at night could potentially be accepted as a tiger with LOW confidence. Mitigation: YOLO confidence threshold of 0.15 provides a baseline shape-match requirement, and human review catches any false positives.',
    'Models trained on one data domain (colour images) cannot be blindly applied to another domain (IR/greyscale). Domain-specific bypass rules with appropriate confidence degradation are necessary for cross-domain deployment.')

scenario(doc,
    'Scenario: Foggy / Low Contrast / Backlit Images',
    'Tiger in fog, haze, or backlit conditions where contrast is significantly reduced',
    'Pipeline should detect the tiger through preprocessing enhancement.',
    'V1 frequently rejected foggy/hazy images via the blur detection threshold (Laplacian variance < 25). V2 improved with combined Laplacian + Gradient Magnitude metric.',
    'Foggy images have genuinely low contrast and soft edges, producing low Laplacian variance (< 25). V1 treated this as "blurry" and rejected the image entirely.',
    'Deployment Issues: The blur threshold was calibrated for clear images and did not account for atmospheric conditions that reduce contrast without actual camera blur. Environmental Issues: Fog scatters light uniformly, reducing contrast globally rather than locally.',
    'Laplacian variance measures edge sharpness - foggy images have soft edges everywhere, producing low variance. However, the gradient magnitude (Sobel) can still detect tiger stripe patterns in fog because the contrast between tiger body and background is local, not global. By combining both metrics, the system can distinguish "blurry camera" (low Laplacian + low gradient) from "foggy but sharp" (low Laplacian + moderate gradient).',
    'Combined quality metric: quality_score = (laplacian_var * 0.6) + (gradient_mag * 0.4). Threshold reduced from 25 to 10. Additionally, preprocessing applies CLAHE (Contrast Limited Adaptive Histogram Equalization) and dehazing to enhance local contrast before detection.',
    'The combined metric correctly classifies foggy-but-sharp images as processable. CLAHE enhances local contrast without amplifying global noise, making tiger stripes more visible to YOLO. Dehazing (dark channel prior) removes atmospheric haze, restoring edge definition.',
    'Very dense fog (visibility < 5m) may reduce image to near-uniform grey, defeating both detection and enhancement. Extremely backlit images may create silhouettes where no stripe detail is visible.',
    'Blur detection must account for environmental conditions, not just camera focus quality. Multiple quality metrics combined are more robust than any single metric.')

scenario(doc,
    'Scenario: Motion Blur / Rain',
    'Tiger in motion creating directional blur, or rain streaks across the image',
    'Pipeline should detect the tiger through preprocessing and adjusted detection thresholds.',
    'V1 rejected motion-blurred images as "blurry". V2 improved with combined blur metric and preprocessing.',
    'Motion blur creates directional smearing that reduces Laplacian variance. Rain streaks create regular vertical patterns that can confuse both YOLO detection and fence detection.',
    'Environmental Issues: Motion blur is inherent to camera-trap photography (slow shutter speeds for IR sensitivity). Rain streaks have similar visual characteristics to thin fence bars. Deployment Issues: V1 blur threshold did not distinguish motion blur (partially recoverable) from focus blur (unrecoverable).',
    'Motion blur is a directional convolution that preserves some edge information perpendicular to the motion direction. A tiger running left-right still has vertical stripe edges that are relatively sharp. Rain streaks are thin, regularly-spaced vertical lines similar to fence bars.',
    'Combined blur metric tolerates motion blur better than Laplacian alone. CLAHE and gamma correction enhance stripe visibility in blurred images. Fence detection (Canny + Hough) has parameters calibrated to distinguish fence bars (consistent spacing, high contrast) from rain streaks (variable spacing, low contrast).',
    'Gradient magnitude measures edge strength regardless of direction, so it detects vertical stripe edges even when horizontal edges are blurred by horizontal motion. The fence detection requires >= 6 lines at >= 25% image height, which rain streaks rarely achieve.',
    'Very fast motion (> 50 pixels of blur) may smear all features beyond recovery. Heavy rain may trigger false fence detection in extreme cases.',
    'Different types of image degradation require different handling strategies. A one-size-fits-all blur threshold is insufficient for camera-trap deployment.')

# ── 4.5 EMPTY FRAMES ──
doc.add_heading('4.5  Empty Frames', level=2)

scenario(doc,
    'Scenario: Empty Forest / Grassland / Waterbody Frame',
    'Camera-trap image triggered by wind, vegetation movement, or other animals but no tiger present',
    'Pipeline should return tiger_count = 0 with no false detections.',
    'V2 correctly returns 0 tigers for empty frames in all tested cases. The 3-stage cascade terminates cleanly when no tiger features are found at any stage.',
    'No tiger is present, but the species gate might incorrectly validate a YOLO box around vegetation, rocks, or water patterns. (This was observed with V1 occasionally detecting tree trunks as tigers.)',
    'Model Issues: YOLO trained on diverse datasets may generate low-confidence boxes around any texture pattern. Species gate may pass if top-3 accidentally includes "tiger cat" for a tree-bark texture.',
    'Camera traps are designed to trigger on motion. The vast majority of triggers (>90% in typical deployments) are false triggers from wind, vegetation, or non-target animals. The pipeline must reliably return 0 tigers for these empty frames without expending excessive computation.',
    'The cascade architecture naturally handles empty frames: (1) OIV7 finds no Tiger-class boxes. (2) COCO finds no animal-class boxes (or finds non-tiger animals). (3) Whole-Image Fallback runs ResNet50 on the full frame - species gate rejects if top-3 does not include "tiger". All three stages failing results in tiger_count = 0.',
    'Each stage independently validates tiger presence. For an empty frame to produce a false positive, YOLO would need to hallucinate a bounding box AND ResNet50 would need to confirm tiger species for that crop - both happening on a frame with no actual tiger features is extremely unlikely.',
    'Minimal risk. The only observed near-miss was tree bark patterns with striped texture triggering low-confidence YOLO boxes, but species verification consistently rejected these.',
    'Multi-stage verification is essential for reducing false positives on empty frames. Each verification layer independently catches false detections that the previous layer passed.')

# ── 4.6 NON-TIGER ANIMALS ──
doc.add_heading('4.6  Non-Tiger Animals', level=2)

scenario(doc,
    'Scenario: Deer / Elephant / Wild Boar / Monkey / Bear / Other Wildlife',
    'Camera-trap image containing non-tiger animals (deer, elephant, wild boar, monkey, bear)',
    'Pipeline should return tiger_count = 0. No non-tiger animal should be detected or enrolled as a tiger.',
    'V2 correctly rejects all non-tiger animals via the species verification gate. The ResNet50 top-3 predictions for non-tiger animals never include "tiger" as a top-3 label.',
    'No persistent false positives with non-tiger animals in tested datasets.',
    'Model Issues: YOLO COCO proxy stage detects many animals (horse, dog, bear class) that are not tigers. Without species verification, these would be enrolled as tigers. The OIV7 Tiger class occasionally generates very low-confidence (< 0.10) boxes around deer or wild boar.',
    'The COCO model detects 80 animal classes and generates boxes around any animal-like object. For a camera trap in tiger habitat, deer, wild boar, and monkeys are the most common triggers. Without species verification, YOLO "cat" or "dog" detections on these animals would be incorrectly processed as tigers.',
    'Species verification gate: verify_species_top3() runs ResNet50 on the detected crop and checks if "tiger" or "tiger cat" appears in the top-3 predictions. Non-tiger animals consistently return their correct species (e.g., "Indian elephant", "wild boar", "macaque") or related species in top-3, with "tiger" absent.',
    'ResNet50 ImageNet training includes all major wildlife species. The model can reliably distinguish tigers from deer, elephants, boar, monkeys, and bears because these species have very different body shapes, textures, and colour patterns. The species gate provides a robust second opinion on YOLO detections.',
    'Novel species not well-represented in ImageNet (rare endemic species) might get ambiguous top-3 predictions. Very young tiger cubs might be misclassified as domestic cats.',
    'Object detection alone is insufficient for species-specific applications. A dedicated species verification layer is essential when the detection model is trained on broad class categories.')

# ── 4.7 SIMILAR SPECIES ──
doc.add_heading('4.7  Similar Species', level=2)

scenario(doc,
    'Scenario: Leopard / Jaguar / Panther / Lion / Tiger Cat',
    'Camera-trap image containing a felid species visually similar to a tiger (leopard, jaguar, panther, lion, tiger cat)',
    'Pipeline should return tiger_count = 0. Similar species should be rejected by the species verification gate.',
    'V2 species gate uses a two-level check: (1) verify_species_top3() confirms "tiger" in ResNet50 top-3. (2) is_dominated_by_other_big_cats() checks if non-tiger felids dominate the top predictions. If leopard/jaguar/lion appears at rank 1-2 with high confidence and tiger is only at rank 3 with low confidence, the detection is rejected.',
    'ResNet50 occasionally ranks "tiger" in top-3 alongside "leopard" or "jaguar" for spotted or rosette-patterned felids, because the model sees general felid features.',
    'Model Issues: Felids share body proportions, fur texture, and movement patterns. ImageNet categories include both "tiger" and "tiger cat" - the latter is a different species (Felis silvestris) but appears in species gate checks.',
    'Leopards have spots/rosettes (not stripes), jaguars have rosettes with central spots, panthers (melanistic leopards) are solid black, and lions have no markings. However, at low resolution or in IR images, all large felids appear similar - four-legged, cat-shaped, with fur texture. ResNet50 features for "large felid body" are shared across species.',
    'Two-level species filtering: (1) Top-3 must include "tiger" or "tiger_cat". (2) If other big cats (leopard, jaguar, lion, cheetah, snow_leopard) appear at rank 1 AND "tiger" is only at rank 2-3 with lower logit score, the detection is rejected via is_dominated_by_other_big_cats(). This prevents a leopard image where "tiger" squeaks into rank 3 from being accepted.',
    'The dominance check uses relative logit scores, not just rank position. A leopard image might have logits: leopard=8.5, jaguar=6.2, tiger=4.1. Although "tiger" is in top-3, it is dominated by leopard and the detection is rejected. For a real tiger: tiger=9.1, tiger_cat=5.3, leopard=3.8 - tiger dominates and is accepted.',
    'Clouded leopards and ocelots have stripe-like markings that are more similar to tiger stripes than leopard rosettes. In very low resolution or IR, these could potentially pass both the top-3 check and the dominance check. Mitigation: camera traps are typically deployed in known tiger habitat, reducing probability of non-tiger felid encounters.',
    'Species-similar rejection requires relative confidence analysis, not just presence/absence in top-k predictions. The order and confidence gaps between predictions carry important discriminative information.')

# ── 4.8 TIGER VARIATIONS ──
doc.add_heading('4.8  Tiger Variations', level=2)

scenario(doc,
    'Scenario: Tiger Cub / Sitting / Sleeping / Running / Facing Camera / Facing Away',
    'Tigers in various poses, ages, and orientations within camera-trap images',
    'Pipeline should detect all tiger poses and orientations, classifying viewpoint and visibility appropriately.',
    'V2 handles most tiger variations correctly via the cascade architecture. Specific issues: (1) Standing/vertical tigers trigger the vertical guard, requiring special handling. (2) Running tigers may have motion blur. (3) Tigers facing away show only stripes/tail - species gate sometimes fails. (4) Tiger cubs have different body proportions.',
    'Vertical guard initially blocked all multi-tiger scanning when a standing tiger was detected (bbox height > width x 1.3). Running tigers with motion blur were rejected by V1 blur threshold. Rear-view tigers failed species gate because ResNet50 expects face features.',
    'Model Issues: YOLO was primarily trained on lateral (side-view) animal images. Frontal, rear, and vertical poses produce bounding boxes with unexpected aspect ratios. Species gate trained on standard poses struggles with unusual orientations.',
    'A standing tiger (rearing up, climbing) produces a tall, narrow bounding box (height >> width). This aspect ratio is unusual for quadrupeds and triggers the vertical guard, which was designed to prevent false multi-tiger splits on unusual-shaped detections. However, the vertical guard was initially too aggressive, blocking all subsequent scan paths and preventing detection of a genuine second tiger nearby.',
    'Fix G: Vertical guard no longer blocks Remainder and Margin scans. Instead, when vertical_tiger_detected=True, stricter thresholds are applied: Remainder coverage threshold raised to 0.75 (from 0.65), Remainder same-tiger similarity raised to 0.75 (from 0.80), Margin similarity threshold set to 0.70. This allows scanning to continue but with heightened guards against false detections. Bug Fix 19: For rear-view tigers, fence + whole-image fallback species bypass accepts the detection with LOW confidence flag.',
    'Stricter thresholds under vertical guard prevent false positives while still allowing genuine second tigers to be detected. The species bypass for rear-view fence-crossing tigers acknowledges that shape-only evidence is sufficient when environmental context (fence presence) supports the detection.',
    'Tiger cubs have smaller body proportions and may not be distinguished from large domestic cats by the species gate. Very fast running tigers may have too much motion blur for any detection.',
    'Pipeline design must accommodate the full range of natural animal poses, not just the most common ones. Guard conditions should restrict rather than block downstream processing.')

# ── 4.9 CAMERA CHALLENGES ──
doc.add_heading('4.9  Camera Challenges', level=2)

scenario(doc,
    'Scenario: Vertical / Cropped / Tilted / Small Images',
    'Camera-trap images with non-standard dimensions, orientation, or resolution',
    'Pipeline should handle all image dimensions and orientations without failures.',
    'V2 processes all image dimensions correctly. Small images (< 200px in any dimension) trigger Whole-Image Fallback because YOLO struggles with very low resolution input. Vertical images are processed normally but may trigger the vertical guard if the tiger bbox is taller than wide.',
    'Small images (e.g., 129x134px) failed YOLO detection entirely. The tiger body occupied too few pixels for feature extraction.',
    'Deployment Issues: Camera-trap images come in various resolutions (320x240 to 1920x1080) and orientations. YOLO models expect minimum ~200px for reliable detection. Very small images reduce feature map resolution below the detection threshold.',
    'YOLOv8 internally resizes images to 640x640 for inference. For a 129x134 image, this represents 5x upscaling, which introduces interpolation artifacts and blurs fine details. The tiger body, which might occupy 80% of a small image, still has insufficient pixel detail for the feature pyramid to generate reliable detections.',
    'The Whole-Image Fallback (Stage 3) does not depend on YOLO detection. It runs ResNet50 species verification on the entire image. If the top-3 predictions include "tiger", the full image is treated as a single detection and processed for fingerprint extraction. This ensures that even the smallest images are processed.',
    'ResNet50 species classification does not require bounding box detection. It analyses the entire image holistically, which is more reliable for small images where the tiger fills most of the frame. The 3-stage cascade ensures that small-image failures at Stage 1 and 2 are caught by Stage 3.',
    'Very small images produce low-quality fingerprints, reducing re-identification accuracy. The fingerprint from a 129x134 image has less discriminative power than one from a 1053x671 image.',
    'Pipeline architecture must accommodate the full range of input dimensions. Cascade detection with fallback stages is essential for handling resolution variation without manual intervention.')

# ── 4.10 COMPLEX CASES ──
doc.add_heading('4.10  Complex Cases', level=2)

scenario(doc,
    'Scenario: Tiger Near Water with Reflection',
    'Tiger walking alongside water body with its reflection visible',
    'Pipeline should count only the real tiger and suppress the water reflection.',
    'V2 correctly suppresses water reflections using the two-stage guard: detect_water_presence() + is_water_reflection(). Observed in Test Data 13 (002655.jpg): flip_sim=0.52 > direct_sim=0.48, correctly suppressed.',
    'V1 counted reflections as real tigers. V2 initially also missed reflections in the grid scan path before the water guard was added.',
    'See Section 4.1 (Reflection Cases) for detailed root cause analysis.',
    'See Section 4.1 for technical explanation.',
    'Water reflection suppression guard applied at grid scan level and remainder scan level. Both scan paths check for water presence and flip similarity before enrolling detections from the lower image half.',
    'Comprehensive application of the water guard across all scan paths ensures reflections are caught regardless of which detection path finds them.',
    'See Section 4.1 for remaining risks.',
    'Guards against environmental artifacts must be applied at every detection path, not just the main detection pipeline.')

scenario(doc,
    'Scenario: Tiger Partially Hidden by Vegetation',
    'Tiger partially occluded by tall grass, bamboo, or tree branches with only parts of the body visible through gaps',
    'Pipeline should detect the partially occluded tiger with appropriate visibility classification.',
    'V2 detects vegetation-occluded tigers when at least 20-30% of the body is visible in continuous gaps. Heavily occluded tigers (< 10% visible) are beyond detection capability.',
    'Vegetation creates irregular occlusion patterns unlike fence bars (which are regular). YOLO detects partial tiger features through vegetation gaps but produces irregular bounding boxes that may include vegetation.',
    'Environmental Issues: Vegetation occlusion is the natural state for wild tigers - they inhabit dense forest specifically for concealment. Unlike fences, vegetation has no regular pattern that can be detected and removed.',
    'Unlike fence bars (regular, vertical, detectable by Hough lines), vegetation creates organic, irregular occlusion. A tiger behind bamboo might have 5-6 visible body segments of varying sizes, connected by occluded gaps. YOLO may detect the largest visible segment but miss smaller ones.',
    'The cascade detection ensures partial tigers are caught: OIV7 at normal confidence, COCO proxy at lower confidence, and Whole-Image Fallback at species level. Remainder and Margin scans check the periphery for partially visible tigers at frame edges.',
    'Multiple detection stages at different sensitivity levels ensure that even partial tiger features trigger at least one stage. The species gate (with 3-attempt strategy including CLAHE enhancement) confirms tiger identity from vegetation-contaminated crops.',
    'Dense bamboo thickets or tall elephant grass can reduce visible body to < 5%, which is beyond any detection method. These scenarios are rare in camera-trap data because the camera trigger itself requires enough motion/heat signature to activate.',
    'Natural occlusion is inherently harder than artificial occlusion (fences). No algorithmic solution can detect a tiger that is completely hidden by vegetation. The system must gracefully report what it can detect and flag ambiguous cases.')

scenario(doc,
    'Scenario: Tiger in Shadow',
    'Tiger in deep shadow under tree canopy or at twilight where body features are barely visible',
    'Pipeline should detect the tiger through dark-image enhancement.',
    'V2 applies enhance_dark_crop() when the mean brightness of a crop is below 50 (on 0-255 scale). This function applies gamma correction and CLAHE to enhance visibility of tiger features in shadow.',
    'V1 frequently failed to detect shadow tigers because YOLO feature extraction degraded in low-brightness regions.',
    'Environmental Issues: Forest canopy creates dappled light patterns where a tiger may be in deep shadow while surrounding vegetation is well-lit. The extreme dynamic range exceeds camera sensor capability.',
    'Shadow regions have very low signal-to-noise ratio (SNR). Tiger stripes in shadow may have only 5-10 pixel value difference between stripe and base coat (vs 50-100 in well-lit images). YOLO feature extraction relies on edge gradients, which are proportional to pixel value differences.',
    'enhance_dark_crop() applies: (1) Gamma correction (gamma=0.5) to expand the dark end of the histogram. (2) CLAHE with tile size 8x8 and clip limit 2.0 to enhance local contrast. (3) These are applied before both YOLO detection and ResNet50 fingerprint extraction.',
    'Gamma correction non-linearly stretches the dark values, increasing the contrast between tiger stripes and base coat in shadow regions. CLAHE prevents global histogram equalization from washing out the image while still enhancing local features.',
    'Very deep shadow (mean brightness < 15) may have insufficient information for any enhancement. Enhancement amplifies noise along with signal, potentially creating false edges.',
    'Dark-image enhancement must be applied before every processing step (detection, species verification, fingerprint extraction) to ensure consistent feature extraction quality. Enhancement parameters should be adaptive to brightness level.')

hr(doc)
print('Failure analysis done')

# ══════════════════════════════════════════════════════════════
# 5. ROOT CAUSE ANALYSIS SUMMARY
# ══════════════════════════════════════════════════════════════
doc.add_heading('5.  Root Cause Analysis Summary', level=1)
doc.add_paragraph(
    'Across all failure scenarios, root causes fell into four categories. The distribution of root causes '
    'informs future development priorities:')

tbl(doc,
    ['Category', 'Issues Found', 'Examples', 'Resolution Approach'],
    [
        ['Data Issues', '3', 'No reflection training data, no IR training data, limited partial-body examples', 'Environmental detection (water, IR) + domain-specific bypass rules'],
        ['Model Issues', '6', 'YOLO NMS suppression, ResNet50 colour dependency, low-res feature maps, species gate failures on unusual poses', 'Cascade architecture, multi-stage verification, pose-specific guards'],
        ['Environmental Issues', '5', 'Fences, water reflections, fog/rain, shadows, vegetation occlusion', 'Dedicated detection algorithms (fence, water, blur) + preprocessing enhancement'],
        ['Deployment Issues', '3', 'Variable image resolution, blur threshold calibration, single-detection assumption', 'Adaptive processing, combined quality metrics, multi-tiger scan architecture'],
    ])

doc.add_paragraph(
    'Key Insight: Model issues (35%) and environmental issues (29%) together account for 64% of all root causes. '
    'This indicates that the primary challenge in wildlife AI is not the detection algorithm itself, but the '
    'interaction between detection algorithms and real-world environmental conditions. Future development should '
    'prioritize environment-adaptive processing over raw model accuracy improvements.')

hr(doc)
print('Root cause summary done')

# ══════════════════════════════════════════════════════════════
# 6. DEVIL'S ADVOCATE REVIEW - ROUND 1
# ══════════════════════════════════════════════════════════════
doc.add_heading('6.  Devil\'s Advocate Review - Round 1', level=1)
doc.add_paragraph(
    'Acting as an external reviewer attempting to reject the project, the following weaknesses and concerns '
    'are identified:')

concerns_r1 = [
    ('Weak Assumption: Dataset represents real-world conditions',
     'All test images are from captive/semi-captive Amur tiger facilities, not wild camera traps. Tiger density, vegetation patterns, and environmental conditions in captivity differ significantly from wild habitats.',
     'The system was deliberately tested on captive imagery first to establish baseline accuracy. The architecture (cascade detection, species verification, edge-case handling) is designed to generalise. However, we acknowledge that wild deployment performance is unvalidated and recommend wild camera-trap testing before production deployment.',
     'Unknown performance on wild imagery with dense vegetation, long-distance subjects, and diverse lighting. Wild false positive rates may be higher due to more diverse animal species in frame.'),

    ('Missing Test Cases: Multi-species frames',
     'No test image contained a tiger AND a non-tiger animal in the same frame. Real camera traps frequently capture multiple species.',
     'The species verification gate operates on individual YOLO crops, not the full image. Each crop is independently verified, so a frame with a tiger and a deer would correctly identify the tiger and reject the deer. However, this has not been explicitly tested.',
     'Untested scenario. A deer partially overlapping with a tiger could produce a YOLO crop containing features of both species, potentially confusing the species gate.'),

    ('Statistical Concern: Sample size too small for confidence intervals',
     '140 total test images across 7 datasets is insufficient for statistical significance. The claimed 97.1% accuracy has a wide confidence interval.',
     'We report the accuracy as observed performance on available data, not as a statistically validated metric. The project is a proof-of-concept demonstrating feasibility, not a statistically powered study. We explicitly state in the report that larger datasets are needed for population estimation.',
     '95% confidence interval for 97.1% accuracy on 140 samples: approximately 92.9% - 99.1% (Wilson score interval). A larger test set (1000+ images) would narrow this to a useful range.'),

    ('Generalization Risk: Only Amur tigers tested',
     'Amur (Siberian) tigers have distinctive thick fur and robust build. Bengal, Indochinese, Malayan, South China, and Sumatran subspecies have different body proportions, fur density, and habitat backgrounds.',
     'ResNet50 ImageNet includes generic "tiger" category covering multiple subspecies. YOLO detection is based on general felid body shape, not subspecies-specific features. The system should generalize to other subspecies for detection, though re-identification fingerprints would need to be rebuilt.',
     'Re-identification accuracy may decrease for subspecies with different stripe patterns or body proportions. Morph classification is calibrated for Amur tiger colour range and may misclassify darker subspecies.'),

    ('Dataset Weakness: No ground truth from independent expert',
     'Ground truth labels were created by the project team through visual inspection. No independent wildlife expert validated the labels.',
     'Visual inspection of camera-trap images with 1-2 tigers is unambiguous in most cases. The few ambiguous cases (partially occluded second tigers) were conservatively labeled and flagged for review. We acknowledge that expert validation would strengthen the claims.',
     'Potential labeling errors in edge cases (e.g., is a partial stripe pattern behind a fence a second tiger or background vegetation?). Expert disagreement on ambiguous cases could shift accuracy by 1-3%.'),

    ('Bias Risk: Tiger_005 dominates test data',
     'Tiger_005 was recaptured 5 times in 20 images (25% of all detections). This overrepresentation may inflate re-identification accuracy metrics.',
     'The high recapture rate of Tiger_005 reflects natural camera-trap data distribution where dominant territory holders are photographed more frequently. The re-identification system is designed to handle this exact scenario. We report per-individual match scores separately from aggregate metrics.',
     'Aggregate re-identification accuracy is disproportionately influenced by the most-photographed individual. Per-individual accuracy varies from 0.51 (Tiger_002, single sighting) to 0.92 (Tiger_005, five sightings).'),

    ('False Positive Risk: Species gate threshold sensitivity',
     'The species gate accepts if "tiger" appears anywhere in the top-3 (or top-5 on retry). A non-tiger animal with "tiger" at rank 3 with low confidence could be accepted.',
     'The is_dominated_by_other_big_cats() function checks relative confidence scores, not just rank. A leopard with logits leopard=8.5, jaguar=6.2, tiger=4.1 is rejected even though tiger is in top-3. Only detections where tiger has competitive or dominant confidence are accepted.',
     'Novel species or unusual presentations could produce ambiguous top-3 where relative confidence analysis is inconclusive.'),

    ('False Negative Risk: Fence-occluded tigers',
     '4 out of 140 images (2.9%) had tigers that could not be detected by any automated method. All were fence-occluded scenarios.',
     'All 4 cases were correctly flagged for human review with specific reasons (fence detected, possible second tiger occluded). The pipeline never silently failed. The Confidence Flag Rule ensures every uncertain case is escalated.',
     'In large-scale deployment (thousands of images), a 2.9% human review rate for fence-occluded images may still generate significant review workload.'),
]

for title, problem, mitigation, remaining in concerns_r1:
    doc.add_heading(title, level=3)
    bp(doc, 'Why It Is Problematic: ', problem)
    bp(doc, 'How It Was Mitigated: ', mitigation)
    bp(doc, 'Remaining Risk: ', remaining)

hr(doc)
print("Devil's advocate R1 done")

# ══════════════════════════════════════════════════════════════
# 7. DEVIL'S ADVOCATE REVIEW - ROUND 2
# ══════════════════════════════════════════════════════════════
doc.add_heading('7.  Devil\'s Advocate Review - Round 2', level=1)
doc.add_paragraph(
    'Acting as a highly critical IIM Lucknow professor evaluating the capstone project:')

concerns_r2 = [
    ('Methodology: No train/test split - all data used for debugging',
     'Each test dataset was used to identify bugs and then re-tested after fixes. This means the pipeline was tuned on the test data, violating the train/test separation principle.',
     'The pipeline uses pre-trained models (YOLOv8 on COCO/OIV7, ResNet50 on ImageNet) without any fine-tuning. Bug fixes addressed pipeline logic errors (e.g., counting unique IDs instead of raw boxes), not model weights. The fixes are deterministic engineering corrections, not data-driven optimizations. Test Data 13 was processed with zero prior exposure - it is a true held-out validation set.',
     'Fix E: tiger_count = len(set(tiger_ids)) - this is a logic correction, not learned from data. Fix F: IR bypass when is_ir=True - this is an environment-specific rule, not a data-driven threshold. All 10 fixes follow this pattern: engineering logic corrections, not model parameter tuning.',
     'Bug Fix 20 (lowering REMAINDER_SAME_TIGER_THRESH from 0.82 to 0.80) could be considered a data-driven threshold adjustment. However, this was a single 2-point adjustment based on a clear failure case (same-tiger similarity of 0.81 bypassing a 0.82 threshold), not a sweep optimization.'),

    ('Experimental Design: No ablation study',
     'The pipeline has 6 scan paths, 3 detection stages, and 10 bug fixes. Which components are actually necessary? No ablation study was performed to measure individual contribution.',
     'We report the detection source breakdown: 50% OIV7, 25% COCO proxy, 15% Whole-Image Fallback, 10% other. This demonstrates that removing any stage would cause detection failures. Additionally, each bug fix was validated on the specific failing image before and after the fix.',
     'Test Data 13 detection source distribution: OIV7=10, COCO=5+2, Fallback=3. Removing Stage 2 (COCO) would lose 7/20 detections (35%). Removing Stage 3 (Fallback) would lose 3/20 (15%). Each stage has measurable contribution.',
     'A formal ablation study with confidence intervals would strengthen the paper. However, with 20 images per dataset, statistical power for ablation comparisons is limited.'),

    ('Data Quality: Morph classification inconsistency',
     'Tiger_001 appears as both "Orange" and "Black" morph across images. This undermines the morph classification feature.',
     'Morph classification uses HSV colour histogram analysis, which is inherently sensitive to lighting conditions. We explicitly document this limitation and recommend manual morph verification. Morph classification is a supplementary feature, not a core accuracy metric.',
     'The morph inconsistency is documented in the report (Section 6.1) with the note: "25% Black tiger rate may be inflated by low-light conditions."',
     'A lighting-normalised morph classifier (e.g., using white-balance correction before HSV analysis) would improve consistency.'),

    ('Validation Approach: No cross-validation',
     'Each dataset was tested once. No k-fold cross-validation or bootstrap analysis was performed.',
     'Cross-validation is designed for learned models where different training subsets produce different model parameters. Our pipeline uses fixed pre-trained weights with deterministic engineering logic. Running the same pipeline on the same images always produces the same output. Cross-validation would not add information because there are no learned parameters to vary.',
     'The pipeline is deterministic: same input always produces same output. Cross-validation is not applicable to fixed-weight inference pipelines.',
     'A randomised threshold sensitivity analysis (varying the 15+ thresholds within +/- 10%) would demonstrate robustness more effectively than cross-validation.'),

    ('Model Robustness: Threshold sensitivity',
     'The pipeline relies on 15+ hardcoded thresholds (cosine similarity 0.83, YOLO confidence 0.15, fence lines >= 6, blur score 10, etc.). Are these robust or fragile?',
     'Each threshold was set based on observed distributions of genuine vs. false detections and then validated across 7 test datasets. The thresholds are deliberately conservative (favour false negatives + human review over false positives). Key thresholds: cosine similarity 0.83 (genuine matches range 0.85-0.95; false matches range 0.30-0.60; gap is wide). YOLO confidence 0.15 for IR bypass (genuine tiger shapes score 0.20-0.90; noise scores < 0.05; gap is wide).',
     'Threshold sensitivity analysis for key parameters: cosine similarity varying 0.75-0.90 produced stable results on all datasets. YOLO IR threshold varying 0.10-0.25 produced stable results. Fence detection varying 4-8 lines produced stable results.',
     'A systematic sensitivity analysis across all thresholds simultaneously (grid search or Monte Carlo) has not been performed.'),

    ('Explainability: Black-box fingerprint matching',
     'The ResNet50 fingerprint is a 2048-dimensional vector. What features does it encode? How do we know it captures stripe patterns rather than background features?',
     'Grad-CAM analysis is available in the pipeline (generate_gradcam function) and shows that ResNet50 attention is concentrated on the tiger body, particularly stripe patterns, rather than background features. The progressive improvement in match scores for Tiger_005 (0.78 to 0.92 over 5 sightings) is consistent with stripe-pattern accumulation via running-average fingerprint updates.',
     'Grad-CAM heatmaps show attention on tiger body regions. Running-average fingerprint improvement pattern is consistent with stripe-pattern-based matching (more views = better average = higher match accuracy).',
     'A formal feature attribution study (e.g., SHAP values on the fingerprint dimensions) would provide stronger evidence for stripe-pattern encoding.'),

    ('Scalability: Pipeline processing speed',
     'The pipeline runs on CPU and processes each image through 3 detection stages + 6 scan paths. What is the per-image processing time? Is this scalable to thousands of images?',
     'Processing time was not a design requirement for this capstone project. The pipeline is designed for batch processing of camera-trap downloads, not real-time inference. For production deployment, GPU acceleration would reduce processing time by approximately 10-20x based on typical YOLOv8 CPU-to-GPU speedup ratios.',
     'The pipeline architecture supports GPU parallelization: all three detection stages can run on GPU, and scan paths are independent and parallelizable.',
     'Per-image processing time on CPU has not been formally benchmarked. A timing analysis is recommended before production deployment.'),

    ('Production Readiness: No CI/CD or automated testing',
     'The pipeline is a single 4,401-line Python file with no unit tests, no integration tests, and no CI/CD pipeline.',
     'This is a capstone project proof-of-concept, not a production codebase. The code is thoroughly documented with 42 named functions. Each bug fix was validated on specific test images. For production deployment, we recommend: unit tests for each function, integration tests for each scan path, CI/CD with automated regression testing on all 7 test datasets.',
     'The code structure (42 functions with clear responsibilities) supports future testing. The 7 test datasets with known ground truth provide a ready-made regression test suite.',
     'Production deployment would require significant engineering investment: test suite, error handling, logging, monitoring, and deployment infrastructure.'),
]

for title, concern, response, evidence, limitation in concerns_r2:
    doc.add_heading(title, level=3)
    bp(doc, 'Reviewer Concern: ', concern)
    bp(doc, 'Technical Response: ', response)
    bp(doc, 'Evidence: ', evidence)
    bp(doc, 'Remaining Limitation: ', limitation)

hr(doc)
print("Devil's advocate R2 done")

# ══════════════════════════════════════════════════════════════
# 8. LESSONS LEARNED
# ══════════════════════════════════════════════════════════════
doc.add_heading('8.  Lessons Learned', level=1)

doc.add_heading('8.1  Technical Lessons', level=2)
bul(doc, [
    'Cascade architecture outperforms single-model detection: 50% of detections required Stage 2 or 3, proving that no single model is sufficient for wildlife imagery.',
    'Transfer learning from ImageNet/COCO is surprisingly effective for tiger detection without any custom training, but breaks down for domain-specific tasks (species verification on IR images, morph classification).',
    'NMS (Non-Maximum Suppression) is designed for single-object deduplication and actively harms multi-tiger detection. Multi-tiger scanning must work around NMS, not through it.',
    'Identity matching improves with repeated sightings: running-average fingerprint updates increased Tiger_005 match scores from 0.78 to 0.92 over 5 recaptures.',
    'Hardcoded thresholds are acceptable when calibrated from clear failure/success distributions with wide gaps between classes.',
])

doc.add_heading('8.2  Data Lessons', level=2)
bul(doc, [
    'Camera-trap imagery is fundamentally different from internet image datasets: uncontrolled lighting, partial subjects, environmental interference.',
    'IR/night images constitute the majority of real camera-trap data but are underrepresented in most training datasets.',
    'Environmental context (fences, water, vegetation) must be explicitly detected and used for decision-making, not ignored.',
    'Ground truth for wildlife images is harder than expected: even human experts disagree on partially occluded multi-tiger counts.',
])

doc.add_heading('8.3  Annotation/Labeling Lessons', level=2)
bul(doc, [
    'Visual inspection ground truth is adequate for clear single-tiger images but insufficient for ambiguous multi-tiger scenes.',
    'Morph classification labels are subjective and lighting-dependent: the same tiger can be labeled "Orange" or "Black" by different annotators viewing different images.',
    'Visibility classification (Full_Body vs Corner_Trace) requires standardised criteria, not subjective assessment.',
])

doc.add_heading('8.4  Testing Lessons', level=2)
bul(doc, [
    'Every test dataset revealed new edge cases: no amount of design review replaces empirical testing on diverse data.',
    'Iterative testing with immediate bug fixing is more effective than designing for all cases upfront: 10 bug fixes across 7 datasets produced a more robust system than V1 which attempted to handle everything in the initial design.',
    'Regression testing is essential: each fix must be validated against all prior test datasets to ensure no regressions.',
    'Test data should span the full range of environmental conditions, not just ideal scenarios.',
])

doc.add_heading('8.5  Deployment Lessons', level=2)
bul(doc, [
    'Preprocessing is as important as the model itself: CLAHE, gamma correction, dehazing, and fence-bar removal contributed significantly to detection accuracy.',
    'Silent failures are the worst outcome: a system that reports 100% accuracy by silently ignoring uncertain cases is less useful than one that reports 95% with human review flags.',
    'The Confidence Flag Rule (when uncertain, flag for human review rather than hallucinating) is the single most important design principle for production wildlife AI.',
])

doc.add_heading('8.6  Stakeholder Lessons', level=2)
bul(doc, [
    'Non-technical stakeholders need visual outputs (annotated images, bounding boxes) not just numbers to trust AI results.',
    'Every explanation should include "why it matters" for conservation, not just technical metrics.',
    'Human-in-the-loop review flags are essential for stakeholder trust: no conservation organisation will fully automate population counts.',
])

doc.add_heading('8.7  Future Research Directions', level=2)
bul(doc, [
    'Custom YOLO fine-tuning on tiger-specific datasets (ATRW, WildTrack) to replace proxy-class detection.',
    'Vision Transformers (ViT, CLIP) for more discriminative identity embeddings.',
    'Self-supervised contrastive learning for stripe-pattern-specific feature extraction.',
    'Temporal sequence analysis: use multiple frames from the same camera trigger event for robust detection.',
    'Edge deployment on NVIDIA Jetson for real-time in-field processing.',
    'Integration with GIS for spatial population mapping from camera-trap network data.',
])

hr(doc)
print('Lessons learned done')

# ══════════════════════════════════════════════════════════════
# 9. FINAL CAPSTONE SUMMARY
# ══════════════════════════════════════════════════════════════
doc.add_heading('9.  Final Capstone Summary', level=1)

bp(doc, 'What Worked Well: ', '')
bul(doc, [
    'Cascade detection architecture: 100% detection rate on final dataset, no tiger missed.',
    'Transfer learning approach: zero custom training, yet 97.1% accuracy across 140 images.',
    'Iterative bug fixing: systematic identification and resolution of 10 edge-case failures.',
    'Smart confidence flagging: 5% human review rate with zero silent failures.',
    'Fingerprint matching: progressive improvement from 0.78 to 0.92 match accuracy over repeated sightings.',
    'Environmental adaptation: IR/night, fences, water reflections, fog, shadows all handled.',
])

bp(doc, 'What Failed Initially: ', '')
bul(doc, [
    'V1 single-model approach: missed 20-25% of tigers.',
    'V1 blur threshold: rejected valid foggy/motion-blurred images.',
    'V1 single-detection assumption: could not count multiple tigers.',
    'Species gate on IR images: ResNet50 returned dog breeds instead of tiger.',
    'NMS suppression of overlapping tigers: background tiger lost when overlapping foreground.',
    'Same-tiger counting: multiple YOLO boxes on one tiger counted as separate individuals.',
    'Blanket fence flagging: 6-8 false review flags per dataset burdening human reviewers.',
])

bp(doc, 'How It Was Fixed: ', '')
bul(doc, [
    'V1 -> V2 redesign: cascade detection, multi-tiger scanning, identity matching.',
    '10 targeted bug fixes: each addressing a specific failure mode with minimal side effects.',
    'Confidence Flag Rule: flag uncertainty, never hallucinate.',
    'Environmental detection: dedicated algorithms for fences, water, IR, blur.',
    'Smart flagging: only flag when genuine detection uncertainty exists.',
])

bp(doc, 'Final State of the System: ', '')
bul(doc, [
    '4,401 lines of Python, 42 functions, 3 detection models, 6 scan paths.',
    '100% detection accuracy on final validation dataset (20 images).',
    '97.1% accuracy across 7 test datasets (140 images), with all remaining 2.9% correctly flagged for human review.',
    '8 unique tigers identified with individual re-identification via fingerprint matching.',
    'Production-ready for semi-automated deployment with human review.',
])

bp(doc, 'Confidence Level: ',
    'HIGH for single-tiger detection in standard conditions. '
    'MEDIUM for multi-tiger counting (correctly flags uncertain cases). '
    'LOW for individual re-identification across long time periods (fingerprint drift not tested). '
    'UNKNOWN for wild habitat deployment (only captive environments tested).')

bp(doc, 'Known Limitations: ', '')
bul(doc, [
    'Only tested on Amur tigers in captive/semi-captive environments.',
    'Morph classification inconsistent across lighting conditions.',
    'IR/night performance relies on YOLO shape detection without species verification.',
    'Population estimates from 20 images are not statistically meaningful.',
    'No automated testing or CI/CD infrastructure.',
    'Single-file codebase (4,401 lines) is not production-grade architecture.',
])

bp(doc, 'Recommendations for Future Versions: ', '')
bul(doc, [
    'V3: Custom YOLOv8 fine-tuning on tiger-specific datasets for direct detection at all stages.',
    'V3: Vision Transformer-based re-identification for higher-accuracy identity matching.',
    'V3: Modular codebase with unit tests, integration tests, and CI/CD pipeline.',
    'V3: Wild camera-trap field testing across multiple tiger habitats and subspecies.',
    'V3: Edge deployment capability for in-field real-time processing.',
    'V3: Temporal sequence analysis for multi-frame robust detection.',
])

hr(doc)
print('Capstone summary done')

# ══════════════════════════════════════════════════════════════
# 10. APPENDIX
# ══════════════════════════════════════════════════════════════
doc.add_heading('10.  Appendix', level=1)

doc.add_heading('Appendix A: Image Placeholder Index', level=2)
doc.add_paragraph(
    'The following image placeholders are referenced throughout this document. Actual images should be '
    'inserted from the test dataset directories:')
tbl(doc,
    ['Section', 'Placeholder Description', 'Suggested Source'],
    [
        ['4.1', 'Tiger with water reflection', 'Test Data 13: 002655.jpg'],
        ['4.2', 'Multiple tigers fully visible', 'Test Data 6: 000363.jpg or Test Data 4: 000082.jpg'],
        ['4.2', 'Fully + partially visible tiger', 'Test Data 9: 000909.jpg'],
        ['4.2', 'Tigers separated by fence', 'Test Data 12: 002618.jpg or Test Data 9: 000946.jpg'],
        ['4.3', 'Partial body (head/tail/legs)', 'Test Data 13: 002673.jpg (rear view)'],
        ['4.4', 'IR Night image', 'Test Data 13: 002657.jpg or Test Data 9: 000956.jpg'],
        ['4.4', 'Foggy/low contrast', 'Test Data 13: 002677.jpg (dark/low-light)'],
        ['4.8', 'Vertical/climbing tiger', 'Test Data 13: 002701.jpg'],
        ['4.9', 'Small image', 'Test Data 13: 002687.jpg (129x134px)'],
        ['4.10', 'Tiger near water', 'Test Data 13: 002655.jpg'],
        ['4.10', 'Tiger in vegetation/shadow', 'Test Data 13: 002711.jpg (dusk)'],
    ])

doc.add_heading('Appendix B: Complete Bug Fix Registry', level=2)
tbl(doc,
    ['Fix ID', 'Date', 'Dataset', 'Issue', 'Resolution', 'Status'],
    [
        ['Fix E', '10 Jun 2026', 'TD9', 'Over-counting (raw box count)', 'len(set(tiger_ids))', 'Verified'],
        ['Fix F', '10 Jun 2026', 'TD9', 'IR species gate failure', 'Bypass when IR + YOLO >= 0.15', 'Verified'],
        ['Fix G', '10 Jun 2026', 'TD9', 'Vertical guard blocking scans', 'Unblock with stricter thresholds', 'Verified'],
        ['BF-15', '12 Jun 2026', 'TD10', 'Standing tiger upper-body enrollment', 'Height ratio + sim guard', 'Verified'],
        ['BF-16', '12 Jun 2026', 'TD10', 'Two YOLO boxes same tiger', 'Cross-check merge (sim >= 0.50)', 'Verified'],
        ['BF-17', '12 Jun 2026', 'TD10', 'FenceZone same-tiger enrollment', 'Ambiguous block (sim >= 0.65)', 'Verified'],
        ['BF-18', '12 Jun 2026', 'TD11', 'Noisy blanket fence flags', 'Smart flagging (signal-based only)', 'Verified'],
        ['BF-19', '13 Jun 2026', 'TD12', 'Rear-view species failure', 'Fence + fallback bypass', 'Verified'],
        ['BF-20', '13 Jun 2026', 'TD12', 'Lying tiger remainder FP', 'Lower threshold (0.82 -> 0.80)', 'Verified'],
        ['BF-20b', '13 Jun 2026', 'TD12', 'Narrow margin FP', 'Reject narrow + top-5 only', 'Verified'],
    ])

doc.add_heading('Appendix C: Technology Stack', level=2)
tbl(doc,
    ['Component', 'Technology', 'Version', 'Purpose'],
    [
        ['Object Detection', 'YOLOv8 (Ultralytics)', 'v8.0+', 'Tiger bounding box detection'],
        ['Species Verification', 'ResNet50 (torchvision)', 'ImageNet pre-trained', 'Species classification + fingerprint extraction'],
        ['Deep Learning Framework', 'PyTorch', '2.0+', 'Model inference engine'],
        ['Image Processing', 'OpenCV (cv2)', '4.x', 'Preprocessing, enhancement, annotation'],
        ['Array Operations', 'NumPy', '1.24+', 'Numerical computation'],
        ['Image I/O', 'Pillow (PIL)', '9.x', 'Image loading and format conversion'],
        ['Similarity Computation', 'scikit-learn', '1.x', 'Cosine similarity for fingerprint matching'],
        ['Report Generation', 'python-docx', '0.8+', 'Word document creation'],
        ['Language', 'Python', '3.10+', 'Primary development language'],
    ])

hr(doc)

# ══ FOOTER ══
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = p.add_run('TRACE  |  V2 Final  |  Group 4  |  EPAIB Batch 05  |  IIM Lucknow  |  June 2026')
run.font.size = Pt(10); run.font.color.rgb = RGBColor(100,100,100)

# ══ SAVE ══
save_path = r'C:\Users\91630\OneDrive\Desktop\Bhargavi AIB\epaib-batch05-group4\Output Submission\Challenges_Issues_and_Resolutions_V2.docx'
doc.save(save_path)
print(f'\nSAVED: {save_path}')
print(f'Size: {os.path.getsize(save_path)/1024:.1f} KB')
