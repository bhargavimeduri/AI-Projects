"""
Generates 5 professor review HTML files for TRACE pipeline.
One file per professor — open in browser, print as PDF if needed.
"""
from pathlib import Path

OUT = Path("presentation/professor_reviews")
OUT.mkdir(exist_ok=True)

CSS = """
<style>
  body { font-family: 'Segoe UI', Arial, sans-serif; max-width: 820px; margin: 40px auto; padding: 0 24px; color: #212121; background: #fafafa; }
  .header { padding: 28px 28px 20px; border-radius: 8px 8px 0 0; }
  .header h1 { margin: 0; font-size: 24px; }
  .header p  { margin: 6px 0 0; font-size: 13px; opacity: 0.85; }
  .body { background: white; border: 1px solid #e0e0e0; border-radius: 0 0 8px 8px; padding: 28px; }
  h2 { font-size: 17px; border-bottom: 2px solid; padding-bottom: 6px; margin-top: 28px; }
  h3 { font-size: 15px; margin-top: 20px; }
  table { width: 100%; border-collapse: collapse; margin: 14px 0; font-size: 13px; }
  th { padding: 9px 13px; text-align: left; color: white; }
  td { padding: 9px 13px; border-bottom: 1px solid #eee; }
  tr:nth-child(even) td { background: #f9f9f9; }
  .verdict { padding: 14px 18px; border-radius: 6px; font-size: 14px; font-weight: bold; margin: 14px 0; }
  .green  { background: #E8F5E9; color: #2E7D32; border-left: 5px solid #2E7D32; }
  .amber  { background: #FFF8E1; color: #F57F17; border-left: 5px solid #F57F17; }
  .red    { background: #FFEBEE; color: #C62828; border-left: 5px solid #C62828; }
  .blue   { background: #E3F2FD; color: #0D47A1; border-left: 5px solid #0D47A1; }
  .gap    { background: #FFF3E0; border-left: 4px solid #E65100; padding: 10px 14px; border-radius: 4px; font-size: 13px; margin: 8px 0; }
  .strong { background: #E8F5E9; border-left: 4px solid #2E7D32; padding: 10px 14px; border-radius: 4px; font-size: 13px; margin: 8px 0; }
  .action { background: #0D2B55; color: white; padding: 16px 20px; border-radius: 6px; margin-top: 24px; font-size: 13px; }
  .action h3 { color: #90CAF9; margin-top: 0; }
  .footer { margin-top: 32px; font-size: 11px; color: #999; text-align: center; }
  ul { font-size: 13px; line-height: 2.0; }
  p  { font-size: 13px; line-height: 1.7; }
  code { background: #f5f5f5; padding: 2px 6px; border-radius: 3px; font-size: 12px; }
</style>
"""

def page(title, subtitle, header_color, content):
    return f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><title>{title}</title>{CSS}</head>
<body>
<div class="header" style="background:{header_color}; color:white;">
  <h1>{title}</h1>
  <p>{subtitle}</p>
</div>
<div class="body">
{content}
</div>
<div class="footer">
  TRACE — Tiger Recognition And Census Engine &nbsp;|&nbsp;
  EPAIB Batch 05, Group 4, IIM Lucknow &nbsp;|&nbsp;
  Code Review Session — 30 May 2026
</div>
</body></html>"""

# ══════════════════════════════════════════════════════════════════════════════
# PROF-MB
# ══════════════════════════════════════════════════════════════════════════════
mb = """
<div class="verdict amber">Overall Verdict: Technically sound foundation. Three critical gaps before final submission.</div>

<h2 style="border-color:#1A4F8A; color:#1A4F8A;">Professor Profile</h2>
<p><strong>Mahesh Balan Umaithanu, PhD</strong> — Principal Data Scientist, Walmart Global Tech India.<br>
IIT Madras PhD | Ex-PayPal, Amazon, ZS Associates | Visiting Faculty: IIM Lucknow, IIT Madras, IIM Calcutta.<br>
<em>Taught: Neural Networks, Deep Learning, Generative AI — EPAIB Batches 01–05.</em></p>

<h2 style="border-color:#1A4F8A; color:#1A4F8A;">Architecture Review</h2>
<table>
  <tr><th style="background:#0D2B55;">Component</th><th style="background:#0D2B55;">Decision</th><th style="background:#0D2B55;">Prof-MB's Assessment</th></tr>
  <tr><td>ResNet50 <code>[:-1]</code></td><td>Removes 1000-class head, keeps 2048-dim features</td><td>Correct. Feature extraction not classification. Right choice.</td></tr>
  <tr><td>CenterCrop(224) not Resize(224,224)</td><td>Preserves aspect ratio</td><td>Correct. ImageNet standard. Prevents stripe distortion.</td></tr>
  <tr><td>Cosine similarity over Euclidean</td><td>Scale-invariant identity matching</td><td>Correct. Day vs night images have different scales — cosine handles this.</td></tr>
  <tr><td>Running average embedding</td><td>DB improves with each sighting</td><td>Correct. Elegant solution for open-set recognition.</td></tr>
  <tr><td>YOLO proxy classes {15,16,17,24}</td><td>cat/dog/horse/zebra as tiger stand-ins</td><td>Acceptable interim only. Not production-grade. Fine-tune YOLO on tiger data.</td></tr>
  <tr><td>Similarity threshold 0.83</td><td>Hardcoded</td><td><strong>Not derived from data. This is a guess. Must be calibrated via ROC curve.</strong></td></tr>
</table>

<h2 style="border-color:#E65100; color:#E65100;">Critical Gaps — Non-Negotiable Before Submission</h2>

<div class="gap"><strong>Gap 1: Grad-CAM Missing</strong><br>
Your ResNet50 extracts 2048-dim embeddings from tiger crops. But is it looking at stripes or at background trees?
You do not know. Grad-CAM generates a heatmap showing which pixels in the image most influenced the embedding.
Without Grad-CAM, you cannot claim the model learned stripe patterns. <em>"A model you can't explain is a model you can't trust."</em></div>

<div class="gap"><strong>Gap 2: FNR Not Measured</strong><br>
A missed tiger is a false negative. For wildlife conservation, the False Negative Rate is your real exposure.
You have 51 tigers identified from 187 images. How many tigers did you miss? Currently unknown.
This is your most significant unresolved risk. FNR target must be &lt; 5% per project specification.</div>

<div class="gap"><strong>Gap 3: Threshold Not Calibrated</strong><br>
0.83 is not science — it is a starting point. Every fraud model built at PayPal went through threshold calibration:
plot FPR vs FNR across all thresholds, find the operating point that minimises conservation risk.
This requires ground truth labels on at least 30 images.</div>

<h2 style="border-color:#E65100; color:#E65100;">Output Analysis — Tiger_003 Red Flag</h2>
<div class="gap"><strong>Tiger_003 has 149 sightings across 187 images — this is suspicious.</strong><br>
Either: (a) the 0.83 threshold is too loose and many different tigers are collapsing into Tiger_003,
or (b) one tiger genuinely dominates the dataset. You cannot distinguish these cases without ground truth.
Investigate before presenting this number to Dr. Sowmya.</div>

<h2 style="border-color:#2E7D32; color:#2E7D32;">What Is Working Well</h2>
<div class="strong">Preprocessing chain (CLAHE + Dark Channel Prior dehazing) is correctly designed for Sundarbans conditions.</div>
<div class="strong">3-model architecture is the right decomposition. One model for everything would have failed.</div>
<div class="strong">Partial body handling (update_db=False) protects DB embedding quality. Good engineering discipline.</div>
<div class="strong">Running on CPU noted — GPU required at production scale for 1TB processing.</div>

<div class="action">
  <h3>Prof-MB's Action Items for Bhargavi</h3>
  <ul>
    <li>Implement Grad-CAM on at least 5 tiger images — show what the model is attending to</li>
    <li>Annotate 30 ground truth images to compute FNR and calibrate threshold via ROC curve</li>
    <li>Investigate Tiger_003 — is 149 sightings real or a threshold collapse?</li>
    <li>Be ready to answer: "Why ResNet50 over EfficientNet?" in the review session</li>
  </ul>
</div>
"""

# ══════════════════════════════════════════════════════════════════════════════
# PROF-SW
# ══════════════════════════════════════════════════════════════════════════════
sw = """
<div class="verdict green">Overall Verdict: Good problem-solution alignment. Methodology needs academic rigour before final submission.</div>

<h2 style="border-color:#1A4F8A; color:#1A4F8A;">Professor Profile</h2>
<p><strong>Dr. Sowmya Subramaniam</strong> — Associate Professor, IIM Lucknow.<br>
Faculty Supervisor for TRACE Capstone Project, EPAIB Batch 05, Group 4.<br>
<em>Reviewing project approach, methodology, and academic rigour.</em></p>

<h2 style="border-color:#1A4F8A; color:#1A4F8A;">Problem Statement Alignment</h2>
<table>
  <tr><th style="background:#0D2B55;">Requirement</th><th style="background:#0D2B55;">Status</th><th style="background:#0D2B55;">Evidence</th></tr>
  <tr><td>Count tigers per image</td><td>Addressed</td><td>YOLOv8 counts bounding boxes per frame — Tigers_In_Frame column in CSV</td></tr>
  <tr><td>Identify individual tigers</td><td>Addressed</td><td>ResNet50 + cosine similarity — 51 unique IDs in registry database</td></tr>
  <tr><td>Classify tiger types</td><td>Partially addressed</td><td>Color morph: rule-based HSV. Subspecies: placeholder only. Must be labelled clearly.</td></tr>
  <tr><td>No manual intervention</td><td>Addressed</td><td>Pipeline runs end-to-end. Forest officials upload → results generated automatically.</td></tr>
  <tr><td>Handle camera trap conditions</td><td>Addressed</td><td>Phase 1 triage + CLAHE + dehazing + filters handle fog, night, partial bodies.</td></tr>
  <tr><td>Scalable to 1TB</td><td>Not demonstrated</td><td>Tested on 187 images only. No benchmarking at scale.</td></tr>
</table>

<h2 style="border-color:#E65100; color:#E65100;">Methodology Concerns</h2>

<div class="gap"><strong>No Ground Truth Labels</strong><br>
The output "51 unique tigers" is a pipeline output — not a validated result.
Without manually verified labels on at least a subset of images, you cannot compute accuracy, precision, or recall.
This must be stated clearly in the presentation and in the project report.</div>

<div class="gap"><strong>Subspecies Classifier Presented as Complete</strong><br>
The Laplacian variance-based subspecies classification is a rule-based heuristic, not a trained model.
In the PPT and report, this must be clearly labelled as "Phase 3 placeholder — ML classifier planned."
Presenting it as a working classifier without this disclaimer is academically incorrect.</div>

<div class="gap"><strong>No Train/Val/Test Split Documentation</strong><br>
How was the 187-image dataset split? What was the rationale?
For a course capstone, the methodology must be documented — even if the dataset is small.</div>

<div class="gap"><strong>Tiger_003: 149 Sightings Unexplained</strong><br>
One tiger appearing in 149 images out of 187 is an outlier that requires explanation.
Either the dataset is skewed, or the threshold is collapsing identities. This needs investigation.</div>

<h2 style="border-color:#2E7D32; color:#2E7D32;">What the Committee Will Appreciate</h2>
<div class="strong">Real-world partnership with Wildlife Department, Government of West Bengal — strong industry connect.</div>
<div class="strong">Honest limitations slide in the PPT — academic integrity is demonstrated.</div>
<div class="strong">3-model architecture with clear justification for each model — good problem decomposition.</div>
<div class="strong">Preprocessing chain specifically designed for Sundarbans conditions — domain awareness shown.</div>

<h2 style="border-color:#1A4F8A; color:#1A4F8A;">Questions I Will Ask in the Review Session</h2>
<ul>
  <li>What is your evaluation plan? How will you measure success when real forest department data arrives?</li>
  <li>Why is your database showing 51 tigers but only 33 valid detections in this run?</li>
  <li>How confident are you in the subspecies classification? What is its accuracy?</li>
  <li>Walk me through one row of your CSV — explain every column.</li>
</ul>

<div class="action">
  <h3>Prof-SW's Action Items for Bhargavi</h3>
  <ul>
    <li>Manually label 30 images and compute precision/recall — even a small ground truth set is acceptable</li>
    <li>Add an evaluation plan section to the PPT: "How we will validate accuracy with real data"</li>
    <li>Clearly label subspecies as "rule-based placeholder" in PPT slide and notebook</li>
    <li>Prepare an answer for: "51 tigers in DB vs 33 detections this run — explain the difference"</li>
  </ul>
</div>
"""

# ══════════════════════════════════════════════════════════════════════════════
# PROF-LG
# ══════════════════════════════════════════════════════════════════════════════
lg = """
<div class="verdict green">Overall Verdict: Strong business alignment. Deployment readiness is the gap to close before going live.</div>

<h2 style="border-color:#1A4F8A; color:#1A4F8A;">Professor Profile</h2>
<p><strong>Professor Laxminarayanan G</strong> — Head of Data, AI & GenAI, ACL Digital.<br>
Two-time TEDx Speaker | IIM Lucknow & IIIT Bangalore Alumnus | Visiting Faculty: IIMs and IITs.<br>
<em>Taught: Machine Learning, GenAI Strategy, Enterprise AI — EPAIB Batches 01–05.</em></p>

<h2 style="border-color:#1A4F8A; color:#1A4F8A;">Business Outcome Assessment</h2>
<table>
  <tr><th style="background:#0D2B55;">Business Question</th><th style="background:#0D2B55;">Status</th><th style="background:#0D2B55;">Comment</th></tr>
  <tr><td>Does it solve the forest department's problem?</td><td>Yes</td><td>Count + identify + classify — all three deliverables addressed</td></tr>
  <tr><td>Time saved per census cycle</td><td>Not quantified</td><td>Manual process takes weeks — AI pipeline runtime not benchmarked</td></tr>
  <tr><td>Accuracy improvement over manual</td><td>Not measured</td><td>No baseline human accuracy established for comparison</td></tr>
  <tr><td>Can forest officials use it without technical help?</td><td>Not yet</td><td>Requires Python knowledge — Gradio UI not built yet</td></tr>
  <tr><td>Audit trail for government use</td><td>Partial</td><td>CSV output exists but missing timestamp and camera ID metadata</td></tr>
</table>

<h2 style="border-color:#E65100; color:#E65100;">Deployment Gaps</h2>

<div class="gap"><strong>No User Interface for Forest Officials</strong><br>
Forest officials cannot run Python scripts. The solution requires a Gradio or web-based upload interface.
Two lines of Gradio code would transform this from a developer tool to a deployable product.
<em>"Good models aren't enough — good adoption is what delivers value."</em></div>

<div class="gap"><strong>No Scalability Benchmarking</strong><br>
Pipeline tested on 187 images. Forest department has 1TB (~500,000+ images).
What is your processing time per image on CPU? On GPU?
At current CPU speed, 1TB will take days. This must be addressed in the next steps plan.</div>

<div class="gap"><strong>Missing Metadata for Government Reporting</strong><br>
The CSV output has Range and Beat fields — good start.
Missing: Timestamp (when was this image taken?), Camera ID (which camera trap?), Session ID.
Government conservation reports require this metadata for legal and policy documentation.</div>

<h2 style="border-color:#2E7D32; color:#2E7D32;">What the Business Stakeholder Will Value</h2>
<div class="strong">CSV output has tiger count, individual ID, color morph, subspecies, location — exactly what a forest official needs.</div>
<div class="strong">Persistent database across census sessions — accumulates knowledge over time. Strong enterprise design.</div>
<div class="strong">Real partnership with Government of West Bengal — this is a live business use case, not a classroom exercise.</div>
<div class="strong">Phase 1 triage rejects unusable images automatically — saves processing cost at scale.</div>

<h2 style="border-color:#1A4F8A; color:#1A4F8A;">Strategic Recommendation</h2>
<p>The pipeline is technically ready for a pilot. The next step is a <strong>10-image proof-of-concept</strong> with the forest department's actual data.
Take 10 real camera trap images from Mr. Anamitra Lahiri, run TRACE, and compare output with what the ranger manually identified.
That validation exercise is worth more than any metric computed on Google Images.</p>

<div class="action">
  <h3>Prof-LG's Action Items for Bhargavi</h3>
  <ul>
    <li>Add Gradio upload interface — 2 lines of code, transforms tool into product</li>
    <li>Benchmark processing time: how long per image on CPU? Extrapolate to 1TB.</li>
    <li>Add Timestamp and Camera_ID columns to output CSV for government reporting</li>
    <li>Quantify the business case: "Manual census takes X weeks. TRACE does it in Y hours."</li>
  </ul>
</div>
"""

# ══════════════════════════════════════════════════════════════════════════════
# PROF-PR
# ══════════════════════════════════════════════════════════════════════════════
pr = """
<div class="verdict amber">Overall Verdict: Correct architecture and problem decomposition. Data strategy is the primary risk to manage.</div>

<h2 style="border-color:#1A4F8A; color:#1A4F8A;">Professor Profile</h2>
<p><strong>Dr. V S Prakash Attili</strong> — Data Science, Analytics & Strategy.<br>
Visiting Faculty: IIM Lucknow | EPAIB Batches 01–05.<br>
<em>Taught: Data Science, Analytics, Business Strategy applications of ML.</em></p>

<h2 style="border-color:#1A4F8A; color:#1A4F8A;">Architecture Assessment</h2>

<div class="strong"><strong>3-Model Architecture is the right decision.</strong><br>
One model trying to do everything — triage + detection + identity + classification — would have failed.
The decomposition into Phase 1 (Triage), Phase 2 (Detection + Identity), Phase 3 (Classification) shows
mature problem-solving thinking. This is how production ML systems are actually built.</div>

<div class="strong"><strong>Preprocessing chain is strategically correct.</strong><br>
CLAHE + Dark Channel Prior dehazing directly addresses the two dominant camera trap failure modes:
low light (60-70% of Sundarbans images are night shots) and fog/rain scatter (monsoon conditions).
This was not an obvious design decision — it demonstrates domain awareness.</div>

<h2 style="border-color:#E65100; color:#E65100;">Primary Risk: Domain Shift</h2>

<div class="gap"><strong>Training Data vs Production Data Mismatch</strong><br>
The pipeline was built on Google Images of tigers — largely zoo photographs, wildlife documentaries,
controlled lighting. The forest department's camera traps produce images that are fundamentally different:
fixed position, motion-triggered, low resolution, varying weather, 24/7 operation.<br><br>
This is called <strong>domain shift</strong> — the model learned from one data distribution but will run on another.
A model trained on zoo tigers will not generalise well to Sundarbans camera trap images at 2AM in monsoon rain.</div>

<h2 style="border-color:#1A4F8A; color:#1A4F8A;">Filter Rate Analysis</h2>
<table>
  <tr><th style="background:#0D2B55;">Metric</th><th style="background:#0D2B55;">This Run</th><th style="background:#0D2B55;">Interpretation</th></tr>
  <tr><td>Total detections</td><td>190</td><td>YOLO found 190 bounding boxes across 187 images</td></tr>
  <tr><td>Filtered out</td><td>142 (75%)</td><td>Wrong animal class / low confidence / partial body</td></tr>
  <tr><td>Valid detections</td><td>33 (17%)</td><td>Passed all 3 filters — proceeded to identity matching</td></tr>
  <tr><td>Unusable images</td><td>15 (8%)</td><td>Rejected at Phase 1 triage — blur/overexposure</td></tr>
</table>
<p>A 75% filter rate on Google Images is expected — YOLO is seeing many non-tiger images.
On real forest department data where every image was triggered by animal motion, the filter rate should drop
to 30-40%. If it stays high on real data, the YOLO proxy class strategy is failing and fine-tuning is urgent.</p>

<h2 style="border-color:#E65100; color:#E65100;">Data Strategy Recommendations</h2>

<div class="gap"><strong>Do not wait for all 1TB before retraining</strong><br>
The moment 50 real camera trap images arrive from the forest department, use them immediately.
Annotate with Roboflow, fine-tune YOLO, rerun the pipeline. Early validation on real data is worth
more than waiting for complete data. Validate the domain fit as early as possible.</div>

<div class="gap"><strong>Dataset balance not documented</strong><br>
How many images per tiger type in your training data?
If 90% are Orange Standard tigers, your color morph classifier will be biased.
Document the class distribution and address imbalance before the final submission.</div>

<h2 style="border-color:#2E7D32; color:#2E7D32;">What Is Well-Designed</h2>
<div class="strong">Partial body handling protects database quality — a deliberate, correct engineering decision.</div>
<div class="strong">Persistent JSON database — accumulates knowledge across multiple census sessions over time.</div>
<div class="strong">Running average embedding — database improves with each tiger sighting. Intelligent design.</div>

<div class="action">
  <h3>Prof-PR's Action Items for Bhargavi</h3>
  <ul>
    <li>Document your training data composition — how many images per tiger type?</li>
    <li>Request 50 real camera trap images from Mr. Anamitra Lahiri for early domain validation</li>
    <li>Monitor filter rate when real data arrives — if still 75%, YOLO fine-tuning is urgent</li>
    <li>Add domain shift as a documented risk in the limitations section of the PPT</li>
  </ul>
</div>
"""

# ══════════════════════════════════════════════════════════════════════════════
# PROF-SS
# ══════════════════════════════════════════════════════════════════════════════
ss = """
<div class="verdict green">Overall Verdict: Pipeline runs, results are clean, notebook is shareable. Three quick wins will make this demo-ready.</div>

<h2 style="border-color:#1A4F8A; color:#1A4F8A;">Professor Profile</h2>
<p><strong>Sumit Kumar Singh</strong> — Founder, AceAI.Club (50,000+ community) & Garage Labs Technologies.<br>
Harvard Business School | IIM Lucknow MBA | IIT Delhi | ex-Microsoft | TEDx Speaker.<br>
<em>Taught: Prompt Engineering, Google Colab Working Sessions — EPAIB Batch 05.</em></p>

<h2 style="border-color:#1A4F8A; color:#1A4F8A;">Product Readiness Check</h2>
<table>
  <tr><th style="background:#0D2B55;">Question</th><th style="background:#0D2B55;">Status</th></tr>
  <tr><td>Does the code run without errors?</td><td>Yes — ran successfully on 187 images</td></tr>
  <tr><td>Does it solve the stated problem?</td><td>Yes — counts, identifies, classifies tigers</td></tr>
  <tr><td>Can a non-technical person use it?</td><td>Not yet — needs Gradio UI</td></tr>
  <tr><td>Is the Colab notebook shareable?</td><td>Yes — TRACE_Tiger_Pipeline.ipynb is ready</td></tr>
  <tr><td>Is there a visual output (image with boxes)?</td><td>No — only CSV output currently</td></tr>
  <tr><td>Can results be downloaded from Colab?</td><td>Yes — files.download() cell added</td></tr>
</table>

<h2 style="border-color:#E65100; color:#E65100;">Three Quick Wins Before the Demo</h2>

<div class="gap"><strong>Win 1: Add Bounding Box Visualisation</strong><br>
Right now the output is a CSV table. Add one cell to the notebook that draws bounding boxes on the image
with the tiger ID and color morph printed on top. <em>Forest officials understand a picture — not a DataFrame.</em>
This is your best demo moment. One image, three boxes: Tiger_003 | Orange_Standard | Bengal.
That visual alone will make the faculty review session memorable.</div>

<div class="gap"><strong>Win 2: Simplify the IMAGE_DIR instruction in Cell 3</strong><br>
Currently the instruction is too technical for a beginner. Replace with a clear, guided comment:<br>
<code># CHANGE ONLY THIS LINE — paste your Google Drive folder path below</code><br>
Team members with zero coding experience should be able to run this in under 5 minutes.</div>

<div class="gap"><strong>Win 3: Add Gradio Upload Interface</strong><br>
Two lines of code transforms a notebook into a product:<br>
<code>import gradio as gr</code><br>
<code>gr.Interface(fn=run_pipeline, inputs="file", outputs="file").launch()</code><br>
Forest officials upload a folder → pipeline runs → CSV downloads. No Python knowledge required.
<em>"The prompt is the product. The notebook IS the interface — make it so simple anyone can run it."</em></div>

<h2 style="border-color:#2E7D32; color:#2E7D32;">What Is Working Well</h2>
<div class="strong">Notebook installs all dependencies in Cell 1 — no setup friction for team members.</div>
<div class="strong">Google Drive mount in Cell 2 — team can use their own images immediately.</div>
<div class="strong">Results CSV auto-downloaded at end — end-to-end flow complete.</div>
<div class="strong">Phase-by-phase structure with markdown headers — beginner-friendly, self-explanatory.</div>

<h2 style="border-color:#1A4F8A; color:#1A4F8A;">Demo Strategy for Sowmya Mam's Session</h2>
<ul>
  <li>Open the Colab notebook — <strong>Runtime → Run All</strong> — show it running live</li>
  <li>Show the bounding box visualisation cell — one image with tiger boxes drawn on it</li>
  <li>Open the CSV — show Tiger_003 with 149 sightings — explain the persistent database concept</li>
  <li>Show the Phase 1 triage output — one unusable image that was <em>rejected before ML ran</em></li>
</ul>

<div class="action">
  <h3>Prof-SS's Action Items for Bhargavi</h3>
  <ul>
    <li>Add bounding box visualisation cell to the Colab notebook — draw boxes on image, print tiger ID</li>
    <li>Simplify Cell 3 IMAGE_DIR instruction for zero-experience team members</li>
    <li>Add Gradio interface as the final cell — 5 lines of code, complete product demo</li>
    <li>Rehearse the demo: Run All → show visualisation → show CSV → explain one row</li>
  </ul>
</div>
"""

# ══════════════════════════════════════════════════════════════════════════════
# WRITE FILES
# ══════════════════════════════════════════════════════════════════════════════
reviews = [
    ("Prof_MB_Review.html",
     "Prof-MB Review — Mahesh Balan PhD",
     "Neural Networks & Deep Learning | IIT Madras | Walmart | PayPal",
     "#0D2B55", mb),
    ("Prof_SW_Review.html",
     "Prof-SW Review — Dr. Sowmya Subramaniam",
     "Faculty Supervisor | Associate Professor | IIM Lucknow",
     "#1A237E", sw),
    ("Prof_LG_Review.html",
     "Prof-LG Review — Laxminarayanan G",
     "Head of Data, AI & GenAI | ACL Digital | TEDx Speaker | IIM Lucknow",
     "#1B5E20", lg),
    ("Prof_PR_Review.html",
     "Prof-PR Review — Dr. V S Prakash Attili",
     "Data Science & Analytics | IIM Lucknow Visiting Faculty",
     "#4A148C", pr),
    ("Prof_SS_Review.html",
     "Prof-SS Review — Sumit Kumar Singh",
     "AceAI.Club | Garage Labs | Harvard | IIM Lucknow | ex-Microsoft | TEDx",
     "#E65100", ss),
]

for filename, title, subtitle, color, content in reviews:
    path = OUT / filename
    path.write_text(page(title, subtitle, color, content), encoding="utf-8")
    print(f"Saved: {path}")

print(f"\nAll 5 professor reviews saved to: {OUT.resolve()}")
