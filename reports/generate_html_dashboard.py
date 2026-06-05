"""
Generate interactive HTML dashboard for Tiger AI Ultimate v3.0 run results.
Covers both the Amur Tigers Train run (3392 images) and Test Bh run (35 images).
Usage: py -3 reports/generate_html_dashboard.py
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
from pathlib import Path
from datetime import datetime

OUTPUT_HTML = Path(__file__).parent / "Tiger_AI_Dashboard.html"
CSV_FILE    = Path(__file__).parent / "ultimate_population_report.csv"

# ── Colour palette ─────────────────────────────────────────────────────────────
GREEN_DARK   = "#1B5E20"
GREEN_MID    = "#2E7D32"
GREEN_LIGHT  = "#81C784"
GREEN_PALE   = "#E8F5E9"
ORANGE_T     = "#E65C00"
GOLD_T       = "#F9A825"
BLACK_T      = "#212121"
GREY_MID     = "#757575"
WHITE        = "#FFFFFF"

MORPH_COLORS = {
    "Orange_Standard":        "#E65C00",
    "Golden":                 "#F9A825",
    "Black_Pseudomelanistic": "#424242",
    "White":                  "#90CAF9",
    "Snow_White":             "#E3F2FD",
}

SRC_COLORS = {
    "OIV7_Direct":           GREEN_MID,
    "COCO_Proxy":            GOLD_T,
    "Whole_Image_Fallback":  "#FF8F00",
}

# ── Data — Train Run (3392 images, from pipeline output) ──────────────────────
train = {
    "total_images":      3392,
    "tiger_images":      2445,
    "no_tiger":          62,
    "unusable":          885,
    "total_detections":  2457,
    "unique_tigers":     28,
    "census_estimate":   29,
    "mean_score":        0.822,
    "oiv7":              1063,
    "coco":              31,
    "fallback":          1413,
    "blurry":            833,
    "underexposed":      52,
    "orange":            1885,
    "golden":            455,
    "black":             115,
    "white":             2,
    "score_bins":        [31, 0, 0, 1, 23, 83, 236, 372, 766, 945],
    "tiger_sightings": {
        "Tiger_001": 320, "Tiger_013": 7, "Tiger_024": 5,
        "Tiger_012": 4,   "Tiger_011": 3, "Tiger_010": 2,
        "Tiger_015": 2,   "Tiger_005": 2, "Tiger_020": 2,
        "Tiger_002": 1,   "Tiger_003": 1, "Tiger_004": 1,
        "Tiger_006": 1,   "Tiger_007": 1, "Tiger_008": 1,
        "Tiger_009": 1,   "Tiger_014": 1, "Tiger_016": 1,
        "Tiger_017": 1,   "Tiger_018": 1, "Tiger_019": 1,
        "Tiger_021": 1,   "Tiger_022": 1, "Tiger_023": 1,
        "Tiger_025": 1,   "Tiger_026": 1, "Tiger_027": 1,
        "Tiger_028": 1,
    }
}

# ── Data — Test Bh Run (35 images, from CSV) ───────────────────────────────────
df = pd.read_csv(str(CSV_FILE))

# ══════════════════════════════════════════════════════════════════════════════
# BUILD CHARTS
# ══════════════════════════════════════════════════════════════════════════════

charts_html = []

def fig_to_html(fig, div_id):
    return pio.to_html(fig, full_html=False, include_plotlyjs=False,
                       div_id=div_id, default_width="100%", default_height="100%")


# ── CHART 1: Image Quality Breakdown (Train) — Donut ─────────────────────────
fig1 = go.Figure(go.Pie(
    labels=["Tiger Found", "No Tiger", "Blurry", "Underexposed"],
    values=[train["tiger_images"], train["no_tiger"], train["blurry"], train["underexposed"]],
    hole=0.55,
    marker_colors=[GREEN_MID, GREY_MID, "#EF9A9A", "#FFCC02"],
    textinfo="label+percent",
    hovertemplate="%{label}: %{value} images<extra></extra>",
))
fig1.update_layout(
    title=dict(text="Image Quality — Train Set (3,392 images)", font_size=14, font_color=GREEN_DARK),
    annotations=[dict(text="3,392<br>images", x=0.5, y=0.5, font_size=14,
                      font_color=GREEN_DARK, showarrow=False)],
    margin=dict(t=50, b=20, l=20, r=20), height=340,
    paper_bgcolor=GREEN_PALE, plot_bgcolor=GREEN_PALE,
    legend=dict(font_size=11),
)
charts_html.append(("chart1", fig_to_html(fig1, "chart1")))

# ── CHART 2: Detection Stage Cascade — Bar ────────────────────────────────────
fig2 = go.Figure(go.Bar(
    x=["OIV7 Direct\n(Tiger class)", "COCO Proxy\n(cat/dog proxy)", "Whole-Image\nFallback"],
    y=[train["oiv7"], train["coco"], train["fallback"]],
    marker_color=[GREEN_MID, GOLD_T, ORANGE_T],
    text=[f"{train['oiv7']:,}<br>({train['oiv7']/train['total_detections']*100:.1f}%)",
          f"{train['coco']:,}<br>({train['coco']/train['total_detections']*100:.1f}%)",
          f"{train['fallback']:,}<br>({train['fallback']/train['total_detections']*100:.1f}%)"],
    textposition="outside",
    hovertemplate="%{x}: %{y} detections<extra></extra>",
))
fig2.update_layout(
    title=dict(text="Cascade Detection Stages — Train Set", font_size=14, font_color=GREEN_DARK),
    yaxis_title="Detections", xaxis_title="Detection Stage",
    margin=dict(t=50, b=20, l=50, r=20), height=340,
    paper_bgcolor=GREEN_PALE, plot_bgcolor=WHITE,
)
charts_html.append(("chart2", fig_to_html(fig2, "chart2")))

# ── CHART 3: Colour Morph Breakdown — Horizontal Bar ─────────────────────────
fig3 = go.Figure(go.Bar(
    y=["Orange Tiger", "Golden Tiger", "Black Tiger", "White Tiger"],
    x=[train["orange"], train["golden"], train["black"], train["white"]],
    orientation="h",
    marker_color=[ORANGE_T, GOLD_T, "#424242", "#90CAF9"],
    text=[f"{train['orange']:,} ({train['orange']/train['total_detections']*100:.1f}%)",
          f"{train['golden']:,} ({train['golden']/train['total_detections']*100:.1f}%)",
          f"{train['black']:,} ({train['black']/train['total_detections']*100:.1f}%)",
          f"{train['white']:,} ({train['white']/train['total_detections']*100:.1f}%)"],
    textposition="outside",
))
fig3.update_layout(
    title=dict(text="Colour Morph Distribution — Train Set", font_size=14, font_color=GREEN_DARK),
    xaxis_title="Detections", margin=dict(t=50, b=20, l=120, r=80), height=300,
    paper_bgcolor=GREEN_PALE, plot_bgcolor=WHITE,
)
charts_html.append(("chart3", fig_to_html(fig3, "chart3")))

# ── CHART 4: Cosine Similarity Score Distribution — Histogram ─────────────────
bin_labels = ["0.0–0.1","0.1–0.2","0.2–0.3","0.3–0.4","0.4–0.5",
              "0.5–0.6","0.6–0.7","0.7–0.8","0.8–0.9","0.9–1.0"]
fig4 = go.Figure(go.Bar(
    x=bin_labels,
    y=train["score_bins"],
    marker_color=[
        "#EF9A9A","#EF9A9A","#EF9A9A","#EF9A9A","#FFE082",
        "#FFE082","#A5D6A7","#66BB6A",GREEN_MID,GREEN_DARK
    ],
    text=train["score_bins"],
    textposition="outside",
    hovertemplate="Score %{x}: %{y} detections<extra></extra>",
))
fig4.add_vline(x=7.3, line_dash="dash", line_color=ORANGE_T, line_width=2,
               annotation_text="Threshold 0.83", annotation_font_color=ORANGE_T)
fig4.update_layout(
    title=dict(text="Cosine Similarity Score Distribution — All 2,457 Detections",
               font_size=14, font_color=GREEN_DARK),
    xaxis_title="Score Range", yaxis_title="Number of Detections",
    margin=dict(t=50, b=60, l=50, r=20), height=340,
    paper_bgcolor=GREEN_PALE, plot_bgcolor=WHITE,
)
charts_html.append(("chart4", fig_to_html(fig4, "chart4")))

# ── CHART 5: Tiger Sighting Counts — Log Bar ──────────────────────────────────
tigers_sorted = sorted(train["tiger_sightings"].items(), key=lambda x: -x[1])
tids   = [t[0] for t in tigers_sorted]
counts = [t[1] for t in tigers_sorted]
colors = [GREEN_DARK if c > 50 else (GREEN_MID if c > 3 else GREEN_LIGHT)
          for c in counts]

fig5 = go.Figure(go.Bar(
    x=tids, y=counts,
    marker_color=colors,
    text=counts, textposition="outside",
    hovertemplate="%{x}: %{y} sightings<extra></extra>",
))
fig5.update_layout(
    title=dict(text="Individual Tiger Sighting Counts — 28 Unique Tigers",
               font_size=14, font_color=GREEN_DARK),
    xaxis_title="Tiger ID", yaxis_title="Sightings (log scale)",
    yaxis_type="log", xaxis_tickangle=-45,
    margin=dict(t=50, b=80, l=60, r=20), height=380,
    paper_bgcolor=GREEN_PALE, plot_bgcolor=WHITE,
)
charts_html.append(("chart5", fig_to_html(fig5, "chart5")))

# ── CHART 6: Test Bh — Match Scores per Tiger ─────────────────────────────────
df_primary = df.sort_values("Match_Score", ascending=False).drop_duplicates("Image_File")
df_sorted  = df_primary.sort_values("Tiger_ID")
morph_color_list = [MORPH_COLORS.get(m, GREY_MID) for m in df_sorted["Color_Morph"]]

fig6 = go.Figure(go.Bar(
    x=df_sorted["Image_File"],
    y=df_sorted["Match_Score"],
    marker_color=morph_color_list,
    text=[f"{s:.2f}" for s in df_sorted["Match_Score"]],
    textposition="outside",
    customdata=df_sorted[["Tiger_ID","Color_Morph","Detection_Source"]].values,
    hovertemplate="<b>%{x}</b><br>Tiger: %{customdata[0]}<br>Score: %{y:.3f}"
                  "<br>Morph: %{customdata[1]}<br>Source: %{customdata[2]}<extra></extra>",
))
fig6.add_hline(y=0.83, line_dash="dash", line_color=ORANGE_T, line_width=2,
               annotation_text="Match threshold 0.83", annotation_font_color=ORANGE_T)
fig6.update_layout(
    title=dict(text="Match Scores — Test Bh (35 images)", font_size=14, font_color=GREEN_DARK),
    xaxis_title="Image", yaxis_title="Cosine Similarity Score",
    xaxis_tickangle=-45, yaxis_range=[0, 1.05],
    margin=dict(t=50, b=90, l=60, r=20), height=380,
    paper_bgcolor=GREEN_PALE, plot_bgcolor=WHITE,
)
charts_html.append(("chart6", fig_to_html(fig6, "chart6")))

# ── CHART 7: Train vs Test Bh Comparison ──────────────────────────────────────
categories = ["Total Images", "Tiger Images", "Unusable", "Unique Tigers", "Mean Score ×100"]
train_vals = [3392, 2445, 885, 28, int(0.822*100)]
test_vals  = [35,   24,   11,  9,  int(0.865*100)]

fig7 = go.Figure()
fig7.add_trace(go.Bar(name="Train (3,392)", x=categories, y=train_vals,
                      marker_color=GREEN_MID, text=train_vals, textposition="outside"))
fig7.add_trace(go.Bar(name="Test Bh (35)", x=categories, y=test_vals,
                      marker_color=ORANGE_T, text=test_vals, textposition="outside"))
fig7.update_layout(
    title=dict(text="Train Set vs Test Bh — Comparison", font_size=14, font_color=GREEN_DARK),
    barmode="group", yaxis_type="log",
    margin=dict(t=50, b=40, l=60, r=20), height=360,
    paper_bgcolor=GREEN_PALE, plot_bgcolor=WHITE,
    legend=dict(orientation="h", y=1.12),
)
charts_html.append(("chart7", fig_to_html(fig7, "chart7")))

# ── CHART 8: Detection Source — Test Bh Pie ───────────────────────────────────
src_counts = df["Detection_Source"].value_counts()
fig8 = go.Figure(go.Pie(
    labels=src_counts.index.tolist(),
    values=src_counts.values.tolist(),
    hole=0.5,
    marker_colors=[SRC_COLORS.get(s, GREY_MID) for s in src_counts.index],
    textinfo="label+percent",
    hovertemplate="%{label}: %{value}<extra></extra>",
))
fig8.update_layout(
    title=dict(text="Detection Source — Test Bh", font_size=14, font_color=GREEN_DARK),
    annotations=[dict(text="26<br>detects", x=0.5, y=0.5, font_size=13,
                      font_color=GREEN_DARK, showarrow=False)],
    margin=dict(t=50, b=20, l=20, r=20), height=320,
    paper_bgcolor=GREEN_PALE, plot_bgcolor=GREEN_PALE,
)
charts_html.append(("chart8", fig_to_html(fig8, "chart8")))


# ══════════════════════════════════════════════════════════════════════════════
# BUILD HTML PAGE
# ══════════════════════════════════════════════════════════════════════════════

charts_dict = dict(charts_html)
ts = datetime.now().strftime("%d %B %Y  %H:%M")

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Tiger AI Ultimate — Run Dashboard</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  :root {{
    --green-dark:  #1B5E20;
    --green-mid:   #2E7D32;
    --green-light: #E8F5E9;
    --orange:      #E65C00;
    --gold:        #F9A825;
    --text:        #212121;
    --grey:        #757575;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: 'Segoe UI', Arial, sans-serif;
    background: #F1F8E9;
    color: var(--text);
  }}

  /* ── HEADER ── */
  .header {{
    background: linear-gradient(135deg, var(--green-dark) 0%, #2E7D32 60%, #388E3C 100%);
    color: white;
    padding: 36px 40px 28px;
    border-bottom: 4px solid var(--gold);
  }}
  .header h1 {{ font-size: 2rem; font-weight: 700; letter-spacing: 0.5px; }}
  .header .sub {{ font-size: 1rem; opacity: 0.85; margin-top: 6px; }}
  .header .meta {{ font-size: 0.82rem; opacity: 0.65; margin-top: 10px; }}

  /* ── KPI STRIP ── */
  .kpi-strip {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 16px;
    padding: 24px 40px;
    background: var(--green-dark);
  }}
  .kpi {{
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 10px;
    padding: 16px 14px;
    text-align: center;
    color: white;
  }}
  .kpi .val {{ font-size: 1.9rem; font-weight: 700; color: var(--gold); }}
  .kpi .lbl {{ font-size: 0.75rem; opacity: 0.8; margin-top: 4px; text-transform: uppercase; letter-spacing: 0.5px; }}

  /* ── SECTION ── */
  .section {{
    padding: 30px 40px 10px;
  }}
  .section-title {{
    font-size: 1.1rem;
    font-weight: 700;
    color: var(--green-dark);
    border-left: 4px solid var(--green-mid);
    padding-left: 12px;
    margin-bottom: 18px;
    text-transform: uppercase;
    letter-spacing: 0.4px;
  }}

  /* ── GRID ── */
  .grid-2 {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
    margin-bottom: 20px;
  }}
  .grid-1 {{
    display: grid;
    grid-template-columns: 1fr;
    gap: 20px;
    margin-bottom: 20px;
  }}
  .chart-card {{
    background: white;
    border-radius: 12px;
    padding: 16px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    border: 1px solid #C8E6C9;
  }}

  /* ── OBSERVATION CARDS ── */
  .obs-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 16px;
    margin-bottom: 28px;
  }}
  .obs-card {{
    background: white;
    border-radius: 10px;
    padding: 16px 18px;
    border-left: 4px solid var(--green-mid);
    box-shadow: 0 2px 6px rgba(0,0,0,0.06);
  }}
  .obs-card.warn {{ border-left-color: var(--orange); }}
  .obs-card.gold {{ border-left-color: var(--gold); }}
  .obs-title {{ font-weight: 600; font-size: 0.9rem; color: var(--green-dark); margin-bottom: 6px; }}
  .obs-card.warn .obs-title {{ color: var(--orange); }}
  .obs-card.gold .obs-title {{ color: #795548; }}
  .obs-body {{ font-size: 0.84rem; color: var(--grey); line-height: 1.55; }}

  /* ── TABLE ── */
  .data-table {{ width: 100%; border-collapse: collapse; font-size: 0.85rem; }}
  .data-table th {{
    background: var(--green-dark); color: white;
    padding: 9px 12px; text-align: left; font-weight: 600;
  }}
  .data-table td {{ padding: 8px 12px; border-bottom: 1px solid #E8F5E9; }}
  .data-table tr:nth-child(even) {{ background: #F1F8E9; }}
  .badge {{
    display: inline-block; padding: 2px 8px; border-radius: 10px;
    font-size: 0.78rem; font-weight: 600;
  }}
  .badge-green  {{ background: #C8E6C9; color: var(--green-dark); }}
  .badge-orange {{ background: #FFE0B2; color: #E65100; }}
  .badge-gold   {{ background: #FFF9C4; color: #795548; }}
  .badge-black  {{ background: #E0E0E0; color: #212121; }}
  .badge-red    {{ background: #FFCDD2; color: #C62828; }}

  /* ── FOOTER ── */
  .footer {{
    background: var(--green-dark); color: rgba(255,255,255,0.6);
    text-align: center; padding: 18px; font-size: 0.8rem;
    margin-top: 40px;
  }}

  @media (max-width: 700px) {{
    .grid-2 {{ grid-template-columns: 1fr; }}
    .kpi-strip {{ grid-template-columns: repeat(2, 1fr); }}
    .header {{ padding: 20px; }}
    .section {{ padding: 16px 16px 6px; }}
  }}
</style>
</head>
<body>

<!-- HEADER -->
<div class="header">
  <h1>🐯  Tiger AI Ultimate — v3.0 Run Dashboard</h1>
  <div class="sub">Fused Multi-Source Pipeline &nbsp;|&nbsp; EPAIB Batch 05, Group 4 &nbsp;|&nbsp; IIM Lucknow</div>
  <div class="meta">Dataset: ATRW Amur Tigers Train (3,392) + Test Bh (35) &nbsp;|&nbsp; Generated: {ts}</div>
</div>

<!-- KPI STRIP — TRAIN -->
<div class="kpi-strip">
  <div class="kpi"><div class="val">3,392</div><div class="lbl">Train Images</div></div>
  <div class="kpi"><div class="val">2,457</div><div class="lbl">Detections</div></div>
  <div class="kpi"><div class="val">28</div><div class="lbl">Unique Tigers</div></div>
  <div class="kpi"><div class="val">29</div><div class="lbl">Census Estimate</div></div>
  <div class="kpi"><div class="val">0.822</div><div class="lbl">Mean Match Score</div></div>
  <div class="kpi"><div class="val">885</div><div class="lbl">Unusable Images</div></div>
  <div class="kpi"><div class="val">55%</div><div class="lbl">Fallback Rate</div></div>
  <div class="kpi"><div class="val">44%</div><div class="lbl">OIV7 Direct</div></div>
</div>

<!-- SECTION 1: TRAIN OVERVIEW -->
<div class="section">
  <div class="section-title">1 · Train Set — Image Quality &amp; Detection Sources</div>
  <div class="grid-2">
    <div class="chart-card">{charts_dict["chart1"]}</div>
    <div class="chart-card">{charts_dict["chart2"]}</div>
  </div>
</div>

<!-- SECTION 2: MORPHS & SCORES -->
<div class="section">
  <div class="section-title">2 · Colour Morphs &amp; Identity Match Scores</div>
  <div class="grid-2">
    <div class="chart-card">{charts_dict["chart3"]}</div>
    <div class="chart-card">{charts_dict["chart4"]}</div>
  </div>
</div>

<!-- SECTION 3: TIGER SIGHTINGS -->
<div class="section">
  <div class="section-title">3 · Individual Tiger Sighting Counts — 28 Unique Tigers</div>
  <div class="grid-1">
    <div class="chart-card">{charts_dict["chart5"]}</div>
  </div>
</div>

<!-- SECTION 4: TEST BH -->
<div class="section">
  <div class="section-title">4 · Test Bh Dataset — Prof. Mahesh Balan's Images (35 images)</div>

  <!-- KPI mini strip -->
  <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:12px;margin-bottom:18px;">
    {"".join(f'<div class="kpi" style="background:var(--green-mid);padding:12px;border-radius:8px;"><div class="val" style="font-size:1.4rem">{v}</div><div class="lbl">{l}</div></div>' for v,l in [
      ("35","Images"),("24","Tiger Found"),("9","Unique Tigers"),
      ("2","Cross-DB Matches"),("7","Unusable"),("4","No Tiger"),
    ])}
  </div>

  <div class="grid-2">
    <div class="chart-card">{charts_dict["chart6"]}</div>
    <div class="chart-card">{charts_dict["chart8"]}</div>
  </div>
</div>

<!-- SECTION 5: COMPARISON -->
<div class="section">
  <div class="section-title">5 · Train vs Test Bh Comparison</div>
  <div class="grid-1">
    <div class="chart-card">{charts_dict["chart7"]}</div>
  </div>
</div>

<!-- SECTION 6: PER-IMAGE TABLE -->
<div class="section">
  <div class="section-title">6 · Test Bh — Per-Image Detection Detail</div>
  <div class="chart-card" style="overflow-x:auto;">
    <table class="data-table">
      <thead>
        <tr>
          <th>Image</th><th>Tiger ID</th><th>Morph</th>
          <th>Visibility</th><th>Detection Source</th>
          <th>Match Score</th><th>Status</th>
        </tr>
      </thead>
      <tbody>
"""

# Skipped images
skipped_data = [
    ("000010.jpg","—","—","—","—","—","UNUSABLE — Underexposed"),
    ("000464.jpg","—","—","—","—","—","UNUSABLE — Underexposed"),
    ("000544.jpg","—","—","—","—","—","UNUSABLE — Underexposed"),
    ("001082.jpg","—","—","—","—","—","UNUSABLE — Underexposed"),
    ("000725.jpg","—","—","—","—","—","UNUSABLE — Blurry"),
    ("000973.jpg","—","—","—","—","—","UNUSABLE — Blurry"),
    ("001110.jpg","—","—","—","—","—","UNUSABLE — Blurry"),
    ("000764.jpg","—","—","—","—","—","NO TIGER — hog/wild boar"),
    ("0012q.jpeg","—","—","—","—","—","NO TIGER — IR shadow only"),
    ("0013q.jpeg","—","—","—","—","—","NO TIGER — IR shadow only"),
    ("0014q.jpeg","—","—","—","—","—","NO TIGER — fox squirrel/dhole"),
]

morph_badge = {
    "Orange_Standard":        "badge-orange",
    "Golden":                 "badge-gold",
    "Black_Pseudomelanistic": "badge-black",
    "White":                  "badge-green",
}
src_badge = {
    "OIV7_Direct":           "badge-green",
    "COCO_Proxy":            "badge-gold",
    "Whole_Image_Fallback":  "badge-orange",
}

df_sorted2 = df.sort_values("Image_File")
for _, row in df_sorted2.iterrows():
    mb  = morph_badge.get(row["Color_Morph"], "badge-green")
    sb  = src_badge.get(row["Detection_Source"], "badge-green")
    sc  = row["Match_Score"]
    scol = "color:#1B5E20;font-weight:600" if sc >= 0.83 else "color:#E65C00;font-weight:600"
    html += f"""
        <tr>
          <td>{row["Image_File"]}</td>
          <td><strong>{row["Tiger_ID"]}</strong></td>
          <td><span class="badge {mb}">{row["Color_Morph"].replace("_"," ")}</span></td>
          <td>{row["Visibility"]}</td>
          <td><span class="badge {sb}" style="font-size:0.72rem">{row["Detection_Source"].replace("_"," ")}</span></td>
          <td style="{scol}">{sc:.3f}</td>
          <td>{row["ID_Status"]}</td>
        </tr>"""

for sk in skipped_data:
    html += f"""
        <tr style="background:#FFF3E0">
          <td>{sk[0]}</td>
          <td colspan="5" style="color:#999">—</td>
          <td><span class="badge badge-red">{sk[6]}</span></td>
        </tr>"""

html += """
      </tbody>
    </table>
  </div>
</div>

<!-- SECTION 7: KEY OBSERVATIONS -->
<div class="section">
  <div class="section-title">7 · Key Observations</div>
  <div class="obs-grid">
    <div class="obs-card">
      <div class="obs-title">OIV7 Tiger Class Works</div>
      <div class="obs-body">1,063 of 2,507 processable images detected by OIV7 directly using the
      "Tiger" class — no proxy workaround. This is the biggest accuracy improvement over TRACE v2.0
      which used cat/dog/horse as tiger proxies.</div>
    </div>
    <div class="obs-card warn">
      <div class="obs-title">Fallback Carries 55% of the Load</div>
      <div class="obs-body">1,413 detections came from the whole-image fallback — more than OIV7.
      This happens because close-up tiger crops fill the entire frame and YOLO cannot detect them
      as a separate object. The fallback (Yeshvir Script B idea) is non-negotiable.</div>
    </div>
    <div class="obs-card gold">
      <div class="obs-title">Cross-Dataset Recognition at 0.84–0.95</div>
      <div class="obs-body">Tiger_001 was recognised in 15 Test Bh images using a fingerprint
      built entirely from the train set — no retraining. Mean score for these matches: 0.895.
      This validates the running average embedding approach.</div>
    </div>
    <div class="obs-card">
      <div class="obs-title">Species Gate Blocked Real False Positives</div>
      <div class="obs-body">Labels like "prison", "bulletproof vest", and "cage" confirm that
      YOLO was detecting zoo enclosure bars in some frames. The ResNet top-3 gate blocked all
      of them. Without it, phantom tiger IDs would corrupt the census.</div>
    </div>
    <div class="obs-card warn">
      <div class="obs-title">1 in 4 Images Unusable</div>
      <div class="obs-body">885 of 3,392 images (26%) were rejected due to motion blur or
      darkness. This matches real camera trap deployment rates. Our pipeline handles it
      gracefully — these images are skipped without crashing.</div>
    </div>
    <div class="obs-card gold">
      <div class="obs-title">Tiger_001 = 320 Sightings</div>
      <div class="obs-body">One dominant individual accounts for 13% of all detections.
      The 320:7 ratio (Tiger_001 vs next highest) is typical for wildlife datasets — one
      frequently-photographed individual. The DB correctly maintained a single ID throughout.</div>
    </div>
  </div>
</div>

<!-- FOOTER -->
<div class="footer">
  Tiger AI Ultimate v3.0 &nbsp;|&nbsp; Group 4 &nbsp;|&nbsp; EPAIB Batch 05 &nbsp;|&nbsp;
  IIM Lucknow &nbsp;|&nbsp; 2026-06-02 &nbsp;|&nbsp;
  Pipeline: YOLOv8n-OIV7 → COCO-proxy → Fallback → ResNet50 Species Gate → Cosine Similarity DB
</div>

</body>
</html>"""

OUTPUT_HTML.write_text(html, encoding="utf-8")
print(f"SUCCESS: Dashboard saved -> {OUTPUT_HTML}")
