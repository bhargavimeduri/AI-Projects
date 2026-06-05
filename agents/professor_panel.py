"""
╔══════════════════════════════════════════════════════════════════╗
║   EPAIB Professor Panel — Multi-Agent Review System             ║
║   EPAIB Batch 05 | Group 4 | IIM Lucknow                        ║
║                                                                  ║
║   Architecture:                                                  ║
║     Orchestrator  → decides which professors to consult         ║
║     Professor Agents → each reads project files + responds      ║
║     Synthesis     → final panel verdict                         ║
║                                                                  ║
║   Usage:                                                         ║
║     py -3 agents/professor_panel.py                             ║
║     py -3 agents/professor_panel.py "review my pipeline"        ║
║     py -3 agents/professor_panel.py --all "explain Grad-CAM"   ║
╚══════════════════════════════════════════════════════════════════╝
"""

import anthropic
import json
import sys
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).resolve().parent.parent
COMMANDS_DIR  = Path.home() / ".claude" / "commands"
DOCS_DIR      = BASE_DIR / "docs"
SRC_DIR       = BASE_DIR / "src"

# ── Anthropic client ───────────────────────────────────────────────────────
client = anthropic.Anthropic()
MODEL  = "claude-sonnet-4-6"

# ── Professor registry ─────────────────────────────────────────────────────
PROFESSORS = {
    "Prof-MB": {
        "name"    : "Prof. Mahesh Balan Umaithanu, PhD",
        "title"   : "Deep Learning | Neural Networks | Explainable AI",
        "domain"  : "deep learning, model architecture, Grad-CAM, embeddings, FNR, YOLOv8, ResNet50",
    },
    "Prof-SW": {
        "name"    : "Dr. Sowmya Subramaniam",
        "title"   : "Faculty Supervisor | Quantitative Methods | Capstone",
        "domain"  : "capstone evaluation, ground truth, statistical validation, research methodology",
    },
    "Prof-LG": {
        "name"    : "Prof. Laxminarayanan G",
        "title"   : "AI Strategy | Enterprise GenAI | Business Outcomes",
        "domain"  : "AI strategy, business impact, enterprise deployment, ROI, scalability",
    },
    "Prof-PR": {
        "name"    : "Dr. V S Prakash Attili",
        "title"   : "Agile | Digital Transformation | IT Systems",
        "domain"  : "agile methodology, project management, system design, digital transformation",
    },
    "Prof-SS": {
        "name"    : "Prof. Sumit Kumar Singh",
        "title"   : "GenAI Tools | Prompt Engineering | Product Thinking",
        "domain"  : "GenAI, prompt engineering, product thinking, Google Colab, demo, UI",
    },
}

# ── Orchestrator system prompt ──────────────────────────────────────────────
ORCHESTRATOR_SYSTEM = """
You are an academic panel coordinator at IIM Lucknow coordinating a review
of an AI capstone project on tiger enumeration (TRACE system).

Your job: Given a question or task, decide which professors to consult.
Return ONLY a valid JSON array of professor IDs.

Professor domains:
- Prof-MB : deep learning, YOLOv8, ResNet50, Grad-CAM, embeddings, FNR, loss curves
- Prof-SW : capstone evaluation, supervisor feedback, ground truth, reproducibility
- Prof-LG : AI strategy, enterprise deployment, business impact, scalability, ROI
- Prof-PR : agile, project management, system design, IT transformation
- Prof-SS : GenAI tools, prompt engineering, product demo, Colab, UI/UX

Rules:
- Technical architecture/model questions  → always include Prof-MB
- Capstone review / submission questions  → always include Prof-SW
- Strategy / business / deployment        → always include Prof-LG
- Process / planning / agile              → include Prof-PR
- Demo / prompt / GenAI / UI             → include Prof-SS
- General review of full project          → include all five
- Limit to 3 professors unless explicitly asked for all

Return ONLY the JSON array. Example: ["Prof-MB", "Prof-SW"]
"""

# ── Load project context for professors ────────────────────────────────────
def load_project_context() -> str:
    """Read key project files so professors have full context."""
    context_parts = []

    files_to_load = [
        (DOCS_DIR / "architecture.md",      "SYSTEM ARCHITECTURE"),
        (DOCS_DIR / "problem_statement.md",  "PROBLEM STATEMENT"),
        (SRC_DIR  / "tiger_pipeline.py",     "PIPELINE CODE (first 100 lines)"),
    ]

    for filepath, label in files_to_load:
        if filepath.exists():
            text = filepath.read_text(encoding="utf-8", errors="ignore")
            if "CODE" in label:
                text = "\n".join(text.splitlines()[:100])  # limit code to 100 lines
            context_parts.append(f"=== {label} ===\n{text}\n")

    return "\n".join(context_parts) if context_parts else "Project files not found."


# ── Load professor system prompt from .md file ─────────────────────────────
def load_professor_prompt(prof_id: str) -> str:
    md_path = COMMANDS_DIR / f"{prof_id}.md"
    if md_path.exists():
        return md_path.read_text(encoding="utf-8", errors="ignore")
    # Fallback: build a minimal prompt from the registry
    info = PROFESSORS[prof_id]
    return f"""
You are {info['name']} — {info['title']}.
You are visiting faculty at IIM Lucknow teaching EPAIB Batch 05.
Your student Bhargavi Meduri has built the TRACE tiger enumeration pipeline
using YOLOv8n + ResNet50 + cosine similarity on 187 training images.
Respond from your domain expertise: {info['domain']}.
Be direct, technically precise, and give actionable feedback.
"""


# ── Orchestrator: decide which professors to consult ───────────────────────
def select_panel(question: str, force_all: bool = False) -> list[str]:
    if force_all:
        return list(PROFESSORS.keys())

    response = client.messages.create(
        model   = MODEL,
        max_tokens = 100,
        system  = ORCHESTRATOR_SYSTEM,
        messages = [{"role": "user", "content": question}]
    )
    text = response.content[0].text.strip()

    try:
        start = text.find("[")
        end   = text.rfind("]") + 1
        panel = json.loads(text[start:end])
        return [p for p in panel if p in PROFESSORS]
    except Exception:
        # Default: technical + supervisor if parsing fails
        return ["Prof-MB", "Prof-SW"]


# ── Single professor agent call ────────────────────────────────────────────
def ask_professor(prof_id: str, question: str, project_context: str,
                  prior_feedback: str = "") -> tuple[str, str]:
    """
    Call one professor agent with:
      - Their persona as system prompt
      - Project files as context
      - Prior professors' feedback (so they can respond to each other)
    Returns (prof_id, response_text)
    """
    system_prompt = load_professor_prompt(prof_id)

    user_content = f"""
PROJECT CONTEXT (for your reference):
{project_context}

{"FEEDBACK FROM COLLEAGUES SO FAR:\n" + prior_feedback if prior_feedback else ""}

QUESTION / TASK:
{question}

Respond as {PROFESSORS[prof_id]['name']}. Be specific to this project.
"""

    response = client.messages.create(
        model      = MODEL,
        max_tokens = 800,
        system     = system_prompt,
        messages   = [{"role": "user", "content": user_content}]
    )
    return prof_id, response.content[0].text.strip()


# ── Parallel professor calls ───────────────────────────────────────────────
def ask_panel_parallel(panel: list[str], question: str,
                        project_context: str) -> dict[str, str]:
    """Call all selected professors in parallel (faster)."""
    results = {}
    with ThreadPoolExecutor(max_workers=len(panel)) as executor:
        futures = {
            executor.submit(ask_professor, prof_id, question,
                            project_context): prof_id
            for prof_id in panel
        }
        for future in as_completed(futures):
            prof_id, response = future.result()
            results[prof_id] = response
    return results


# ── Sequential professor calls (each sees previous feedback) ──────────────
def ask_panel_sequential(panel: list[str], question: str,
                          project_context: str) -> dict[str, str]:
    """Call professors sequentially — each one sees prior responses."""
    results      = {}
    prior        = ""
    for prof_id in panel:
        _, response        = ask_professor(prof_id, question,
                                           project_context, prior)
        results[prof_id]   = response
        prof_name          = PROFESSORS[prof_id]["name"]
        prior             += f"\n--- {prof_name} ---\n{response}\n"
    return results


# ── Synthesis: orchestrator summarises all professor feedback ──────────────
def synthesize(question: str, panel_responses: dict[str, str]) -> str:
    feedback_text = "\n\n".join(
        f"=== {PROFESSORS[pid]['name']} ===\n{resp}"
        for pid, resp in panel_responses.items()
    )

    response = client.messages.create(
        model   = MODEL,
        max_tokens = 500,
        system  = (
            "You are an academic panel coordinator at IIM Lucknow. "
            "Synthesize professor feedback into a concise action summary. "
            "List: (1) Points of agreement, (2) Top 3 action items for the student, "
            "(3) One critical open question the student must answer."
        ),
        messages = [{
            "role": "user",
            "content": f"Question asked: {question}\n\nProfessor responses:\n{feedback_text}"
        }]
    )
    return response.content[0].text.strip()


# ── Print helpers ──────────────────────────────────────────────────────────
def divider(char="═", width=65):
    print(char * width)

def print_header():
    divider()
    print("  EPAIB PROFESSOR PANEL — MULTI-AGENT REVIEW")
    print("  TRACE Tiger Enumeration Project | Group 4 | Batch 05")
    divider()

def print_response(prof_id: str, response: str):
    info = PROFESSORS[prof_id]
    print(f"\n{'─'*65}")
    print(f"  {info['name']}")
    print(f"  {info['title']}")
    print(f"{'─'*65}")
    print(response)


# ── Main entry point ────────────────────────────────────────────────────────
def run(question: str, mode: str = "parallel", force_all: bool = False):
    print_header()
    print(f"\n  Question: {question}")
    print(f"  Mode    : {mode}")

    print("\n  [1/4] Loading project context...")
    project_context = load_project_context()

    print("  [2/4] Orchestrator selecting panel...")
    panel = select_panel(question, force_all=force_all)
    print(f"  Panel   : {', '.join(panel)}")

    print(f"  [3/4] Consulting {len(panel)} professor(s)...\n")

    if mode == "sequential":
        # Sequential: each professor sees prior feedback
        responses = ask_panel_sequential(panel, question, project_context)
    else:
        # Parallel: all professors respond independently (faster)
        responses = ask_panel_parallel(panel, question, project_context)

    # Print each professor's response in panel order
    for prof_id in panel:
        if prof_id in responses:
            print_response(prof_id, responses[prof_id])

    print(f"\n{'═'*65}")
    print("  PANEL SYNTHESIS")
    print(f"{'═'*65}\n")
    print("  [4/4] Synthesizing...")
    verdict = synthesize(question, responses)
    print(f"\n{verdict}")
    divider()


# ── CLI ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args      = [a for a in sys.argv[1:] if not a.startswith("--")]
    force_all = "--all" in sys.argv
    sequential= "--sequential" in sys.argv
    mode      = "sequential" if sequential else "parallel"

    if args:
        question = " ".join(args)
    else:
        print("\nEPAIB Professor Panel — Multi-Agent Review System")
        print("─" * 50)
        print("Flags:")
        print("  --all         consult all 5 professors")
        print("  --sequential  each professor sees prior feedback")
        print("─" * 50)
        question = input("\nAsk the panel: ").strip()
        if not question:
            question = "Review my tiger enumeration pipeline architecture"

    run(question, mode=mode, force_all=force_all)
