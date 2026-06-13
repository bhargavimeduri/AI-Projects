"""Revert change 10 — restore original time reduction text in Section 3."""

from docx import Document

INPUT = r"c:\Users\91630\Downloads\Study_Proposal_PCCF_WestBengal_Updated.docx"
OUTPUT = r"c:\Users\91630\Downloads\Study_Proposal_PCCF_WestBengal_Updated.docx"

doc = Document(INPUT)

for i, p in enumerate(doc.paragraphs):
    if "reduction in time and manual effort" in p.text and "Proof-of-concept validation on 141 images" in p.text:
        for run in p.runs:
            run.text = ""
        p.runs[0].text = (
            "Significant reduction in time and manual effort required for camera trap image analysis "
            "- estimated reduction from weeks to hours per enumeration cycle."
        )
        print(f"Change 10 reverted at paragraph {i}")
        break

doc.save(OUTPUT)
print(f"Saved to: {OUTPUT}")
