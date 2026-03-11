"""
QC Report Generator -- PDF Format
Renders quality control reports as professional PDF documents using fpdf2.
"""

import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fpdf import FPDF

logger = logging.getLogger(__name__)


class QCReportPDF(FPDF):
    """Custom PDF class for QC reports."""

    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=20)

    def header(self):
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(30, 30, 30)
        self.cell(0, 10, "CircuitSight -- QC Report", align="C", new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(0, 102, 204)
        self.set_line_width(0.5)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}  |  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", align="C")


def generate_pdf_report(
    report_data: dict,
    annotated_images: dict = None,
    output_path: str = "reports/qc_report.pdf",
    thumbnail_size: tuple = (320, 320),
) -> str:
    """
    Generate a PDF QC report.

    Args:
        report_data: Report dict from qc_report.generate_qc_report()
        annotated_images: Dict mapping image_path to annotated numpy image (optional)
        output_path: Output PDF path
        thumbnail_size: Size for image thumbnails

    Returns:
        Path to generated PDF
    """
    pdf = QCReportPDF()
    pdf.alias_nb_pages()
    pdf.add_page()

    meta = report_data.get("report_metadata", {})
    summary = report_data.get("summary", {})
    inspections = report_data.get("inspections", [])

    # -- Report metadata --
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(30, 30, 30)
    pdf.cell(0, 8, "Report Information", new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Helvetica", "", 9)
    info_items = [
        ("Batch ID", meta.get("batch_id", "--")),
        ("Operator", meta.get("operator", "auto")),
        ("Generated", meta.get("generated_at", "--")),
        ("Software", meta.get("software_version", "--")),
    ]
    for label, value in info_items:
        pdf.set_font("Helvetica", "B", 9)
        pdf.cell(35, 6, f"{label}:", new_x="RIGHT")
        pdf.set_font("Helvetica", "", 9)
        pdf.cell(0, 6, str(value), new_x="LMARGIN", new_y="NEXT")

    pdf.ln(5)

    # -- Batch verdict banner --
    verdict = summary.get("batch_verdict", "UNKNOWN")
    verdict_colors = {
        "PASS": (0, 153, 0),
        "FAIL": (204, 0, 0),
        "NEEDS_REVIEW": (255, 153, 0),
    }
    color = verdict_colors.get(verdict, (128, 128, 128))

    pdf.set_fill_color(*color)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 14, f"  BATCH VERDICT: {verdict}", fill=True, align="L", new_x="LMARGIN", new_y="NEXT")

    pdf.set_text_color(30, 30, 30)
    pdf.ln(5)

    # -- Summary statistics --
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "Inspection Summary", new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Helvetica", "", 9)
    stats = [
        ("Total inspected", str(summary.get("total_inspected", 0))),
        ("Pass rate", f"{summary.get('pass_rate', 0)}%"),
        ("Total defects", str(summary.get("total_defects_found", 0))),
        ("Avg inference time", f"{summary.get('avg_inference_time_ms', 0)} ms"),
    ]
    for label, value in stats:
        pdf.set_font("Helvetica", "B", 9)
        pdf.cell(45, 6, f"{label}:", new_x="RIGHT")
        pdf.set_font("Helvetica", "", 9)
        pdf.cell(0, 6, value, new_x="LMARGIN", new_y="NEXT")

    # Verdict breakdown
    pdf.ln(3)
    verdicts = summary.get("verdict_breakdown", {})
    pdf.set_font("Helvetica", "B", 9)
    pdf.cell(0, 6, "Verdict Breakdown:", new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Helvetica", "", 9)
    for v, count in verdicts.items():
        pdf.cell(10, 5, "", new_x="RIGHT")
        pdf.cell(0, 5, f"{v}: {count}", new_x="LMARGIN", new_y="NEXT")

    pdf.ln(5)

    # -- Individual inspections --
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "Individual Inspections", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    # Table header
    pdf.set_font("Helvetica", "B", 8)
    pdf.set_fill_color(230, 230, 230)
    col_widths = [60, 20, 20, 30, 30, 30]
    headers = ["Image", "Defects", "Verdict", "Max Conf", "Time (ms)", "Detections"]
    for w, h in zip(col_widths, headers):
        pdf.cell(w, 7, h, border=1, fill=True, align="C", new_x="RIGHT")
    pdf.ln()

    # Table rows
    pdf.set_font("Helvetica", "", 7)
    for i, insp in enumerate(inspections[:100]):  # Limit to 100 rows
        img_name = Path(insp.get("image_path", "")).name[:25]
        n_defects = str(insp.get("n_defects", 0))
        v = insp.get("verdict", "--")
        max_conf = max((d.get("confidence", 0) for d in insp.get("detections", [])), default=0)
        time_ms = str(round(insp.get("inference_time_ms", 0), 1))
        n_det = str(insp.get("n_detections", 0))

        # Row color
        vcolors = {"PASS": (240, 255, 240), "FAIL": (255, 240, 240), "NEEDS_REVIEW": (255, 250, 230)}
        rc = vcolors.get(v, (255, 255, 255))
        pdf.set_fill_color(*rc)

        row_data = [img_name, n_defects, v, f"{max_conf:.3f}", time_ms, n_det]
        for w, d in zip(col_widths, row_data):
            pdf.cell(w, 6, d, border=1, fill=True, align="C", new_x="RIGHT")
        pdf.ln()

    if len(inspections) > 100:
        pdf.set_font("Helvetica", "I", 8)
        pdf.cell(0, 6, f"... and {len(inspections) - 100} more inspections", new_x="LMARGIN", new_y="NEXT")

    # -- Annotated images (if provided) --
    if annotated_images:
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 8, "Annotated Images (Failures & Reviews)", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(3)

        tmp_dir = Path(output_path).parent / ".tmp_thumbnails"
        tmp_dir.mkdir(parents=True, exist_ok=True)

        img_count = 0
        for img_path, img in annotated_images.items():
            if img_count >= 20:  # Limit
                break

            # Save thumbnail
            thumb = cv2.resize(img, thumbnail_size)
            tmp_path = str(tmp_dir / f"thumb_{img_count}.jpg")
            cv2.imwrite(tmp_path, thumb)

            # Check space on page
            if pdf.get_y() > 220:
                pdf.add_page()

            pdf.set_font("Helvetica", "I", 7)
            pdf.cell(0, 5, Path(img_path).name, new_x="LMARGIN", new_y="NEXT")
            pdf.image(tmp_path, w=100)
            pdf.ln(3)
            img_count += 1

        # Cleanup
        shutil.rmtree(tmp_dir, ignore_errors=True)

    # -- Save PDF --
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(path))

    logger.info("PDF report saved: %s", path)
    return str(path)
