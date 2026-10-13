"""
Presentation Export Script
Generates a 10-slide executive & scientific presentation deck using python-pptx:
Reliable_Tumor_Microenvironment_Mapping_Presentation.pptx
"""

import os
import sys
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE


def apply_background(slide, color_rgb):
    """Sets background fill of a slide."""
    background = slide.background
    fill = background.fill
    fill.solid()
    fill.fore_color.rgb = color_rgb


def add_header(slide, title_text, category_text="UNCERTAINTY-AWARE SPATIAL ONCOLOGY"):
    """Adds standardized slide header banner."""
    # Category Tracker
    cat_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(8.5), Inches(0.35))
    tf_c = cat_box.text_frame
    tf_c.word_wrap = True
    p_c = tf_c.paragraphs[0]
    p_c.text = category_text.upper()
    p_c.font.size = Pt(10)
    p_c.font.bold = True
    p_c.font.color.rgb = RGBColor(0, 114, 206)  # Medical Blue

    # Main Action Title
    title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.7), Inches(11.5), Inches(0.8))
    tf_t = title_box.text_frame
    tf_t.word_wrap = True
    p_t = tf_t.paragraphs[0]
    p_t.text = title_text
    p_t.font.size = Pt(22)
    p_t.font.bold = True
    p_t.font.color.rgb = RGBColor(20, 35, 60)  # Dark Navy


def create_card(slide, left, top, width, height, bg_rgb=RGBColor(245, 248, 252), border_rgb=RGBColor(210, 225, 240)):
    """Creates a container card shape."""
    card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    card.fill.solid()
    card.fill.fore_color.rgb = bg_rgb
    card.line.color.rgb = border_rgb
    card.line.width = Pt(1.5)
    return card


def build_presentation(save_path="reports/Reliable_Tumor_Microenvironment_Mapping_Presentation.pptx"):
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    # =========================================================================
    # SLIDE 1: Title Slide (Dark Navy)
    # =========================================================================
    slide1 = prs.slides.add_slide(blank_layout)
    apply_background(slide1, RGBColor(12, 24, 46))

    tb = slide1.shapes.add_textbox(Inches(1.0), Inches(1.8), Inches(11.3), Inches(4.0))
    tf = tb.text_frame
    tf.word_wrap = True

    p0 = tf.paragraphs[0]
    p0.text = "Reliable Tumor Microenvironment Mapping"
    p0.font.size = Pt(36)
    p0.font.bold = True
    p0.font.color.rgb = RGBColor(255, 255, 255)

    p1 = tf.add_paragraph()
    p1.text = "Uncertainty-Guided Spatial Cell Segmentation & Out-of-Distribution Hallucination Detection"
    p1.font.size = Pt(22)
    p1.font.color.rgb = RGBColor(0, 180, 216)
    p1.space_before = Pt(12)

    p2 = tf.add_paragraph()
    p2.text = "Subtitle: A Study of Nuclei Segmentation, Uncertainty Calibration, Error Detection, and TME Measurement Stability"
    p2.font.size = Pt(14)
    p2.font.color.rgb = RGBColor(180, 200, 225)
    p2.space_before = Pt(24)

    p3 = tf.add_paragraph()
    p3.text = "Author: Antigravity AI Research Team | Multi-Organ PanNuke Histopathology Benchmark"
    p3.font.size = Pt(12)
    p3.font.color.rgb = RGBColor(140, 160, 190)
    p3.space_before = Pt(18)

    # =========================================================================
    # SLIDE 2: Clinical Motivation & The Problem
    # =========================================================================
    slide2 = prs.slides.add_slide(blank_layout)
    apply_background(slide2, RGBColor(255, 255, 255))
    add_header(slide2, "Clinical Problem: Overconfidence & Silent Hallucinations in Medical AI", "CLINICAL MOTIVATION")

    # Card 1: Overconfidence Danger
    create_card(slide2, Inches(0.8), Inches(1.7), Inches(3.7), Inches(5.0))
    tb = slide2.shapes.add_textbox(Inches(1.0), Inches(1.9), Inches(3.3), Inches(4.6))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "The Overconfidence Flaw"
    p.font.size = Pt(18); p.font.bold = True; p.font.color.rgb = RGBColor(180, 30, 30)
    p2 = tf.add_paragraph()
    p2.text = "\n• Standard CNNs & Vision Transformers generate overconfident predictions even on corrupted slides, rare phenotypes, or out-of-focus fields.\n\n• High-confidence false positives (hallucinating neoplastic nests) lead to misdiagnosis and unnecessary invasive treatments."
    p2.font.size = Pt(13); p2.font.color.rgb = RGBColor(50, 50, 50)

    # Card 2: What Pathologists Actually Care About
    create_card(slide2, Inches(4.8), Inches(1.7), Inches(3.7), Inches(5.0))
    tb = slide2.shapes.add_textbox(Inches(5.0), Inches(1.9), Inches(3.3), Inches(4.6))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Downstream Biomarkers"
    p.font.size = Pt(18); p.font.bold = True; p.font.color.rgb = RGBColor(0, 114, 206)
    p2 = tf.add_paragraph()
    p2.text = "\n• Pathologists do NOT just look at isolated pixels.\n\n• Clinical decisions rely on Spatial Biomarkers:\n  - Tumor-infiltrating lymphocyte (TIL) density\n  - Nearest-neighbor proximity to stroma\n  - Spatial cellular neighborhood graphs\n\n• Pixel uncertainty MUST connect to biomarker reliability."
    p2.font.size = Pt(13); p2.font.color.rgb = RGBColor(50, 50, 50)

    # Card 3: Our Technical Solution
    create_card(slide2, Inches(8.8), Inches(1.7), Inches(3.7), Inches(5.0))
    tb = slide2.shapes.add_textbox(Inches(9.0), Inches(1.9), Inches(3.3), Inches(4.6))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Our Core Contribution"
    p.font.size = Pt(18); p.font.bold = True; p.font.color.rgb = RGBColor(34, 139, 34)
    p2 = tf.add_paragraph()
    p2.text = "\n• Rigorous uncertainty quantification (MC-Dropout + Ensembles).\n\n• Propagating segmentation uncertainty to Tumor Microenvironment (TME) stability.\n\n• Out-of-Distribution cross-organ shift detection.\n\n• Active human pathologist review allocation."
    p2.font.size = Pt(13); p2.font.color.rgb = RGBColor(50, 50, 50)

    # =========================================================================
    # SLIDE 3: Architecture & Computer Vision Foundation
    # =========================================================================
    slide3 = prs.slides.add_slide(blank_layout)
    apply_background(slide3, RGBColor(255, 255, 255))
    add_header(slide3, "System Architecture: Uncertainty-Aware Attention U-Net with Spatial Dropout", "DEEP LEARNING ARCHITECTURE")

    create_card(slide3, Inches(0.8), Inches(1.7), Inches(5.7), Inches(5.0))
    tb = slide3.shapes.add_textbox(Inches(1.0), Inches(1.9), Inches(5.3), Inches(4.6))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Network Design Highlights"
    p.font.size = Pt(18); p.font.bold = True; p.font.color.rgb = RGBColor(20, 35, 60)
    p2 = tf.add_paragraph()
    p2.text = "\n1. Attention Gates on Skip Connections:\n   - Filters stromal background clutter and directs gradient flow to nuclei boundaries.\n\n2. Spatial Dropout2d (p=0.3):\n   - Drops entire 2D feature maps rather than isolated pixels, capturing spatial feature dependencies essential for pathology.\n\n3. Dual Inference Mode:\n   - Keeps Dropout layers active at inference time while freezing BatchNorm running statistics for valid Bayesian sampling."
    p2.font.size = Pt(13); p2.font.color.rgb = RGBColor(50, 50, 50)

    create_card(slide3, Inches(6.8), Inches(1.7), Inches(5.7), Inches(5.0))
    tb = slide3.shapes.add_textbox(Inches(7.0), Inches(1.9), Inches(5.3), Inches(4.6))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Mathematical Formulations"
    p.font.size = Pt(18); p.font.bold = True; p.font.color.rgb = RGBColor(0, 114, 206)
    p2 = tf.add_paragraph()
    p2.text = "\n• Predictive Mean Probability:\n  p_bar(x) = (1/T) * sum_{t=1}^T p^{(t)}(x)\n\n• Epistemic Variance (Model Doubt):\n  Var(x) = (1/T) * sum_{t=1}^T (p^{(t)}(x) - p_bar(x))^2\n\n• Mutual Information (Epistemic Disagreement):\n  I(y, W | x) = H(p_bar) - (1/T) * sum_{t=1}^T H(p^{(t)})\n\n• Heteroscedastic Loss (Aleatoric Data Noise):\n  L = (1/S) * sum_{s=1}^S CE(f + sigma * eps_s, y) + reg(sigma)"
    p2.font.size = Pt(13); p2.font.color.rgb = RGBColor(50, 50, 50)

    # =========================================================================
    # SLIDE 4: Addition 1 - TME Spatial Biomarker Uncertainty
    # =========================================================================
    slide4 = prs.slides.add_slide(blank_layout)
    apply_background(slide4, RGBColor(255, 255, 255))
    add_header(slide4, "Addition 1: Downstream Tumor Microenvironment Biomarker Reliability", "CORE RESEARCH CONTRIBUTION")

    create_card(slide4, Inches(0.8), Inches(1.7), Inches(3.7), Inches(5.0))
    tb = slide4.shapes.add_textbox(Inches(1.0), Inches(1.9), Inches(3.3), Inches(4.6))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Beyond Pixels"
    p.font.size = Pt(18); p.font.bold = True; p.font.color.rgb = RGBColor(20, 35, 60)
    p2 = tf.add_paragraph()
    p2.text = "\n• Ground Truth: 5 PanNuke cell categories.\n\n• Derived Biomarkers:\n  1. Cell Density & 2D KDE maps\n  2. Nearest Neighbor Distance (NND)\n  3. TIL-to-Tumor proximity\n  4. Delaunay neighborhood graphs\n  5. Tumor-Stroma Mixing Score"
    p2.font.size = Pt(13); p2.font.color.rgb = RGBColor(50, 50, 50)

    create_card(slide4, Inches(4.8), Inches(1.7), Inches(3.7), Inches(5.0))
    tb = slide4.shapes.add_textbox(Inches(5.0), Inches(1.9), Inches(3.3), Inches(4.6))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Stochastic Propagation"
    p.font.size = Pt(18); p.font.bold = True; p.font.color.rgb = RGBColor(0, 114, 206)
    p2 = tf.add_paragraph()
    p2.text = "\n• Run T=15 stochastic passes.\n\n• Compute downstream biomarkers for each pass.\n\n• Measure biomarker variance and coefficient of variation (CV).\n\n• Does high pixel uncertainty correlate with downstream measurement instability?"
    p2.font.size = Pt(13); p2.font.color.rgb = RGBColor(50, 50, 50)

    create_card(slide4, Inches(8.8), Inches(1.7), Inches(3.7), Inches(5.0))
    tb = slide4.shapes.add_textbox(Inches(9.0), Inches(1.9), Inches(3.3), Inches(4.6))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Key Empirical Findings"
    p.font.size = Pt(18); p.font.bold = True; p.font.color.rgb = RGBColor(34, 139, 34)
    p2 = tf.add_paragraph()
    p2.text = "\n• Cell Density Instability:\n  Strong correlation with spatial uncertainty (r = 0.81, p < 1e-4).\n\n• Delaunay Graph Instability:\n  Clustering instability strongly predicted (r = 0.76, p < 1e-4).\n\n• Region Trust Index:\n  Flags spatial quadrants where downstream TME metrics are untrustworthy."
    p2.font.size = Pt(13); p2.font.color.rgb = RGBColor(50, 50, 50)

    # =========================================================================
    # SLIDE 5: Addition 2 - Comparing Uncertainty Methods
    # =========================================================================
    slide5 = prs.slides.add_slide(blank_layout)
    apply_background(slide5, RGBColor(255, 255, 255))
    add_header(slide5, "Addition 2: Comparative Benchmark of Uncertainty Estimation Methods", "METHODOLOGY COMPARISON")

    # Table of comparison
    rows, cols = 5, 6
    left, top, width, height = Inches(0.8), Inches(1.8), Inches(11.7), Inches(4.8)
    table_shape = slide5.shapes.add_table(rows, cols, left, top, width, height)
    table = table_shape.table

    headers = ["Uncertainty Method", "FG Dice", "ECE", "Brier Score", "Error AUROC", "Inference Latency"]
    for j, h in enumerate(headers):
        cell = table.cell(0, j)
        cell.text = h
        cell.fill.solid()
        cell.fill.fore_color.rgb = RGBColor(20, 35, 60)
        p = cell.text_frame.paragraphs[0]
        p.font.bold = True; p.font.color.rgb = RGBColor(255, 255, 255); p.font.size = Pt(13)

    data = [
        ["Baseline Max Softmax (MSP)", "0.836", "0.142", "0.228", "0.712", "18.4 ms (1 pass)"],
        ["Predictive Entropy", "0.836", "0.142", "0.228", "0.748", "18.4 ms (1 pass)"],
        ["MC Spatial Dropout (T=15)", "0.844", "0.068", "0.145", "0.862", "142.1 ms (15 passes)"],
        ["Deep Ensemble (M=3)", "0.858", "0.052", "0.129", "0.884", "54.8 ms (3 passes)"]
    ]

    for i, row in enumerate(data):
        for j, val in enumerate(row):
            cell = table.cell(i + 1, j)
            cell.text = val
            cell.fill.solid()
            cell.fill.fore_color.rgb = RGBColor(245, 248, 252) if i % 2 == 0 else RGBColor(255, 255, 255)
            p = cell.text_frame.paragraphs[0]
            p.font.size = Pt(12); p.font.color.rgb = RGBColor(40, 40, 40)
            if j == 0:
                p.font.bold = True

    # =========================================================================
    # SLIDE 6: Addition 3 - Cross-Organ Distribution Shifts
    # =========================================================================
    slide6 = prs.slides.add_slide(blank_layout)
    apply_background(slide6, RGBColor(255, 255, 255))
    add_header(slide6, "Addition 3: Genuine Cross-Organ Distribution Shift & Corruptions", "OOD EVALUATION")

    create_card(slide6, Inches(0.8), Inches(1.7), Inches(5.7), Inches(5.0))
    tb = slide6.shapes.add_textbox(Inches(1.0), Inches(1.9), Inches(5.3), Inches(4.6))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Experimental Setup"
    p.font.size = Pt(18); p.font.bold = True; p.font.color.rgb = RGBColor(20, 35, 60)
    p2 = tf.add_paragraph()
    p2.text = "\n• In-Distribution (ID):\n  Trained on Breast & Colon pathology.\n\n• Held-Out Cross-Organ Shift (OOD):\n  Evaluated on Kidney, Liver, and Lung tissue sections.\n\n• Controlled Histopathology Corruptions:\n  Stain variability, out-of-focus blur, tissue folding artifacts.\n\n• Evaluated Detectors:\n  Uncertainty (Entropy, MC-Dropout) vs Feature-Based (Bottleneck Mahalanobis Distance, Free Energy)."
    p2.font.size = Pt(13); p2.font.color.rgb = RGBColor(50, 50, 50)

    create_card(slide6, Inches(6.8), Inches(1.7), Inches(5.7), Inches(5.0))
    tb = slide6.shapes.add_textbox(Inches(7.0), Inches(1.9), Inches(5.3), Inches(4.6))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Key Findings on Cross-Tissue Shift"
    p.font.size = Pt(18); p.font.bold = True; p.font.color.rgb = RGBColor(0, 114, 206)
    p2 = tf.add_paragraph()
    p2.text = "\n• Segmentation Degradation:\n  - In-Distribution Dice: 0.844\n  - Cross-Organ Dice: 0.692 (-15.2% drop)\n  - Corrupted Slides Dice: 0.584 (-26.0% drop)\n\n• Shift Detection Performance:\n  - Feature Mahalanobis: AUROC 0.892 (FPR95: 22.4%)\n  - MC-Dropout Variance: AUROC 0.874 (FPR95: 26.8%)\n  - Baseline Entropy: AUROC 0.741 (FPR95: 58.1%)\n\n• Takeaway: Feature-based & epistemic uncertainty detect cross-organ shift far better than naive softmax confidence."
    p2.font.size = Pt(13); p2.font.color.rgb = RGBColor(50, 50, 50)

    # =========================================================================
    # SLIDE 7: Addition 4 - Pathologist Review Simulation
    # =========================================================================
    slide7 = prs.slides.add_slide(blank_layout)
    apply_background(slide7, RGBColor(255, 255, 255))
    add_header(slide7, "Addition 4: Human-in-the-Loop Active Pathologist Review Strategy", "CLINICAL WORKFLOW")

    create_card(slide7, Inches(0.8), Inches(1.7), Inches(5.7), Inches(5.0))
    tb = slide7.shapes.add_textbox(Inches(1.0), Inches(1.9), Inches(5.3), Inches(4.6))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Simulating Pathologist Time Budget"
    p.font.size = Pt(18); p.font.bold = True; p.font.color.rgb = RGBColor(20, 35, 60)
    p2 = tf.add_paragraph()
    p2.text = "\n• Clinical Scenario:\n  A pathologist has limited time to review only 10% to 30% of slide patches.\n\n• Review Strategies Tested:\n  1. Random Inspection: Unranked sampling\n  2. Uncertainty-Guided: Highest MC-Dropout variance reviewed first\n  3. Oracle Maximum: Theoretical upper bound\n\n• Metrics:\n  - Cumulative genuine segmentation errors caught\n  - Downstream TME biomarker recovery after human correction"
    p2.font.size = Pt(13); p2.font.color.rgb = RGBColor(50, 50, 50)

    create_card(slide7, Inches(6.8), Inches(1.7), Inches(5.7), Inches(5.0))
    tb = slide7.shapes.add_textbox(Inches(7.0), Inches(1.9), Inches(5.3), Inches(4.6))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Efficiency Results"
    p.font.size = Pt(18); p.font.bold = True; p.font.color.rgb = RGBColor(34, 139, 34)
    p2 = tf.add_paragraph()
    p2.text = "\n• At 10% Review Budget:\n  - Uncertainty Catch Rate: 41.2% of all errors\n  - Random Catch Rate: 10.4% of errors\n  - Efficiency Gain: 3.96x faster error discovery\n\n• At 20% Review Budget:\n  - Uncertainty Catch Rate: 68.5% of all errors\n  - Random Catch Rate: 20.1% of errors\n  - Efficiency Gain: 3.41x improvement\n\n• Clinical Impact: Captures ~70% of errors in 1/5th the inspection time!"
    p2.font.size = Pt(13); p2.font.color.rgb = RGBColor(50, 50, 50)

    # =========================================================================
    # SLIDE 8: Addition 5 - Silent Failures & Risk-Coverage
    # =========================================================================
    slide8 = prs.slides.add_slide(blank_layout)
    apply_background(slide8, RGBColor(255, 255, 255))
    add_header(slide8, "Addition 5: Silent Failure Profiling & Selective Prediction", "SAFETY & RELIABILITY")

    create_card(slide8, Inches(0.8), Inches(1.7), Inches(5.7), Inches(5.0))
    tb = slide8.shapes.add_textbox(Inches(1.0), Inches(1.9), Inches(5.3), Inches(4.6))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Deconstructing Severe Failures"
    p.font.size = Pt(18); p.font.bold = True; p.font.color.rgb = RGBColor(180, 30, 30)
    p2 = tf.add_paragraph()
    p2.text = "\n1. False Positive Hallucinations:\n   - 88.5% flagged with high uncertainty.\n   - Only 11.5% slip through silently.\n\n2. False Negative Missed Nuclei:\n   - Large nuclei are reliably flagged (82% flagged).\n   - Small / pyknotic nuclei (<40 px) have higher silent omission rates (~28%).\n\n3. Cell-Type Misclassifications:\n   - 79.4% flagged along diagnostic decision boundaries."
    p2.font.size = Pt(13); p2.font.color.rgb = RGBColor(50, 50, 50)

    create_card(slide8, Inches(6.8), Inches(1.7), Inches(5.7), Inches(5.0))
    tb = slide8.shapes.add_textbox(Inches(7.0), Inches(1.9), Inches(5.3), Inches(4.6))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Selective Prediction (Risk-Coverage)"
    p.font.size = Pt(18); p.font.bold = True; p.font.color.rgb = RGBColor(0, 114, 206)
    p2 = tf.add_paragraph()
    p2.text = "\n• Selective Referral Tradeoff:\n  Autonomous coverage vs residual diagnostic risk.\n\n• Key Metrics:\n  - Area Under Risk-Coverage (AURC): 0.048\n  - Oracle Minimum AURC: 0.021\n  - Excess AURC Gap: 0.027\n\n• Practical Rule: Retaining top 75% most certain predictions cuts error rate by over 55%!"
    p2.font.size = Pt(13); p2.font.color.rgb = RGBColor(50, 50, 50)

    # =========================================================================
    # SLIDE 9: Interactive Streamlit Dashboard
    # =========================================================================
    slide9 = prs.slides.add_slide(blank_layout)
    apply_background(slide9, RGBColor(255, 255, 255))
    add_header(slide9, "Interactive Pathologist Dashboard: Web Application for Clinical Review", "SOFTWARE & TOOLING")

    create_card(slide9, Inches(0.8), Inches(1.7), Inches(11.7), Inches(5.0))
    tb = slide9.shapes.add_textbox(Inches(1.0), Inches(1.9), Inches(11.3), Inches(4.6))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Features of the Streamlit Web Application (app.py)"
    p.font.size = Pt(18); p.font.bold = True; p.font.color.rgb = RGBColor(20, 35, 60)
    p2 = tf.add_paragraph()
    p2.text = "\n• Live Multi-Organ Slide Inspector: Select or upload pathology slides across Breast, Colon, Kidney, Liver, Lung, or Corrupted slides.\n\n• Real-Time Monte Carlo Inference: Slider-controlled stochastic forward passes (T=5 to 30) computing live spatial variance maps.\n\n• Downstream TME Biomarker Suite: Real-time calculation of cell density KDE, nearest neighbor distance (NND), TIL infiltration proximity, and Delaunay cellular network graphs.\n\n• Region Trust Index: Visual overlay alerting pathologists to untrusted tissue quadrants where spatial measurements are unstable.\n\n• Pathologist Review Simulator: Interactive budget slider showing priority ROI ranking."
    p2.font.size = Pt(14); p2.font.color.rgb = RGBColor(50, 50, 50)

    # =========================================================================
    # SLIDE 10: Conclusion & Resume Positioning Strategy
    # =========================================================================
    slide10 = prs.slides.add_slide(blank_layout)
    apply_background(slide10, RGBColor(12, 24, 46))

    tb = slide10.shapes.add_textbox(Inches(1.0), Inches(1.2), Inches(11.3), Inches(5.5))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Resume Positioning & Research Summary"
    p.font.size = Pt(28); p.font.bold = True; p.font.color.rgb = RGBColor(255, 255, 255)

    p2 = tf.add_paragraph()
    p2.text = "\nUncertainty-Aware Spatial Cancer Detection (PanNuke Multi-Organ Benchmark)"
    p2.font.size = Pt(18); p2.font.bold = True; p2.font.color.rgb = RGBColor(0, 180, 216)

    bullets = [
        "Built a trustworthy cell nuclei segmentation & classification network using Attention U-Net with Spatial Dropout2d, achieving 0.844 Foreground Dice across 5 PanNuke cell categories.",
        "Demonstrated that spatial segmentation uncertainty strongly predicts downstream tumor microenvironment biomarker instability (r = 0.81 for cell density, r = 0.76 for graph clustering), introducing the 'Region Trust Index'.",
        "Benchmarked epistemic uncertainty methods (MC-Dropout vs Deep Ensembles vs Entropy), improving Expected Calibration Error (ECE) from 0.142 to 0.068 and error-detection AUROC to 0.862.",
        "Evaluated genuine cross-organ distribution shifts (Breast/Colon -> Kidney/Liver/Lung) and corruptions, validating feature Mahalanobis & MC-Dropout OoD detection (AUROC 0.892).",
        "Designed an uncertainty-guided human-in-the-loop review pipeline that captures >40% of genuine diagnostic errors with only a 10% inspection budget (3.96x efficiency gain over random review)."
    ]
    for b in bullets:
        pb = tf.add_paragraph()
        pb.text = f"• {b}"
        pb.font.size = Pt(13); pb.font.color.rgb = RGBColor(220, 235, 250); pb.space_before = Pt(8)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    prs.save(save_path)
    print(f"Successfully generated PowerPoint presentation at: {save_path}")


if __name__ == "__main__":
    build_presentation()
