Samajh gaya bhai 👍
Tum chahte ho ki **VS Code me paste karo → automatically proper headings, spacing, code blocks render ho → aur GitHub pe bhi clean dikhe**.

Iske liye **README.md format perfect hota hai**.
Neeche jo de raha hoon wo **proper GitHub-ready Markdown** hai.

Bas:

1. VS Code me `README.md` open karo
2. Sab delete karo
3. Neeche wala paste karo
4. Save

GitHub pe **perfect arrange ho jayega**.

---

# Document AI Evaluation Harness

## Overview

This repository contains an evaluation harness for benchmarking two document AI models:

1. **TrOCR (`microsoft/trocr-base-printed`)** — Optical Character Recognition (OCR)
2. **LayoutLMv3 (`microsoft/layoutlmv3-base`)** — Document layout understanding

The goal of this project is to measure model performance under controlled conditions and generate structured evaluation reports including:

* Inference latency
* System memory usage
* OCR text accuracy (for TrOCR)

This harness is designed to run **locally on CPU-only systems**, which is important for environments such as hospital machines where GPU availability may be limited.

---

# Evaluation Pipeline

## TrOCR Pipeline

```
Input Images
      ↓
Image Preprocessing
      ↓
TrOCR Model Inference
      ↓
Text Generation
      ↓
Comparison With Ground Truth
      ↓
Metric Computation
      ↓
trocr_report.json
```

---

## LayoutLMv3 Pipeline

LayoutLMv3 requires **text and spatial layout information**, therefore an OCR step is required before model inference.

```
Input Images
      ↓
Tesseract OCR
      ↓
Extract Words + Bounding Boxes
      ↓
Normalize Bounding Boxes (0–1000 scale)
      ↓
LayoutLMv3 Model Inference
      ↓
Metric Computation
      ↓
layoutlm_report.json
```

---

# Metrics Measured

## Latency

Latency measures the time taken for model inference.

* **p50 latency** → median inference time
* **p95 latency** → worst-case latency

Latency is measured using:

```python
time.perf_counter()
```

---

## Peak RAM Usage

Memory usage is measured using:

```python
psutil.Process().memory_info().rss
```

This captures the **actual system memory footprint**, including model weights loaded by PyTorch.

Example measurement:

```
TrOCR peak RAM ≈ 1613 MB
```

This value is important for planning deployments on machines with limited memory (for example hospital systems with 4–8 GB RAM).

---

## Text Accuracy (TrOCR Only)

Text accuracy is computed using string similarity between predicted text and ground truth.

Method used:

```python
difflib.SequenceMatcher
```

Concept:

```
Similarity = matching_characters / total_characters
```

Example:

```
Ground truth:  Patient Ravi Kumar
Prediction:    PETIENT RAVI KUMAR
Similarity ≈ 0.94
```

---

# Why LayoutLMv3 Has No `text_accuracy`

LayoutLMv3 produces **document embeddings**, not text predictions.

Therefore the file:

```
layoutlm_report.json
```

does **not contain a `text_accuracy` field**.

The LayoutLMv3 evaluation measures:

* latency
* peak RAM usage

but **not OCR accuracy**, because text extraction is handled by Tesseract OCR.

---

# Repository Structure

```
document-ai-evaluation
│
├── evaluate.py                # Main evaluation script
├── requirements.txt           # Python dependencies
│
├── images                     # Sample test images
│   ├── image1.png
│   ├── image2.png
│   └── image3.png
│
├── trocr_report.json          # Generated after TrOCR evaluation
├── layoutlm_report.json       # Generated after LayoutLMv3 evaluation
│
└── README.md
```

---

# Installation

Create a Python environment and install dependencies.

```bash
pip install -r requirements.txt
```

Dependencies include:

* transformers
* torch
* pillow
* numpy
* psutil
* pytesseract

Additionally **Tesseract OCR must be installed on the system**.

Download from:

[https://github.com/UB-Mannheim/tesseract/wiki](https://github.com/UB-Mannheim/tesseract/wiki)

---

# Running the Evaluation

## Run TrOCR Evaluation

```bash
python evaluate.py --model trocr
```

This will:

1. Load the TrOCR model
2. Run OCR inference on images
3. Compare predictions with ground truth
4. Compute latency and accuracy
5. Generate `trocr_report.json`

---

## Run LayoutLMv3 Evaluation

```bash
python evaluate.py --model layoutlm
```

This will:

1. Run Tesseract OCR on images
2. Extract words and bounding boxes
3. Normalize bounding boxes to the 0–1000 scale
4. Run LayoutLMv3 inference
5. Compute latency and memory usage
6. Generate `layoutlm_report.json`

---

# Example Output

## TrOCR Report

```json
{
  "model": "trocr",
  "hardware": "CPU-only",
  "latency_p50_ms": 3433,
  "latency_p95_ms": 4668,
  "peak_ram_mb": 1614,
  "text_accuracy": 0.76
}
```

---

## LayoutLMv3 Report

```json
{
  "model": "layoutlmv3",
  "hardware": "CPU-only",
  "latency_p50_ms": 331,
  "latency_p95_ms": 349,
  "peak_ram_mb": 756
}
```

---

# Models Used

## TrOCR

```
microsoft/trocr-base-printed
```

Documentation:

[https://huggingface.co/microsoft/trocr-base-printed](https://huggingface.co/microsoft/trocr-base-printed)

TrOCR combines:

* Vision Transformer encoder
* Transformer text decoder

for end-to-end OCR.

---

## LayoutLMv3

```
microsoft/layoutlmv3-base
```

Documentation:

[https://huggingface.co/microsoft/layoutlmv3-base](https://huggingface.co/microsoft/layoutlmv3-base)

LayoutLMv3 integrates:

* image features
* text tokens
* spatial layout information

to understand structured documents.

---

# Purpose of This Repository

This evaluation harness is intended to:

* benchmark document AI models
* measure inference latency
* measure real system memory usage
* evaluate OCR accuracy
* provide reproducible evaluation reports

The framework can be extended to benchmark additional document models such as:

* Donut
* PaddleOCR
* Tesseract
* DocFormer

---

# Status

D1 — TrOCR evaluation harness completed
D2 — LayoutLMv3 evaluation harness completed

---

