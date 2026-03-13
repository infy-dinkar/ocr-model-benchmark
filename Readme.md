# TrOCR Evaluation Harness

## Overview

This repository contains a standalone evaluation harness for benchmarking the **TrOCR OCR model** (`microsoft/trocr-base-printed`) on document images.

The goal of this script is to evaluate OCR inference performance under controlled conditions and produce a structured report containing latency, memory usage, and text accuracy.

This script is designed to run **locally on offline machines**, which is important for environments such as hospital systems where internet access may be restricted.

---

# Evaluation Pipeline

The evaluation process follows the pipeline below:

```
Input Images
      ↓
Image Preprocessing
      ↓
TrOCR Model Inference
      ↓
Predicted Text Generation
      ↓
Comparison With Ground Truth
      ↓
Metric Computation
      ↓
Evaluation Report (JSON)
```

---

# Metrics Measured

The evaluation harness computes the following metrics:

### Latency

Measures the time taken for a single OCR inference.

* **p50 latency** → median inference time
* **p95 latency** → worst-case latency

### Peak RAM Usage

Tracks maximum memory usage during evaluation using Python's `tracemalloc`.

### Text Accuracy

Accuracy is computed using string similarity between predicted text and ground truth.

---

# Repository Structure

```
trocr-evaluation-harness
│
├── evaluate_trocr.py        # Standalone evaluation script
├── requirements.txt         # Python dependencies
├── images                   # Sample test images
│   ├── image1.png
│   ├── image2.png
│   └── image3.png
│
├── evaluation_report.json   # Generated after script execution
└── README.md
```

---

# Installation

Create a Python environment and install dependencies.

```
pip install -r requirements.txt
```

Dependencies:

* transformers
* torch
* pillow
* numpy

---

# Running the Evaluation

Run the script by passing the image directory as input.

```
python evaluate_trocr.py images
```

The script will:

1. Load the TrOCR model
2. Run OCR inference on all images
3. Compare predictions with ground truth
4. Compute evaluation metrics
5. Generate a report

---

# Example Output

After execution, the script generates:

```
evaluation_report.json
```

Example:

```
{
  "model": "trocr",
  "latency_p50_ms": 3900,
  "latency_p95_ms": 5357,
  "peak_ram_mb": 17.6,
  "text_accuracy": 0.23
}
```

---

# Model Used

The evaluation uses the following pretrained model:

```
microsoft/trocr-base-printed
```

Model documentation:

https://huggingface.co/microsoft/trocr-base-printed

TrOCR is a Transformer-based OCR system that combines a Vision Transformer encoder with a text decoder.

---

# Purpose of This Repository

This evaluation harness is intended to:

* Benchmark OCR inference performance
* Measure system resource consumption
* Provide reproducible OCR evaluation results
* Enable comparison with other document understanding models

This script can be extended to evaluate other models such as:

* LayoutLMv3
* Donut
* PaddleOCR
* Tesseract

---


