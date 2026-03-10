# TrOCR Evaluation Harness

## Overview

This repository contains the **Week 1 deliverable** for evaluating the OCR performance of the `microsoft/trocr-base-printed` model.

The goal of this task is to run OCR on sample images and evaluate the model using the following metrics:

* **Latency (p50, p95)**
* **Peak RAM usage**
* **Text accuracy**

---

## Google Colab Notebook

The full implementation and experiments can be run using the following Google Colab notebook:

**Colab Link:**
https://colab.research.google.com/drive/1mnM-Fd07JYq4dRMs9fMc4obTXDjq3tpB?usp=sharing

---

## Pipeline

Image
↓
TrOCR OCR Model
↓
Predicted Text
↓
Comparison with Ground Truth
↓
Evaluation Metrics

---

## Metrics Evaluated

* **Latency** – time taken for OCR inference
* **Peak RAM usage** – memory consumed during inference
* **Text Accuracy** – similarity between predicted text and ground truth

---

## Sample Dataset

Example test images used for evaluation:

* `img1.png` → Patient Ravi Kumar
* `img2.png` → Age 54
* `img3.png` → Doctor Mehta

---

## Output

The evaluation script generates a JSON report:

```
evaluation_report.json
```

Example output:

```json
{
  "model": "trocr",
  "latency_p50_ms": 52,
  "latency_p95_ms": 75,
  "peak_ram_mb": 920,
  "text_accuracy": 0.91
}
```

---

## Objective of the Evaluation

The purpose of this evaluation harness is to measure the **performance characteristics of the OCR model**, which will later help compare different document processing models within the Laryaa system.
