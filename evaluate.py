import json
import time
import numpy as np
import os
import argparse
import psutil
import pytesseract
import torch

# ---- TESSERACT PATH FIX ----
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

from PIL import Image
from difflib import SequenceMatcher
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    LayoutLMv3Processor,
    LayoutLMv3Model
)


# -----------------------------
# Text normalization
# -----------------------------
def normalize_text(text):
    text = text.lower()
    text = text.replace(":", "")
    text = text.strip()
    return text


# -----------------------------
# TrOCR Evaluation
# -----------------------------
def evaluate_trocr(image_folder):

    print("Loading TrOCR model...")

    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")

    print("Model loaded successfully")

    ground_truth = {
        "image1.png": "Patient Ravi Kumar",
        "image2.png": "Age: 54",
        "image3.png": "Doctor: Mehta"
    }

    latencies = []
    accuracies = []

    process = psutil.Process(os.getpid())

    for img_name in os.listdir(image_folder):

        if not img_name.lower().endswith(".png"):
            continue

        if img_name not in ground_truth:
            print(f"Skipping {img_name} (no ground truth)")
            continue

        img_path = os.path.join(image_folder, img_name)

        image = Image.open(img_path).convert("RGB")

        start = time.perf_counter()

        pixel_values = processor(images=image, return_tensors="pt").pixel_values

        with torch.no_grad():
            generated_ids = model.generate(pixel_values)

        pred_text = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]

        end = time.perf_counter()

        latency = (end - start) * 1000
        latencies.append(latency)

        similarity = SequenceMatcher(
            None,
            normalize_text(pred_text),
            normalize_text(ground_truth[img_name])
        ).ratio()

        accuracies.append(similarity)

        print("\nImage:", img_name)
        print("Prediction:", pred_text)
        print("Accuracy:", similarity)

    peak_ram_mb = process.memory_info().rss / (1024 * 1024)

    report = {
        "model": "trocr",
        "hardware": "CPU-only",
        "latency_p50_ms": float(np.percentile(latencies, 50)),
        "latency_p95_ms": float(np.percentile(latencies, 95)),
        "peak_ram_mb": peak_ram_mb,
        "text_accuracy": float(np.mean(accuracies))
    }

    with open("trocr_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("\nEvaluation complete")
    print(report)


# -----------------------------
# LayoutLMv3 Evaluation
# -----------------------------
def evaluate_layoutlm(image_folder):

    print("Loading LayoutLMv3 model...")

    processor = LayoutLMv3Processor.from_pretrained(
        "microsoft/layoutlmv3-base",
        apply_ocr=False
    )

    model = LayoutLMv3Model.from_pretrained("microsoft/layoutlmv3-base")

    latencies = []

    process = psutil.Process(os.getpid())

    for img_name in os.listdir(image_folder):

        if not img_name.lower().endswith(".png"):
            continue

        img_path = os.path.join(image_folder, img_name)

        image = Image.open(img_path).convert("RGB")

        # OCR step
        ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)

        words = []
        boxes = []

        width, height = image.size

        for i, word in enumerate(ocr_data["text"]):

            if word.strip() == "":
                continue

            x = ocr_data["left"][i]
            y = ocr_data["top"][i]
            w = ocr_data["width"][i]
            h = ocr_data["height"][i]

            words.append(word)

            # ---- Normalize bounding boxes (0-1000 scale) ----
            x0 = int(1000 * x / width)
            y0 = int(1000 * y / height)
            x1 = int(1000 * (x + w) / width)
            y1 = int(1000 * (y + h) / height)

            boxes.append([x0, y0, x1, y1])

        if len(words) == 0:
            continue

        start = time.perf_counter()

        encoding = processor(
            image,
            words,
            boxes=boxes,
            return_tensors="pt"
        )

        with torch.no_grad():
            outputs = model(**encoding)

        end = time.perf_counter()

        latency = (end - start) * 1000
        latencies.append(latency)

    peak_ram_mb = process.memory_info().rss / (1024 * 1024)

    report = {
        "model": "layoutlmv3",
        "hardware": "CPU-only",
        "latency_p50_ms": float(np.percentile(latencies, 50)),
        "latency_p95_ms": float(np.percentile(latencies, 95)),
        "peak_ram_mb": peak_ram_mb
    }

    with open("layoutlm_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("\nEvaluation complete")
    print(report)


# -----------------------------
# CLI Interface
# -----------------------------
def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        required=True,
        choices=["trocr", "layoutlm"],
        help="Select model to evaluate"
    )

    parser.add_argument(
        "--images",
        default="images",
        help="Folder containing evaluation images"
    )

    args = parser.parse_args()

    if args.model == "trocr":
        evaluate_trocr(args.images)

    elif args.model == "layoutlm":
        evaluate_layoutlm(args.images)


if __name__ == "__main__":
    main()