import json
import time
import tracemalloc
import numpy as np
import os
import sys

from PIL import Image
from difflib import SequenceMatcher
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


def normalize_text(text):
    """
    Normalize text for fair comparison
    """
    text = text.lower()
    text = text.replace(":", "")
    text = text.strip()
    return text


def evaluate(image_folder):

    print("Loading TrOCR model...")

    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")

    print("Model loaded successfully")

    # Ground truth mapping
    ground_truth = {
        "image1.png": "Patient Ravi Kumar",
        "image2.png": "Age: 54",
        "image3.png": "Doctor: Mehta"
    }

    latencies = []
    accuracies = []

    tracemalloc.start()

    for img_name in os.listdir(image_folder):

        if not img_name.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        img_path = os.path.join(image_folder, img_name)

        if img_name not in ground_truth:
            print(f"Skipping {img_name} (no ground truth)")
            continue

        gt_text = ground_truth[img_name]

        image = Image.open(img_path).convert("RGB")

        start = time.perf_counter()

        pixel_values = processor(images=image, return_tensors="pt").pixel_values

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
            normalize_text(gt_text)
        ).ratio()

        accuracies.append(similarity)

        print("\nImage:", img_name)
        print("Prediction:", pred_text)
        print("Ground Truth:", gt_text)
        print("Latency(ms):", latency)
        print("Accuracy:", similarity)

    # Safety check
    if len(latencies) == 0:
        print("No images evaluated. Check filenames or ground truth.")
        return

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    peak_ram_mb = peak / (1024 * 1024)

    p50_latency = float(np.percentile(latencies, 50))
    p95_latency = float(np.percentile(latencies, 95))
    mean_accuracy = float(np.mean(accuracies))

    report = {
        "model": "trocr",
        "latency_p50_ms": p50_latency,
        "latency_p95_ms": p95_latency,
        "peak_ram_mb": peak_ram_mb,
        "text_accuracy": mean_accuracy
    }

    with open("evaluation_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("\nEvaluation complete")
    print(report)


def main():

    if len(sys.argv) < 2:
        print("Usage: python evaluate_trocr.py <image_folder>")
        sys.exit(1)

    image_folder = sys.argv[1]

    evaluate(image_folder)


if __name__ == "__main__":
    main()