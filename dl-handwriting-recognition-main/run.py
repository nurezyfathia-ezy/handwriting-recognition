# ============================================================================
# How to run writer identification deployment/evaluation from command line
# ============================================================================
# From the project folder:
#   python run.py
#
# Recommended (explicit labels path):
#   python run.py --model "model.keras" --labels "labels.json" --test_dir "test" --result_csv "result.csv"
# ============================================================================

"""run.py - Deployment & Evaluation (v2, notebook-independent)
- Reads test images from a user-specified folder (Windows CPU, no GPU assumed)
- Loads a saved model (model.keras)
- Computes average accuracy using labels from filenames (first 2 chars)
- Writes result.csv with: filename, actual_label, predicted_label

Concept note (explicit):
- The is trained on *character crops* (64×64×3) extracted from each image. Therefore, to stay consistent, this script includes the same character
  segmentation pipeline from `notebooks/data_preprocessing_v2.ipynb`, then aggregates character predictions to image-level predictions as in `notebooks/model_training_v2.ipynb`.

Deployment packaging note (explicit):
- To map model output indices -> writer IDs, this script expects `labels.json` to be present next to the model (or next to this script).
  `train.py` writes `labels.json` automatically.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

# Force CPU-only (set before importing TensorFlow)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
from tensorflow import keras
import cv2


# ----------------------------
# v2 preprocessing constants (match notebooks/data_preprocessing_v2.ipynb)
# ----------------------------
CHAR_H = 64
CHAR_W = 64
NUM_CHANNELS = 3

MIN_CHAR_WIDTH = 6
MIN_LINE_HEIGHT = 6
MIN_WORD_SIZE = 8

NUM_VERTICAL_SPLITS = 3


# ----------------------------
# Helpers
# ----------------------------
def _require_exists(path: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Not found: {path}")


def _list_images(folder: str) -> list[str]:
    p = Path(folder)
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
    return sorted([str(fp) for fp in p.rglob("*") if fp.is_file() and fp.suffix.lower() in exts])


def label_from_filename(filepath: str) -> str:
    return Path(filepath).name[:2]


def _load_labels_list(model_path: str, num_classes: int, labels_path: str | None = None) -> list[str]:
    """
    Loads class index -> writer label mapping.
    """
    model_dir = os.path.dirname(os.path.abspath(model_path))
    script_dir = os.path.dirname(os.path.abspath(__file__))

    candidate_paths = (
        [os.path.abspath(labels_path)]
        if labels_path
        else [
        os.path.join(model_dir, "labels.json"),
        os.path.join(script_dir, "labels.json"),
        ]
    )
    for p in candidate_paths:
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                labels = json.load(f)
            labels = [str(x) for x in labels]
            if len(labels) != num_classes:
                raise ValueError(
                    f"{p} contains {len(labels)} labels, but model expects {num_classes} classes."
                )
            return labels

    if labels_path:
        raise FileNotFoundError(
            f"labels.json not found at: {os.path.abspath(labels_path)}\n"
            "Fix: pass the correct --labels path (exported by train.py)."
        )

    raise FileNotFoundError(
        "labels.json not found (next to model or script).\n"
        "Fix: either pass --labels labels.json OR copy labels.json next to model.keras."
    )


# ----------------------------
# v2 segmentation pipeline (from notebooks/data_preprocessing_v2.ipynb)
# ----------------------------
def to_grayscale(img: np.ndarray) -> np.ndarray | None:
    if img is None:
        return None
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def resize_and_normalize_char(ch_img: np.ndarray, target_h: int = CHAR_H, target_w: int = CHAR_W) -> np.ndarray:
    """
    - center on white background
    - output float32 in [0,1], shape (H,W,3)
    IMPORTANT: when input is color from cv2.imread, channel order is BGR..
    """
    if ch_img is None:
        padded = 255 * np.ones((target_h, target_w, 3), dtype=np.uint8)
        return padded.astype(np.float32) / 255.0

    if len(ch_img.shape) == 2:
        ch = cv2.cvtColor(ch_img, cv2.COLOR_GRAY2BGR)
    else:
        ch = ch_img.copy()

    h, w = ch.shape[:2]
    scale = min(target_w / max(1, w), target_h / max(1, h))
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(ch, (nw, nh), interpolation=cv2.INTER_AREA)

    pad_left = (target_w - nw) // 2
    pad_top = (target_h - nh) // 2
    padded = 255 * np.ones((target_h, target_w, 3), dtype=np.uint8)
    padded[pad_top : pad_top + nh, pad_left : pad_left + nw, :] = resized
    return padded.astype(np.float32) / 255.0


def segment_lines(gray: np.ndarray) -> list[tuple[int, int]]:
    h = gray.shape[0]
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
    closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=1)

    proj = np.sum(closed, axis=1)
    if proj.max() == 0:
        return [(0, h)]

    thresh = max(1, int(0.03 * proj.max()))
    lines: list[tuple[int, int]] = []
    in_line = False
    start = 0

    for y, v in enumerate(proj):
        if v > thresh and not in_line:
            in_line = True
            start = y
        elif v <= thresh and in_line:
            end = y
            in_line = False
            if end - start >= MIN_LINE_HEIGHT:
                lines.append((max(0, start - 2), min(h, end + 2)))

    if in_line:
        lines.append((start, h))

    return lines if lines else [(0, h)]


def segment_words_from_line(line_img: np.ndarray) -> list[np.ndarray]:
    gray = to_grayscale(line_img)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    dilated = cv2.dilate(th, kernel, iterations=1)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bboxes: list[tuple[int, int, int, int]] = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < MIN_WORD_SIZE or h < MIN_WORD_SIZE:
            continue
        bboxes.append((x, y, w, h))

    bboxes = sorted(bboxes, key=lambda b: b[0])
    words = [line_img[y : y + h, x : x + w] for (x, y, w, h) in bboxes]
    return words if words else [line_img]


def segment_chars_from_word(word_img: np.ndarray, min_char_width: int = MIN_CHAR_WIDTH) -> list[np.ndarray]:
    gray = to_grayscale(word_img)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cols = np.sum(th, axis=0)
    if cols.max() == 0:
        return []

    thresh = max(1, int(0.05 * cols.max()))
    separators = cols <= thresh

    chars: list[np.ndarray] = []
    in_char = False
    start = 0
    for i, is_sep in enumerate(separators):
        if not is_sep and not in_char:
            in_char = True
            start = i
        elif is_sep and in_char:
            end = i
            in_char = False
            if end - start >= min_char_width:
                chars.append(word_img[:, start:end])

    if in_char:
        end = len(separators)
        if end - start >= min_char_width:
            chars.append(word_img[:, start:end])

    return chars


def extract_characters_from_image(img: np.ndarray) -> list[np.ndarray]:
    H, W = img.shape[:2]

    # Same "vertical splits" trick as preprocessing v2 (but no augmentation during inference)
    third = W // NUM_VERTICAL_SPLITS if NUM_VERTICAL_SPLITS > 0 else W
    if third <= 0:
        return [resize_and_normalize_char(img)]

    patches = [img[:, i * third : (i + 1) * third] for i in range(NUM_VERTICAL_SPLITS)]
    if W % NUM_VERTICAL_SPLITS != 0:
        patches[-1] = img[:, (NUM_VERTICAL_SPLITS - 1) * third :]

    all_chars: list[np.ndarray] = []
    for patch in patches:
        gray = to_grayscale(patch)
        lines = segment_lines(gray)

        for y1, y2 in lines:
            line_img = patch[y1:y2, :]
            words = segment_words_from_line(line_img) or [line_img]

            for word in words:
                chars = segment_chars_from_word(word, MIN_CHAR_WIDTH)
                if not chars:
                    all_chars.append(resize_and_normalize_char(word))
                else:
                    for c in chars:
                        all_chars.append(resize_and_normalize_char(c))

    return all_chars


# ----------------------------
# v2 aggregation (from notebooks/model_training_v2.ipynb)
# ----------------------------
def aggregate_char_predictions_weighted_mean(char_probs: np.ndarray, num_classes: int):
    """Weighted-mean aggregation (conf^2 weights), matching v2 notebook."""
    if len(char_probs) == 0:
        return -1, 0.0

    confidences = np.max(char_probs, axis=1)
    weights = confidences**2
    denom = float(np.sum(weights))
    if denom <= 0:
        weights = np.ones_like(weights) / len(weights)
    else:
        weights = weights / denom
    aggregated_probs = np.sum(char_probs * weights[:, np.newaxis], axis=0)

    predicted_class = int(np.argmax(aggregated_probs))
    confidence = float(aggregated_probs[predicted_class])
    return predicted_class, confidence


def _write_result_csv(path: str, rows: list[dict]) -> None:
    fieldnames = ["filename", "actual_label", "predicted_label"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    ap = argparse.ArgumentParser(description="Writer Identification - Deployment/Evaluation (v2)")
    ap.add_argument("--model", default="model.keras", help="Path to model.keras")
    ap.add_argument("--labels", default="", help="Path to labels.json")
    ap.add_argument("--test_dir", default="test", help="Folder containing test images")
    ap.add_argument("--result_csv", default="result.csv", help="Output CSV path")
    args = ap.parse_args()

    model_path = os.path.abspath(args.model)
    test_dir = os.path.abspath(args.test_dir)
    result_csv_path = os.path.abspath(args.result_csv)

    _require_exists(model_path)
    _require_exists(test_dir)

    print("=" * 60)
    print("Writer Identification - Deployment/Evaluation (v2)")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Test dir: {test_dir}")
    print(f"Output: {result_csv_path}")

    test_files = _list_images(test_dir)
    if not test_files:
        raise FileNotFoundError(f"No images found under: {test_dir}")

    # Load model
    model = keras.models.load_model(model_path, compile=False)
    num_classes = int(model.output_shape[-1])

    labels_path = os.path.abspath(args.labels) if str(args.labels).strip() else None
    if labels_path:
        _require_exists(labels_path)
    labels_list = _load_labels_list(model_path=model_path, num_classes=num_classes, labels_path=labels_path)
    label_to_index = {lab: i for i, lab in enumerate(labels_list)}

    rows: list[dict] = []
    correct = 0

    print("-" * 60)
    print(f"Images found: {len(test_files)}")
    print(f"Model classes: {num_classes}")
    print(f"Labels source: labels.json ({len(labels_list)} labels)")
    print("-" * 60)

    processed = 0
    total = len(test_files)

    # Print ~50 updates total (minimum every 1 image)
    progress_every = max(1, total // 50)

    print(f"Starting prediction on {total} images...")

    for fp in test_files:
        filename = os.path.basename(fp)
        actual_label = label_from_filename(filename)

        if actual_label not in label_to_index:
            # If this happens, labels.json doesn't match the dataset naming.
            raise ValueError(
                f"Actual label '{actual_label}' (from {filename}) not found in labels mapping."
            )
        true_class = label_to_index[actual_label]

        img = cv2.imread(fp)
        if img is None:
            rows.append(
                {
                    "filename": filename,
                    "actual_label": actual_label,
                    "predicted_label": "",
                }
            )
            continue

        chars = extract_characters_from_image(img)
        if not chars:
            chars = [resize_and_normalize_char(img)]

        X = np.stack(chars, axis=0).astype(np.float32, copy=False)
        char_probs = model.predict(X, verbose=0)

        pred_class, _confidence = aggregate_char_predictions_weighted_mean(char_probs, num_classes=num_classes)
        pred_label = labels_list[int(pred_class)] if 0 <= int(pred_class) < len(labels_list) else ""
        is_correct = int(pred_class) == int(true_class)
        if is_correct:
            correct += 1

        rows.append(
            {
                "filename": filename,
                "actual_label": actual_label,
                "predicted_label": pred_label,
            }
        )

        processed += 1
        if (processed % progress_every == 0) or (processed == total):
            running_acc = (correct / processed) if processed else 0.0
            print(
                f"[{processed}/{total}] last={filename} running_acc={running_acc*100:.2f}%",
                end="\r",
                flush=True,
            )

    total = len(test_files)
    accuracy = (correct / total) if total else 0.0

    _write_result_csv(result_csv_path, rows)

    print("-" * 60)
    print(f"Average accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Saved: {result_csv_path}")
    print("Done.")


if __name__ == "__main__":
    main()
