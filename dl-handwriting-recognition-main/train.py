"""train.py
- Reads raw training images from a folder
- Applies the same segmentation + vertical-split + augmentation idea as
  `notebooks/data_preprocessing_v2.ipynb`
- Trains the same CNN as `notebooks/model_training_v2.ipynb`

Outputs:
- model.keras
- labels.json  (required by run.py on any PC)
"""

# ============================================================================
# COMMAND LINE (copy/paste)
# ============================================================================
# From the project folder:
#   python train.py
#
# Recommended (explicit paths):
#   python train.py --train_dir "train" --model_out "model.keras" --labels_out "labels.json"

from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter
from pathlib import Path

# Force CPU-only (set before importing TensorFlow)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
import cv2


def list_images(folder: str) -> list[str]:
    p = Path(folder)
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
    return sorted([str(fp) for fp in p.rglob("*") if fp.is_file() and fp.suffix.lower() in exts])


def label_from_filename(fp: str) -> str:
    return Path(fp).name[:2]


def to_grayscale(img: np.ndarray) -> np.ndarray | None:
    if img is None:
        return None
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


# ----------------------------
# segmentation (from notebooks/data_preprocessing_v2.ipynb)
# ----------------------------
def segment_lines(gray: np.ndarray, min_line_height: int) -> list[tuple[int, int]]:
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
            if end - start >= min_line_height:
                lines.append((max(0, start - 2), min(h, end + 2)))
    if in_line:
        lines.append((start, h))
    return lines if lines else [(0, h)]


def segment_words_from_line(line_img: np.ndarray, min_word_size: int) -> list[np.ndarray]:
    gray = to_grayscale(line_img)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    dilated = cv2.dilate(th, kernel, iterations=1)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bboxes: list[tuple[int, int, int, int]] = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < min_word_size or h < min_word_size:
            continue
        bboxes.append((x, y, w, h))
    bboxes = sorted(bboxes, key=lambda b: b[0])
    words = [line_img[y : y + h, x : x + w] for (x, y, w, h) in bboxes]
    return words if words else [line_img]


def segment_chars_from_word(word_img: np.ndarray, min_char_width: int) -> list[np.ndarray]:
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
        if (not is_sep) and (not in_char):
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


def resize_and_normalize_char(ch_img: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    # Match notebook behavior: grayscale->RGB, otherwise keep as-is (cv2.imread gives BGR).
    if len(ch_img.shape) == 2:
        ch = cv2.cvtColor(ch_img, cv2.COLOR_GRAY2RGB)
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
# ----------------------------
# The actual model instance is created (built) and compiled inside `main()` after data prep. Defined here for reusability and readability.
def build_character_cnn(input_shape: tuple[int, int, int], num_classes: int) -> keras.Model:
    layers = keras.layers
    models = keras.models

    inp = layers.Input(shape=input_shape)
    x = layers.RandomRotation(0.05)(inp)
    x = layers.RandomTranslation(0.03, 0.03)(x)

    for filters in [32, 64, 128, 256, 256]:
        x = layers.Conv2D(filters, (3, 3), padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(1024, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    return models.Model(inp, out)

# ----------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Standalone v2 training (character-based CNN)")
    ap.add_argument("--train_dir", default="train", help="Training images folder")
    ap.add_argument("--model_out", default="model.keras", help="Output model path")
    ap.add_argument("--labels_out", default="labels.json", help="Output labels.json path")
    ap.add_argument("--char_h", type=int, default=64)
    ap.add_argument("--char_w", type=int, default=64)
    ap.add_argument("--min_char_width", type=int, default=6)
    ap.add_argument("--min_line_height", type=int, default=6)
    ap.add_argument("--min_word_size", type=int, default=8)
    ap.add_argument("--num_vertical_splits", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--val_fraction", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    # Strong augmentation (pre-segmentation) like preprocessing_v2
    augmentation_preseg = tf.keras.Sequential(
        [
            keras.layers.RandomRotation(0.15),
            keras.layers.RandomTranslation(0.08, 0.08),
            keras.layers.RandomZoom(0.10, 0.10),
            keras.layers.RandomContrast(0.15),
        ],
        name="pre_segmentation_augmentation",
    )

    @tf.function
    def _augment_tensor(x):
        return augmentation_preseg(x, training=True)

    def augment_image_numpy(img_uint8: np.ndarray) -> np.ndarray:
        img = img_uint8.astype(np.float32) / 255.0
        t = tf.convert_to_tensor(img[None, ...], dtype=tf.float32)
        aug = _augment_tensor(t)
        aug_np = aug[0].numpy()
        return np.clip(aug_np * 255.0, 0, 255).astype(np.uint8)

    train_files = list_images(args.train_dir)
    if not train_files:
        raise SystemExit(f"No training images found under: {args.train_dir}")

    # Fit label set from training data only
    labels_list = sorted({label_from_filename(fp) for fp in train_files})
    label_to_index = {lab: i for i, lab in enumerate(labels_list)}

    with open(args.labels_out, "w", encoding="utf-8") as f:
        json.dump(labels_list, f, indent=2)

    print("=" * 60)
    print("STANDALONE v2 TRAINING")
    print("=" * 60)
    print(f"Train dir: {os.path.abspath(args.train_dir)}")
    print(f"Classes:   {len(labels_list)}")
    print(f"Outputs:   {os.path.abspath(args.model_out)} and {os.path.abspath(args.labels_out)}")

    def extract_chars_from_image(img: np.ndarray, apply_augmentation: bool) -> list[np.ndarray]:
        H, W = img.shape[:2]
        splits = max(1, int(args.num_vertical_splits))
        third = W // splits if splits > 0 else W
        patches = [img[:, i * third : (i + 1) * third] for i in range(splits)]
        if splits > 0 and (W % splits) != 0:
            patches[-1] = img[:, (splits - 1) * third :]

        enhanced_patches: list[np.ndarray] = []
        for p in patches:
            enhanced_patches.append(p)
            if apply_augmentation:
                try:
                    enhanced_patches.append(augment_image_numpy(p))
                except Exception:
                    enhanced_patches.append(p.copy())

        all_chars: list[np.ndarray] = []
        for patch in enhanced_patches:
            gray = to_grayscale(patch)
            lines = segment_lines(gray, args.min_line_height)
            for y1, y2 in lines:
                line_img = patch[y1:y2, :]
                words = segment_words_from_line(line_img, args.min_word_size) or [line_img]
                for wimg in words:
                    chars = segment_chars_from_word(wimg, args.min_char_width)
                    if not chars:
                        all_chars.append(resize_and_normalize_char(wimg, args.char_h, args.char_w))
                    else:
                        for c in chars:
                            all_chars.append(resize_and_normalize_char(c, args.char_h, args.char_w))
        return all_chars

    print("\nExtracting training characters (with augmentation)...")
    X_train_list: list[np.ndarray] = []
    y_train_list: list[int] = []

    for fp in train_files:
        img = cv2.imread(fp)
        if img is None:
            continue
        label = label_from_filename(fp)
        chars = extract_chars_from_image(img, apply_augmentation=True)
        for ch in chars:
            X_train_list.append(ch)
            y_train_list.append(label_to_index[label])

    if not X_train_list:
        raise SystemExit("No characters extracted from training images.")

    X_train = np.asarray(X_train_list, dtype=np.float32)
    y_train = np.asarray(y_train_list, dtype=np.int64)

    # Shuffle
    perm = np.random.permutation(len(X_train))
    X_train = X_train[perm]
    y_train = y_train[perm]

    # Validation split
    val_count = max(1, int(args.val_fraction * len(X_train)))
    X_val = X_train[:val_count]
    y_val = y_train[:val_count]
    X_train_final = X_train[val_count:]
    y_train_final = y_train[val_count:]

    y_train_final_cat = keras.utils.to_categorical(y_train_final, num_classes=len(labels_list))
    y_val_cat = keras.utils.to_categorical(y_val, num_classes=len(labels_list))

    # Class weights
    counts = Counter(y_train_final.tolist())
    class_weight = {i: (len(y_train_final) / (len(counts) * counts[i])) for i in counts}

    print("\nDATA")
    print("-" * 60)
    print(f"Train characters: {len(X_train_final)}")
    print(f"Val characters:   {len(X_val)}")
    print(f"X shape:          {X_train.shape}")

    # Build + compile the model inside `main()` (actual model instance used for training).
    model = build_character_cnn((args.char_h, args.char_w, 3), num_classes=len(labels_list))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    training_callbacks = [
        keras.callbacks.ModelCheckpoint(
            args.model_out,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
            mode="max",
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=12,
            restore_best_weights=True,
            verbose=1,
            mode="max",
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            verbose=1,
            min_lr=1e-6,
        ),
    ]

    print("\nTRAINING")
    print("-" * 60)
    history = model.fit(
        X_train_final,
        y_train_final_cat,
        validation_data=(X_val, y_val_cat),
        epochs=args.epochs,
        batch_size=args.batch_size,
        shuffle=True,
        class_weight=class_weight,
        callbacks=training_callbacks,
        verbose=1,
    )

    print("\nDONE")
    best_val_acc = max(history.history.get("val_accuracy", [0.0]))
    print(f"Best val_accuracy: {best_val_acc:.4f} ({best_val_acc * 100:.2f}%)")
    print(f"Saved model:  {os.path.abspath(args.model_out)}")
    print(f"Saved labels: {os.path.abspath(args.labels_out)}")


if __name__ == "__main__":
    main()

