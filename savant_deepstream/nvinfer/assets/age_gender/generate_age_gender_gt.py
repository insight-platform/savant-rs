#!/usr/bin/env python3
"""Generate ground truth for the age_gender e2e test using CPU ONNX runtime.

Loads every 112x112 JPEG from assets/age_gender/, runs them through the
age_gender_mobilenet_v2_dynBatch.onnx model on CPU, and writes per-image
age/gender predictions to assets/age_gender/ground_truth.json.

Preprocessing matches DeepStream nvinfer settings:
    net-scale-factor = 0.007843137254902  (= 1/127.5)
    offsets          = 127.5;127.5;127.5
    => pixel_normalized = (pixel - 127.5) / 127.5

Postprocessing follows the Savant age_gender_converter:
    age    = sum(age_probs * arange(0, 101))
    gender = "male" if argmax(gender_probs) == 0 else "female"
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
ASSETS_DIR = SCRIPT_DIR / "assets" / "age_gender"
MODEL_PATH = SCRIPT_DIR / "assets" / "age_gender_mobilenet_v2_dynBatch.onnx"

GENDER_MAPPING = ["male", "female"]
AGE_RANGE = np.arange(0, 101, dtype=np.float32)


def preprocess(img_path: Path) -> np.ndarray:
    """Load a JPEG, convert to RGB float32, normalize to [-1, 1], CHW layout."""
    img = Image.open(img_path).convert("RGB")
    arr = np.array(img, dtype=np.float32)  # (112, 112, 3)
    arr = (arr - 127.5) * (1.0 / 127.5)
    arr = arr.transpose(2, 0, 1)  # CHW
    return np.expand_dims(arr, 0)  # (1, 3, 112, 112)


def main() -> None:
    if not MODEL_PATH.exists():
        print(f"ERROR: ONNX model not found at {MODEL_PATH}", file=sys.stderr)
        sys.exit(1)

    sess = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])

    results = {}
    for fname in sorted(os.listdir(ASSETS_DIR)):
        if not fname.lower().endswith(".jpg"):
            continue

        inp = preprocess(ASSETS_DIR / fname)
        age_probs, gender_probs = sess.run(None, {"input": inp})
        age_probs = age_probs[0]  # (101,)
        gender_probs = gender_probs[0]  # (2,)

        age = float(np.sum(age_probs * AGE_RANGE))
        gender_idx = int(np.argmax(gender_probs))
        gender = GENDER_MAPPING[gender_idx]

        results[fname] = {"age": round(age, 2), "gender": gender}
        print(f"  {fname}: age={age:6.2f}  gender={gender}")

    out_path = ASSETS_DIR / "ground_truth.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
        f.write("\n")
    print(f"\nSaved GT ({len(results)} images) -> {out_path}")


if __name__ == "__main__":
    main()
