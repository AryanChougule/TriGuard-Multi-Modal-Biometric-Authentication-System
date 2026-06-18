"""
tools/face_embedding.py

Build face embeddings for TriGuard from a raw image dataset.

Expected dataset layout
-----------------------
raw_images/
    student_001/
        img1.jpg
        img2.png
    student_002/
        photo1.jpeg

The script extracts ArcFace embeddings for every readable image, averages the
embeddings per student, and saves a pickle database that matches the runtime
face module.
"""

import argparse
import os
import pickle
import sys
from typing import Dict, List, Optional

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from modules.face_module import preload_face_runtime


DEFAULT_DATASET_ROOT = "data/face_dataset/raw_images"
DEFAULT_OUTPUT = "data/face_embeddings/student_embeddings.pkl"
VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def load_embedding_model():
    preload_face_runtime()
    from deepface import DeepFace  # noqa: F401
    return DeepFace


def discover_dataset(dataset_root: str) -> Dict[str, List[str]]:
    if not os.path.isdir(dataset_root):
        raise FileNotFoundError(f"Dataset directory not found: '{dataset_root}'")

    dataset: Dict[str, List[str]] = {}
    for student_id in sorted(os.listdir(dataset_root)):
        student_dir = os.path.join(dataset_root, student_id)
        if not os.path.isdir(student_dir):
            continue

        images = []
        for name in sorted(os.listdir(student_dir)):
            if name.lower().endswith(VALID_EXTENSIONS):
                images.append(os.path.join(student_dir, name))

        if images:
            dataset[student_id] = images

    return dataset


def extract_embedding(deepface_module, image_path: str, model_name: str = "ArcFace") -> Optional[np.ndarray]:
    image = cv2.imread(image_path)
    if image is None:
        return None

    try:
        representation = deepface_module.represent(
            img_path=image,
            model_name=model_name,
            enforce_detection=False,
        )
        if not representation:
            return None
        return np.asarray(representation[0]["embedding"], dtype=np.float32)
    except Exception:
        return None


def build_embeddings(dataset: Dict[str, List[str]], model_name: str = "ArcFace") -> Dict[str, np.ndarray]:
    deepface_module = load_embedding_model()
    student_db: Dict[str, np.ndarray] = {}

    for student_id, image_paths in dataset.items():
        embeddings = []
        print(f"Processing student: {student_id} ({len(image_paths)} images)")

        for image_path in image_paths:
            emb = extract_embedding(deepface_module, image_path, model_name=model_name)
            if emb is None:
                print(f"  - skipped {os.path.basename(image_path)}")
                continue
            embeddings.append(emb)
            print(f"  - embedded {os.path.basename(image_path)}")

        if embeddings:
            student_db[student_id] = np.mean(np.stack(embeddings, axis=0), axis=0).astype(np.float32)
            print(f"  -> saved averaged embedding for {student_id}")
        else:
            print(f"  -> no valid embeddings for {student_id}, skipped")

    return student_db


def save_database(db: Dict[str, np.ndarray], output_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(db, f)


def parse_args():
    parser = argparse.ArgumentParser(description="Build TriGuard face embeddings")
    parser.add_argument(
        "--dataset_root",
        default=DEFAULT_DATASET_ROOT,
        help="Path to the raw face image dataset root",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Path to save student_embeddings.pkl",
    )
    parser.add_argument(
        "--model_name",
        default="ArcFace",
        help="DeepFace model name",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"Dataset root: {os.path.abspath(args.dataset_root)}")
    print(f"Output: {os.path.abspath(args.output)}")

    dataset = discover_dataset(args.dataset_root)
    if not dataset:
        raise RuntimeError("No student folders with images were found.")

    print(f"Found {len(dataset)} student folder(s).")
    db = build_embeddings(dataset, model_name=args.model_name)
    if not db:
        raise RuntimeError("No embeddings were produced.")

    save_database(db, args.output)
    print(f"Saved {len(db)} student embeddings to {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
