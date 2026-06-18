"""
modules/face_module.py
Face authentication module.

This version follows an attendance-style workflow:
- detect faces in each captured frame with MTCNN,
- extract ArcFace embeddings with DeepFace,
- match each embedding directly against the enrolled student database,
- log recognized IDs per processed frame and vote across processed frames.

It still plugs into the shared TriGuard capture/orchestration flow.
"""

import logging
import math
import os
import pickle
import importlib
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.base_module import BaseModule, ModuleResult

logger = logging.getLogger(__name__)


def preload_face_runtime() -> None:
    """
    Preload and patch the TensorFlow-backed face stack before Qt is imported.

    In this environment, importing PyQt5 before MTCNN/TensorFlow can cause the
    TensorFlow native DLL initialization to fail. DeepFace also expects
    `tensorflow.keras.layers.LocallyConnected2D`, which is missing in newer
    Keras builds even though ArcFace itself does not use that model.
    """
    layers_mod = importlib.import_module("tensorflow.keras.layers")
    if not hasattr(layers_mod, "LocallyConnected2D"):
        base_layer = getattr(layers_mod, "Layer")

        class LocallyConnected2D(base_layer):
            def __init__(self, *args, **kwargs):
                super().__init__(**kwargs)

            def call(self, inputs):
                return inputs

        setattr(layers_mod, "LocallyConnected2D", LocallyConnected2D)

    input_layer_cls = getattr(layers_mod, "InputLayer", None)
    if input_layer_cls is not None and not hasattr(input_layer_cls, "is_placeholder"):
        setattr(input_layer_cls, "is_placeholder", property(lambda self: False))

    # Newer Keras exposes KerasHistory.operation instead of .layer, but
    # older TensorFlow functional-model code still reads .layer directly.
    try:
        history_mod = importlib.import_module("keras.src.ops.node")
        keras_history = getattr(history_mod, "KerasHistory", None)
        if keras_history is not None and not hasattr(keras_history, "layer"):
            setattr(keras_history, "layer", property(lambda self: self.operation))

        node_cls = getattr(history_mod, "Node", None)
        if node_cls is not None and not hasattr(node_cls, "keras_inputs"):
            setattr(
                node_cls,
                "keras_inputs",
                property(lambda self: getattr(self, "input_tensors", None)),
            )
        if node_cls is not None and not hasattr(node_cls, "layer"):
            setattr(node_cls, "layer", property(lambda self: getattr(self, "operation", None)))
    except Exception:
        pass

    importlib.import_module("mtcnn")
    importlib.import_module("deepface")


class FaceModule(BaseModule):
    """
    Face recognition using MTCNN + ArcFace.

    Config keys used
    ----------------
    embeddings_path    : path to .pkl file -> {person_id: arcface_embedding}
    distance_threshold : max L2 distance to accept a match (presentation default 5.0)
    confidence_override_threshold : detection confidence that can override distance
    max_faces_allowed  : frames with more faces are skipped (default 1)
    model_name         : DeepFace model string (default "ArcFace")
    frame_skip         : process one frame every N frames (default 15)
    """

    def __init__(self, module_config: Dict[str, Any]):
        super().__init__(module_config)

        self.embeddings_path: str = module_config["embeddings_path"]
        self.distance_threshold: float = module_config.get("distance_threshold", 5.0)
        self.confidence_override_threshold: float = module_config.get(
            "confidence_override_threshold", 0.90
        )
        self.max_faces_allowed: int = module_config.get("max_faces_allowed", 1)
        self.model_name: str = module_config.get("model_name", "ArcFace")
        self.frame_skip: int = max(1, int(module_config.get("frame_skip", 15)))

        self._student_db: Dict[str, np.ndarray] = {}
        self._detector = None

        self._load_resources()

    def _load_resources(self) -> None:
        preload_face_runtime()

        if not os.path.exists(self.embeddings_path):
            raise FileNotFoundError(
                f"Face embeddings not found at '{self.embeddings_path}'. "
                "Run your enrollment script to generate the .pkl file."
            )

        with open(self.embeddings_path, "rb") as f:
            raw_db = pickle.load(f)

        self._student_db = {
            student_id: np.asarray(stored_emb, dtype=np.float32)
            for student_id, stored_emb in raw_db.items()
        }

        logger.info(
            f"[Face] Loaded {len(self._student_db)} identities from '{self.embeddings_path}'."
        )

        try:
            from mtcnn import MTCNN

            self._detector = MTCNN()
            logger.info("[Face] MTCNN detector initialized.")
        except ImportError as e:
            raise RuntimeError(
                "Face module could not initialize MTCNN/TensorFlow. "
                "Your log shows TensorFlow failed to load its native DLLs, so this is "
                "not just a missing 'mtcnn' package. Fix the TensorFlow runtime first, "
                "then retry."
            ) from e

    def run(
        self,
        frames: Optional[List[np.ndarray]] = None,
        face_abort: bool = False,
        **kwargs,
    ) -> ModuleResult:
        status_callback = kwargs.get("module_status_callback")

        if face_abort:
            self.emit_status(status_callback, "aborted", "Capture aborted due to multiple faces.")
            logger.warning("[Face] Hard abort received - multiple faces detected during capture.")
            return self.build_result(
                passed=False,
                score=0.0,
                identity="unknown",
                details={"reason": "multi_face_abort_from_capture_session"},
            )

        if not frames:
            self.emit_status(status_callback, "error", "No frames provided to face module.")
            logger.error("[Face] No frames provided.")
            return self.build_result(passed=False, score=0.0, error="No frames provided.")

        self.emit_status(
            status_callback,
            "running",
            "Face analysis started.",
            frames_total=len(frames),
            frame_skip=self.frame_skip,
        )
        logger.info(f"[Face] Processing {len(frames)} tracking frame(s).")

        try:
            return self._process_frames(frames, status_callback)
        except Exception as e:
            self.emit_status(status_callback, "error", f"Face analysis failed: {e}")
            logger.exception(f"[Face] Unexpected error: {e}")
            return self.build_result(passed=False, score=0.0, error=str(e))

    def _process_frames(self, frames: List[np.ndarray], status_callback=None) -> ModuleResult:
        preload_face_runtime()
        from deepface import DeepFace

        recognized_events: List[Dict[str, Any]] = []
        skipped_no_face = 0
        skipped_multi_face = 0
        detection_confidences: List[float] = []
        processed_frame_count = 0

        for idx, frame in enumerate(frames):
            if idx % self.frame_skip != 0:
                continue

            processed_frame_count += 1
            faces = self._detector.detect_faces(frame)

            if len(faces) == 0:
                skipped_no_face += 1
                logger.debug(f"[Face] Frame {idx}: no face detected.")
                continue

            if len(faces) > self.max_faces_allowed:
                skipped_multi_face += 1
                logger.warning(
                    f"[Face] Frame {idx}: {len(faces)} faces detected - "
                    f"frame skipped (max allowed: {self.max_faces_allowed})."
                )
                continue

            for face_data in faces:
                conf = float(face_data.get("confidence", 0.0))
                detection_confidences.append(conf)

                crop = self._extract_crop(frame, face_data.get("box"))
                if crop is None:
                    logger.debug(f"[Face] Frame {idx}: invalid crop skipped.")
                    continue

                try:
                    embedding = DeepFace.represent(
                        crop,
                        model_name=self.model_name,
                        enforce_detection=False,
                    )[0]["embedding"]
                except Exception as e:
                    logger.warning(f"[Face] Frame {idx}: embedding failed - {e}.")
                    continue

                recognized_id, min_dist = self._match_embedding(
                    np.asarray(embedding, dtype=np.float32)
                )
                accept_by_distance = min_dist <= self.distance_threshold
                accept_by_confidence = conf >= self.confidence_override_threshold

                if recognized_id and (accept_by_distance or accept_by_confidence):
                    pass_reason = (
                        "confidence_override"
                        if accept_by_confidence and not accept_by_distance
                        else "distance_within_limit"
                    )
                    recognized_events.append(
                        {
                            "StudentID": recognized_id,
                            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "FrameIndex": idx,
                            "Distance": round(min_dist, 4),
                            "DetectionConfidence": round(conf, 4),
                            "Score": round(
                                max(conf, self._dist_to_score(min_dist)),
                                4,
                            ),
                            "PassReason": pass_reason,
                        }
                    )
                    logger.info(
                        f"[Face] Frame {idx}: recognized '{recognized_id}' "
                        f"(dist={min_dist:.4f}, conf={conf:.3f}, reason={pass_reason})."
                    )
                    self.emit_status(
                        status_callback,
                        "running",
                        f"Face frame {idx} matched '{recognized_id}' ({pass_reason}).",
                        frame_index=idx,
                        recognized_id=recognized_id,
                        distance=min_dist,
                        confidence=conf,
                        reason=pass_reason,
                    )
                else:
                    logger.info(
                        f"[Face] Frame {idx}: no valid match "
                        f"(best='{recognized_id}', dist={min_dist:.4f}, conf={conf:.3f})."
                    )

        if not recognized_events:
            reason = "all_frames_multi_face" if skipped_multi_face > 0 else "no_students_recognized"
            logger.warning(
                f"[Face] No students recognized. "
                f"Skipped - no_face={skipped_no_face}, multi_face={skipped_multi_face}."
            )
            self.emit_status(
                status_callback,
                "finished",
                "Face analysis completed with no recognized identity.",
                frames_processed=processed_frame_count,
            )
            return self.build_result(
                passed=False,
                score=0.0,
                details={
                    "reason": reason,
                    "frames_total": len(frames),
                    "frames_processed": processed_frame_count,
                    "skipped_no_face": skipped_no_face,
                    "skipped_multi_face": skipped_multi_face,
                },
            )

        counts = Counter(event["StudentID"] for event in recognized_events)
        dists_by_id: Dict[str, List[float]] = defaultdict(list)
        confs_by_id: Dict[str, List[float]] = defaultdict(list)
        for event in recognized_events:
            dists_by_id[event["StudentID"]].append(float(event["Distance"]))
            confs_by_id[event["StudentID"]].append(float(event["DetectionConfidence"]))

        best_id = max(
            counts,
            key=lambda student_id: (counts[student_id], -float(np.mean(dists_by_id[student_id]))),
        )
        best_dist = float(np.mean(dists_by_id[best_id]))
        best_conf = float(np.mean(confs_by_id[best_id]))
        score = max(best_conf, self._dist_to_score(best_dist))
        passed = (best_dist <= self.distance_threshold) or (
            best_conf >= self.confidence_override_threshold
        )

        avg_conf = float(np.mean(detection_confidences)) if detection_confidences else 0.0

        logger.info(
            f"[Face] Match: '{best_id}' | dist={best_dist:.4f} | "
            f"conf={best_conf:.4f} | score={score:.4f} | passed={passed} | "
            f"recognized_frames={len(recognized_events)} | "
            f"processed_frames={processed_frame_count} | "
            f"votes={dict(counts)}"
        )
        self.emit_status(
            status_callback,
            "finished",
            f"Face analysis complete. Final identity: {best_id}.",
            passed=passed,
            identity=best_id,
            score=score,
            votes=dict(counts),
        )

        return self.build_result(
            passed=passed,
            score=score,
            identity=best_id if passed else "unknown",
            details={
                "matched_id": best_id,
                "distance": round(best_dist, 4),
                "confidence": round(best_conf, 4),
                "threshold": self.distance_threshold,
                "confidence_override_threshold": self.confidence_override_threshold,
                "frames_total": len(frames),
                "frames_processed": processed_frame_count,
                "recognized_events": len(recognized_events),
                "recognized_students": dict(counts),
                "skipped_no_face": skipped_no_face,
                "skipped_multi_face": skipped_multi_face,
                "avg_detection_confidence": round(avg_conf, 4),
                "frame_skip": self.frame_skip,
            },
        )

    def _extract_crop(self, frame: np.ndarray, box: Optional[List[int]]) -> Optional[np.ndarray]:
        if not box or len(box) != 4:
            return None

        x, y, w, h = box
        x = max(0, int(x))
        y = max(0, int(y))
        w = max(1, int(w))
        h = max(1, int(h))

        x2 = min(frame.shape[1], x + w)
        y2 = min(frame.shape[0], y + h)
        crop = frame[y:y2, x:x2]
        if crop.size == 0:
            return None
        return crop

    def _match_embedding(self, embedding: np.ndarray) -> Tuple[Optional[str], float]:
        best_id = None
        min_dist = float("inf")

        for student_id, stored_emb in self._student_db.items():
            dist = float(np.linalg.norm(embedding - stored_emb))
            if dist < min_dist:
                min_dist = dist
                best_id = student_id

        return best_id, min_dist

    def _dist_to_score(self, distance: float) -> float:
        k = 10.0 / max(self.distance_threshold, 1e-6)
        return round(1.0 / (1.0 + math.exp(k * (distance - self.distance_threshold))), 6)
