#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2021 Imperial College London (Pingchuan Ma)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import warnings

from face_alignment import FaceAlignment, LandmarksType
import numpy as np
warnings.filterwarnings("ignore")


class LandmarksDetector:
    def __init__(self, device="cuda:0", model_name="resnet50", smoothing_alpha=0.65):
        self.device = device
        self.smoothing_alpha = smoothing_alpha
        self.detector = FaceAlignment(
            LandmarksType.TWO_D,
            device=device,
            face_detector="sfd",
            compile=False,
        )

    def __call__(self, video_frames, cancel_event=None):
        landmarks = []
        previous_landmarks = None
        for frame in video_frames:
            if cancel_event and cancel_event.is_set():
                break

            faces = self.detector.get_landmarks(frame)
            if not faces:
                landmarks.append(None)
                continue

            best_face = None
            best_score = None
            for face_landmarks in faces:
                x_min, y_min = face_landmarks[:, 0].min(), face_landmarks[:, 1].min()
                x_max, y_max = face_landmarks[:, 0].max(), face_landmarks[:, 1].max()
                area = (x_max - x_min) * (y_max - y_min)
                if previous_landmarks is None:
                    score = float(area)
                else:
                    current_center = face_landmarks.mean(axis=0)
                    previous_center = previous_landmarks.mean(axis=0)
                    drift = np.linalg.norm(current_center - previous_center)
                    score = float(area) - (drift * 25.0)

                if best_score is None or score > best_score:
                    best_score = score
                    best_face = face_landmarks

            if best_face is not None and previous_landmarks is not None:
                best_face = (
                    self.smoothing_alpha * previous_landmarks
                    + (1.0 - self.smoothing_alpha) * best_face
                )

            landmarks.append(best_face)
            previous_landmarks = best_face
        return landmarks
