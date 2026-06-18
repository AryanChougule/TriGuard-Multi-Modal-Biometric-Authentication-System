#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2021 Imperial College London (Pingchuan Ma)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import warnings

from face_alignment import FaceAlignment, LandmarksType

warnings.filterwarnings("ignore")


class LandmarksDetector:
    def __init__(self, device="cuda:0", model_name="resnet50"):
        self.device = device
        self.detector = FaceAlignment(
            LandmarksType.TWO_D,
            device=device,
            face_detector="sfd",
            compile=False,
        )

    def __call__(self, video_frames):
        landmarks = []
        for frame in video_frames:
            faces = self.detector.get_landmarks(frame)
            if not faces:
                landmarks.append(None)
                continue

            best_face = None
            best_area = -1
            for face_landmarks in faces:
                x_min, y_min = face_landmarks[:, 0].min(), face_landmarks[:, 1].min()
                x_max, y_max = face_landmarks[:, 0].max(), face_landmarks[:, 1].max()
                area = (x_max - x_min) * (y_max - y_min)
                if area > best_area:
                    best_area = area
                    best_face = face_landmarks

            landmarks.append(best_face)
        return landmarks
