"""
core/capture_session.py
Shared capture session for camera + microphone input.
"""

import logging
import os
import random
import tempfile
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CaptureResult:
    chosen_sentence: str = ""
    video_path: Optional[str] = None
    audio_path: Optional[str] = None
    frames: List[np.ndarray] = field(default_factory=list)
    face_abort: bool = False
    error: Optional[str] = None
    cancelled: bool = False


class CaptureSession:
    """
    Manages one end-to-end hardware capture session.
    """

    def __init__(
        self,
        system_cfg: Dict[str, Any],
        lip_cfg: Dict[str, Any],
        face_cfg: Dict[str, Any],
        status_callback: Optional[Callable[..., None]] = None,
        preview_callback: Optional[Callable[[np.ndarray], None]] = None,
        stop_event: Optional[threading.Event] = None,
    ):
        self.camera_index: int = system_cfg.get("camera_index", 0)

        self.sentences: List[str] = lip_cfg.get(
            "challenge_sentences",
            ["the quick brown fox jumps over the lazy dog"],
        )
        self.recording_fps: int = lip_cfg.get("recording_fps", 25)
        self.min_record_secs: int = lip_cfg.get("min_record_seconds", 4)
        self.max_record_secs: int = lip_cfg.get("max_record_seconds", 10)
        self.countdown_secs: int = lip_cfg.get("countdown_seconds", 3)

        self.max_faces_allowed: int = face_cfg.get("max_faces_allowed", 1)
        self.tracking_fps: int = face_cfg.get("tracking_fps", 5)

        self._video_path: Optional[str] = None
        self._audio_path: Optional[str] = None
        self._status_callback = status_callback
        self._preview_callback = preview_callback
        self._stop_event = stop_event or threading.Event()
        self._face_detector = None
        self._face_detector_name = "unavailable"
        self._face_detector_failures = 0

    def run(self) -> CaptureResult:
        result = CaptureResult()

        result.chosen_sentence = random.choice(self.sentences)
        self._emit_status(
            "challenge_selected",
            "Challenge sentence selected.",
            chosen_sentence=result.chosen_sentence,
        )

        if not self._show_challenge(result.chosen_sentence):
            result.cancelled = True
            result.error = "Capture cancelled by user."
            self._emit_status("cancelled", result.error)
            return result

        word_count = len(result.chosen_sentence.split())
        duration = max(self.min_record_secs, min(self.max_record_secs, word_count // 2 + 2))

        logger.info(
            f"[Capture] Sentence chosen: '{result.chosen_sentence}' | Recording for {duration}s"
        )
        self._emit_status("recording_prepared", "Capture session prepared.", duration=duration)

        video_error: List[str] = []
        audio_error: List[str] = []
        frames_list: List[np.ndarray] = []
        face_abort_flag = [False]

        self._video_path = tempfile.mktemp(suffix="_challenge.avi")
        self._audio_path = tempfile.mktemp(suffix="_challenge.wav")

        audio_thread = threading.Thread(
            target=self._record_audio,
            args=(self._audio_path, duration, audio_error),
            name="audio_recorder",
        )
        video_thread = threading.Thread(
            target=self._record_video,
            args=(self._video_path, duration, frames_list, face_abort_flag, video_error),
            name="video_recorder",
        )

        print("\n  [●] Recording started - speak the sentence clearly...\n")
        self._emit_status("recording_started", "Recording started.", duration=duration)
        audio_thread.start()
        video_thread.start()

        audio_thread.join()
        video_thread.join()

        print("\n  [■] Recording complete.\n")
        self._emit_status("recording_finished", "Recording finished.")

        if video_error:
            result.error = f"Video capture failed: {video_error[0]}"
            logger.error(f"[Capture] {result.error}")
            self._emit_status("error", result.error)
            return result

        if audio_error:
            result.error = f"Audio capture failed: {audio_error[0]}"
            logger.error(f"[Capture] {result.error}")
            self._emit_status("error", result.error)
            return result

        result.video_path = self._video_path
        result.audio_path = self._audio_path
        result.frames = frames_list
        result.face_abort = face_abort_flag[0]

        if result.face_abort:
            logger.warning("[Capture] HARD ABORT: multiple faces detected during capture.")
            self._emit_status("face_abort", "Multiple faces detected during capture.")
        elif self._should_stop():
            result.cancelled = True
            result.error = "Capture cancelled by user."
            self._emit_status("cancelled", result.error)
        else:
            self._emit_status(
                "capture_complete",
                "Capture complete.",
                frames=len(result.frames),
                audio_path=result.audio_path,
                video_path=result.video_path,
            )

        return result

    def cleanup(self, result: CaptureResult) -> None:
        for path in [result.video_path, result.audio_path]:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                    logger.debug(f"[Capture] Cleaned up temp file: {path}")
                except OSError as e:
                    logger.warning(f"[Capture] Could not delete {path}: {e}")

    def _show_challenge(self, sentence: str) -> bool:
        border = "-" * 58
        print(f'\n  +{border}+')
        print("  |  SECURITY CHALLENGE - read this sentence aloud:        |")
        print("  |                                                          |")
        print(f'  |  "{sentence}"')
        print("  |                                                          |")
        print(f"  +{border}+\n")

        self._emit_status("countdown_started", "Countdown started.", seconds=self.countdown_secs)
        for i in range(self.countdown_secs, 0, -1):
            if self._should_stop():
                return False
            self._emit_status("countdown_tick", f"Starting in {i}...", seconds_remaining=i)
            print(f"  Starting in {i}...", end="\r", flush=True)
            time.sleep(1)
        print(" " * 30, end="\r")
        return True

    def _record_video(
        self,
        output_path: str,
        duration: int,
        frames_out: list,
        face_abort_flag: list,
        error_out: list,
    ) -> None:
        face_detector = self._get_face_detector()

        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            error_out.append(f"Cannot open camera index {self.camera_index}.")
            return

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter(output_path, fourcc, self.recording_fps, (width, height))

        total_frames = self.recording_fps * duration
        track_every_n = max(1, self.recording_fps // self.tracking_fps)
        frame_count = 0
        aborted = False

        try:
            while frame_count < total_frames and not self._should_stop():
                ret, frame = cap.read()
                if not ret:
                    break

                writer.write(frame)
                self._emit_preview(frame)

                if frame_count % track_every_n == 0:
                    frames_out.append(frame.copy())

                    detected_count = self._count_faces(face_detector, frame)
                    if detected_count is None:
                        pass
                    elif detected_count > self.max_faces_allowed:
                        logger.warning(
                            f"[Capture] {detected_count} faces in tracking frame "
                            f"{frame_count} - hard abort triggered."
                        )
                        self._emit_status(
                            "face_abort",
                            "Multiple faces detected during capture.",
                            frame_index=frame_count,
                            face_count=detected_count,
                            detector=self._face_detector_name,
                        )
                        face_abort_flag[0] = True
                        aborted = True
                        break

                frame_count += 1
        finally:
            cap.release()
            writer.release()

        if aborted:
            logger.warning("[Capture] Video recording aborted early due to multi-face detection.")

    def _record_audio(self, output_path: str, duration: int, error_out: list) -> None:
        try:
            import sounddevice as sd
            import scipy.io.wavfile as wav_io
        except ImportError as e:
            error_out.append(f"sounddevice/scipy not installed: {e}")
            return

        sample_rate = 16000
        try:
            chunk_size = 1600
            total_samples = int(duration * sample_rate)
            chunks = []

            with sd.InputStream(
                samplerate=sample_rate,
                channels=1,
                dtype="int16",
                blocksize=chunk_size,
            ) as stream:
                captured = 0
                while captured < total_samples and not self._should_stop():
                    frames_to_read = min(chunk_size, total_samples - captured)
                    data, _overflowed = stream.read(frames_to_read)
                    chunks.append(data.copy())
                    captured += len(data)

            recording = (
                np.concatenate(chunks, axis=0)
                if chunks
                else np.zeros((0, 1), dtype="int16")
            )
            wav_io.write(output_path, sample_rate, recording)
            logger.debug(f"[Capture] Audio saved to '{output_path}'.")
        except Exception as e:
            error_out.append(str(e))
            logger.error(f"[Capture] Audio recording failed: {e}")

    def _emit_status(self, stage: str, message: str, **details: Any) -> None:
        if callable(self._status_callback):
            self._status_callback(stage=stage, message=message, details=details)

    def _emit_preview(self, frame: np.ndarray) -> None:
        if callable(self._preview_callback):
            self._preview_callback(frame.copy())

    def _should_stop(self) -> bool:
        return bool(self._stop_event and self._stop_event.is_set())

    def _get_face_detector(self):
        if self._face_detector is not None:
            return self._face_detector

        try:
            from mtcnn import MTCNN

            self._face_detector = MTCNN()
            self._face_detector_name = "mtcnn"
            logger.info("[Capture] Using MTCNN for capture-time face counting.")
            return self._face_detector
        except Exception as exc:
            logger.warning(
                "[Capture] MTCNN unavailable for capture-time face counting: %s", exc
            )

        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        cascade = cv2.CascadeClassifier(cascade_path)
        if cascade.empty():
            logger.warning("[Capture] OpenCV Haar cascade unavailable; face counting disabled.")
            self._face_detector = False
            self._face_detector_name = "disabled"
            return self._face_detector

        self._face_detector = cascade
        self._face_detector_name = "opencv_haar"
        logger.info("[Capture] Using OpenCV Haar cascade for capture-time face counting.")
        return self._face_detector

    def _count_faces(self, detector, frame: np.ndarray) -> Optional[int]:
        if detector is False:
            return None

        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            logger.debug("[Capture] Skipping empty frame during face counting.")
            return 0

        try:
            if self._face_detector_name == "mtcnn":
                detections = detector.detect_faces(frame)

                if detections is None:
                    return 0

                if isinstance(detections, np.ndarray):
                    if detections.size == 0:
                        return 0
                    return int(detections.shape[0])

                if isinstance(detections, (list, tuple)):
                    if len(detections) == 0:
                        return 0

                    # Filter out malformed empty detections before counting.
                    valid = []
                    for item in detections:
                        if item is None:
                            continue
                        if isinstance(item, dict) and not item.get("box"):
                            continue
                        valid.append(item)
                    return len(valid)

                return 0

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(40, 40),
            )
            return int(len(faces))
        except Exception as det_err:
            err_text = str(det_err)
            self._face_detector_failures += 1

            # MTCNN sometimes forwards an empty [0, H, W, C] batch into TensorFlow.
            # Treat that as "no face in this frame" rather than a fatal tracking error.
            if (
                self._face_detector_name == "mtcnn"
                and (
                    "Incompatible shapes" in err_text
                    or "[0,48,48,3]" in err_text
                    or "batch dimension" in err_text.lower()
                )
            ):
                logger.debug(
                    "[Capture] MTCNN produced an empty detection batch; skipping frame."
                )
                return 0

            logger.debug(f"[Capture] Face detector error during tracking: {det_err}")

            # If MTCNN becomes unstable during capture, fall back to Haar so the session continues.
            if self._face_detector_name == "mtcnn" and self._face_detector_failures >= 3:
                logger.warning(
                    "[Capture] MTCNN failed repeatedly during capture; switching to OpenCV Haar cascade."
                )
                cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                cascade = cv2.CascadeClassifier(cascade_path)
                if not cascade.empty():
                    self._face_detector = cascade
                    self._face_detector_name = "opencv_haar"
                    self._face_detector_failures = 0
                    return self._count_faces(self._face_detector, frame)

            return None
