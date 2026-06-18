"""
modules/voice_module.py
Voice authentication module.

This version follows the standalone voice identifier concept:
- load a pickled speaker embedding database,
- extract an ECAPA-TDNN embedding from audio or video input,
- rank every enrolled speaker by cosine similarity,
- accept the top speaker only if the score clears the threshold.
"""

import logging
import os
import pickle
import subprocess
import tempfile
import numbers
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torchaudio

from core.base_module import BaseModule, ModuleResult

logger = logging.getLogger(__name__)


class VoiceModule(BaseModule):
    """
    Speaker identification using SpeechBrain's ECAPA-TDNN model.

    Config keys used
    ----------------
    embeddings_path      : path to the pre-built voice_embeddings.pkl
    model_dir            : SpeechBrain model cache directory
    similarity_threshold : minimum cosine similarity to accept (presentation default 0.35)
    sample_rate          : target Hz for audio normalization
    ffmpeg_path          : optional ffmpeg executable path for video->wav conversion
    """

    def __init__(self, module_config: Dict[str, Any]):
        super().__init__(module_config)

        self.embeddings_path: str = module_config["embeddings_path"]
        self.model_dir: str = module_config["model_dir"]
        self.similarity_threshold: float = module_config.get("similarity_threshold", 0.35)
        self.sample_rate: int = module_config.get("sample_rate", 16000)
        self.ffmpeg_path: str = module_config.get("ffmpeg_path", "ffmpeg")

        self._device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self._model = None
        self._speaker_db: Dict[str, List[np.ndarray]] = {}

        self._load_resources()

    def _load_resources(self) -> None:
        try:
            from speechbrain.inference.speaker import SpeakerRecognition
        except ImportError as e:
            raise ImportError("SpeechBrain not installed. Run: pip install speechbrain") from e

        logger.info(f"[Voice] Loading ECAPA-TDNN on {self._device}...")
        self._model = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=self.model_dir,
            run_opts={"device": self._device},
        )
        logger.info("[Voice] Model loaded.")

        self._load_embeddings_pkl()

    def _load_embeddings_pkl(self) -> None:
        if not os.path.exists(self.embeddings_path):
            raise FileNotFoundError(
                f"[Voice] Embeddings file not found at '{self.embeddings_path}'. "
                "Run tools/build_voice_embeddings.py first to generate it."
            )

        with open(self.embeddings_path, "rb") as f:
            raw = pickle.load(f)

        if not isinstance(raw, dict):
            raise ValueError(
                f"[Voice] '{self.embeddings_path}' has unexpected format. "
                "Expected dict {person_name: [embedding, ...]}."
            )

        valid_speakers = 0
        for person, embeddings in raw.items():
            try:
                parsed_embeddings = self._parse_enrollment_embeddings(embeddings)
            except ValueError as exc:
                logger.warning("[Voice] '%s' has invalid embeddings - skipping: %s", person, exc)
                continue

            if not parsed_embeddings:
                logger.warning(f"[Voice] '{person}' has no embeddings - skipping.")
                continue

            self._speaker_db[person] = [
                self._normalize_vector(embedding) for embedding in parsed_embeddings
            ]
            valid_speakers += 1

        if valid_speakers == 0:
            raise ValueError(
                f"[Voice] '{self.embeddings_path}' contains no valid speaker entries."
            )

        logger.info(
            f"[Voice] Loaded {valid_speakers} speaker(s) from '{self.embeddings_path}'."
        )
        for person, embs in self._speaker_db.items():
            dims = sorted({int(emb.shape[0]) for emb in embs})
            logger.debug(f"[Voice]   {person}: {len(embs)} embedding(s), dims={dims}")

    def run(self, audio_path: Optional[str] = None, **kwargs) -> ModuleResult:
        status_callback = kwargs.get("module_status_callback")

        if not self._speaker_db:
            self.emit_status(status_callback, "error", "Speaker database is empty.")
            return self.build_result(
                passed=False,
                score=0.0,
                error="Speaker DB is empty - run tools/build_voice_embeddings.py.",
            )

        fallback_recorded = False

        if audio_path is None or not os.path.exists(audio_path):
            logger.warning(
                "[Voice] No audio_path from CaptureSession - falling back to live mic recording."
            )
            self.emit_status(status_callback, "running", "No shared audio found. Recording fallback audio.")
            audio_path = self._record_live_audio()
            if audio_path is None:
                self.emit_status(status_callback, "error", "Microphone recording failed.")
                return self.build_result(
                    passed=False,
                    score=0.0,
                    error="Microphone recording failed.",
                )
            fallback_recorded = True

        try:
            self.emit_status(status_callback, "running", "Voice analysis started.", input_path=audio_path)
            return self._verify(audio_path, status_callback)
        except Exception as e:
            self.emit_status(status_callback, "error", f"Voice analysis failed: {e}")
            logger.exception(f"[Voice] Verification error: {e}")
            return self.build_result(passed=False, score=0.0, error=str(e))
        finally:
            if fallback_recorded and os.path.exists(audio_path):
                os.remove(audio_path)

    def _verify(self, audio_path: str, status_callback=None) -> ModuleResult:
        logger.info(f"[Voice] Verifying input: '{audio_path}'")

        self.emit_status(status_callback, "running", "Extracting voice embedding.")
        test_emb = self._extract_embedding(audio_path)
        self.emit_status(status_callback, "running", "Ranking enrolled speakers.")
        ranked_results = self._identify_speaker(test_emb)
        best_match, best_score = ranked_results[0]
        top_matches = [
            {"rank": idx + 1, "speaker": name, "score": round(score, 4)}
            for idx, (name, score) in enumerate(ranked_results[:5])
        ]

        passed = best_score >= self.similarity_threshold

        logger.info(
            f"[Voice] Best match: '{best_match}' | "
            f"score={best_score:.4f} | "
            f"threshold={self.similarity_threshold} | passed={passed}"
        )
        logger.info(f"[Voice] Top matches: {top_matches}")
        self.emit_status(
            status_callback,
            "finished",
            f"Voice analysis complete. Best match: {best_match}.",
            passed=passed,
            identity=best_match,
            score=best_score,
        )

        return self.build_result(
            passed=passed,
            score=float(best_score),
            identity=best_match if passed else "unknown",
            details={
                "best_match": best_match,
                "best_score": round(best_score, 4),
                "all_scores": {name: round(score, 4) for name, score in ranked_results},
                "top_matches": top_matches,
                "threshold": self.similarity_threshold,
            },
        )

    def _identify_speaker(self, test_emb: np.ndarray) -> List[Tuple[str, float]]:
        results = []
        test_emb = self._normalize_vector(
            self._coerce_embedding_vector(test_emb, source="live_embedding")
        )
        expected_dim = int(test_emb.shape[0])

        for speaker, embeddings in self._speaker_db.items():
            scores = []
            for idx, stored in enumerate(embeddings):
                stored_vec = self._coerce_embedding_vector(
                    stored,
                    expected_dim=expected_dim,
                    source=f"db[{speaker}][{idx}]",
                )
                scores.append(self._cosine_similarity(stored_vec, test_emb))

            if scores:
                results.append((speaker, float(np.mean(scores))))

        if not results:
            known_dims = sorted(
                {
                    int(np.asarray(embedding).size)
                    for embeddings in self._speaker_db.values()
                    for embedding in embeddings
                }
            )
            raise ValueError(
                "No compatible speaker embeddings found for verification. "
                f"Live embedding dim={expected_dim}; database dims={known_dims}. "
                "Rebuild voice_embeddings.pkl with tools/build_voice_embeddings.py."
            )

        results.sort(key=lambda item: item[1], reverse=True)
        return results

    def _extract_embedding(self, audio_path: str) -> np.ndarray:
        temporary_file = None
        source_path = audio_path

        try:
            if self._is_video_file(audio_path):
                logger.info("[Voice] Video input detected. Extracting audio with FFmpeg...")
                temporary_file = self._convert_video_to_wav(audio_path)
                source_path = temporary_file

            signal, fs = torchaudio.load(source_path)

            if signal.shape[0] > 1:
                signal = signal.mean(dim=0, keepdim=True)

            if fs != self.sample_rate:
                signal = torchaudio.transforms.Resample(fs, self.sample_rate)(signal)

            with torch.no_grad():
                embedding = self._model.encode_batch(signal)

            return self._coerce_embedding_vector(
                embedding.detach().cpu().numpy(),
                source=f"live_embedding:{os.path.basename(source_path)}",
            )
        finally:
            if temporary_file and os.path.exists(temporary_file):
                os.remove(temporary_file)

    def _convert_video_to_wav(self, video_path: str) -> str:
        temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_wav.close()

        command = [
            self.ffmpeg_path,
            "-y",
            "-i",
            video_path,
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            str(self.sample_rate),
            "-ac",
            "1",
            temp_wav.name,
        ]
        result = subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "FFmpeg conversion failed. Set a valid 'ffmpeg_path' in config.json."
            )
        return temp_wav.name

    def _record_live_audio(
        self,
        filename: str = "voice_fallback_live.wav",
        duration: int = 5,
    ) -> Optional[str]:
        try:
            import sounddevice as sd
            import scipy.io.wavfile as wav_io
        except ImportError as e:
            logger.error(f"[Voice] sounddevice/scipy missing: {e}")
            return None

        logger.info(f"[Voice] Fallback mic recording for {duration}s...")
        try:
            rec = sd.rec(
                int(duration * self.sample_rate),
                samplerate=self.sample_rate,
                channels=1,
                dtype="int16",
            )
            sd.wait()
            wav_io.write(filename, self.sample_rate, rec)
            logger.info(f"[Voice] Fallback audio saved to '{filename}'.")
            return filename
        except Exception as e:
            logger.error(f"[Voice] Fallback recording failed: {e}")
            return None

    @staticmethod
    def _is_video_file(path: str) -> bool:
        return path.lower().endswith((".mp4", ".mov", ".avi", ".mkv"))

    @staticmethod
    def _to_1d_numpy(value: Any) -> np.ndarray:
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        return np.asarray(value, dtype=np.float32).flatten()

    @classmethod
    def _coerce_embedding_vector(
        cls,
        value: Any,
        expected_dim: Optional[int] = None,
        source: str = "embedding",
    ) -> np.ndarray:
        vector = cls._to_1d_numpy(value)

        if vector.ndim != 1:
            vector = vector.reshape(-1)

        if vector.size <= 1:
            raise ValueError(
                f"{source} collapsed to shape {tuple(vector.shape)}; expected a full embedding vector."
            )

        if expected_dim is not None and int(vector.size) != int(expected_dim):
            raise ValueError(
                f"{source} has dim {int(vector.size)} but expected {int(expected_dim)}."
            )

        return vector

    @classmethod
    def _parse_enrollment_embeddings(cls, value: Any) -> List[np.ndarray]:
        if value is None:
            return []

        if isinstance(value, list):
            if not value:
                return []

            # Stored as a flat numeric vector: [0.12, -0.08, ...]
            if all(isinstance(item, numbers.Number) for item in value):
                return [cls._coerce_embedding_vector(value, source="db_flat_vector")]

            return [
                cls._coerce_embedding_vector(item, source=f"db_embedding[{idx}]")
                for idx, item in enumerate(value)
            ]

        return [cls._coerce_embedding_vector(value, source="db_embedding")]

    @staticmethod
    def _normalize_vector(vector: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            return vector
        return vector / norm

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b))
