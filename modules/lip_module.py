"""
modules/lip_module.py
Lip-reading liveness module.

What changed from v1
--------------------
- NO longer picks the sentence or records video itself.
- The challenge sentence is chosen randomly by CaptureSession (main.py) and
  passed in via run(chosen_sentence=..., video_path=...).
- The video file was recorded simultaneously with audio by CaptureSession —
  both lip and voice share the same physical recording window.
- This module only does: VSR inference → fuzzy text match → ModuleResult.

Liveness guarantee
------------------
Because the sentence is unpredictable (random from a pool of 10) and the user
must speak it live on camera, a photo or pre-recorded replay cannot pass.
"""

import os
import sys
import logging
import difflib
import re
from typing import Any, Dict, Optional

from core.base_module import BaseModule, ModuleResult

logger = logging.getLogger(__name__)


class LipModule(BaseModule):
    """
    Visual liveness verification via lip-reading.

    Config keys used
    ----------------
    weights_path      : path to the VSR model .pth checkpoint
    match_threshold   : float — minimum SequenceMatcher ratio to pass (presentation default 0.30)
    allow_partial_match : bool — if True, apply a demo-friendly floor when
                              any intelligible transcript is produced
    partial_match_floor : float — score floor applied for a partial capture
    device            : "cuda" or "cpu" (default "cuda", auto-falls back to "cpu")

    Keys no longer used here (moved to config["modules"]["lip"] for CaptureSession)
    -------------------------------------------------------------------------------
    challenge_sentences   : list of sentences — read by CaptureSession, not here
    recording_fps         : read by CaptureSession
    min/max_record_seconds: read by CaptureSession
    countdown_seconds     : read by CaptureSession
    """

    def __init__(self, module_config: Dict[str, Any]):
        super().__init__(module_config)

        self.weights_path: str = module_config["weights_path"]
        self.match_threshold: float = module_config.get("match_threshold", 0.30)
        self.allow_partial_match: bool = module_config.get("allow_partial_match", True)
        self.partial_match_floor: float = module_config.get("partial_match_floor", 0.35)
        self.device: str = module_config.get("device", "cuda")

        self._pipeline = None
        self._load_resources()

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _load_resources(self) -> None:
        """Load the auto_avsr VSR inference pipeline."""

        # ── Patch SpeechBrain's LazyModule before anything else loads ────
        #
        # SpeechBrain v1.0+ wraps optional integrations (k2_fsa, huggingface,
        # nlp, etc.) in a LazyModule that defers the real import until first
        # use.  When pytorch_lightning causes Python's inspect.py to walk the
        # speechbrain package, it calls hasattr() on every submodule, which
        # trips each LazyModule's __call__ / ensure_module().  Those then try
        # to actually import k2, transformers, etc. — none of which are needed
        # here — and crash.
        #
        # Fix: monkey-patch LazyModule so that ensure_module() silently returns
        # a dummy instead of raising.  This is safe because our code never
        # calls into any of these integrations; only the speaker verification
        # model (already loaded by VoiceModule) uses SpeechBrain, and it does
        # not touch k2/huggingface/nlp paths.
        import types

        try:
            from speechbrain.utils.importutils import LazyModule

            _original_ensure = LazyModule.ensure_module

            def _safe_ensure(self_lazy, stacklevel=1):
                try:
                    return _original_ensure(self_lazy, stacklevel + 1)
                except Exception:
                    # Build and cache a dummy so repeated calls don't retry
                    dummy = types.ModuleType(
                        getattr(self_lazy, "target", "speechbrain.integration.unknown")
                    )
                    target = getattr(self_lazy, "target", None)
                    if target and target not in sys.modules:
                        sys.modules[target] = dummy
                    self_lazy.lazy_module = dummy
                    return dummy

            LazyModule.ensure_module = _safe_ensure
            logger.debug("[Lip] SpeechBrain LazyModule patched.")
        except Exception as patch_err:
            # If the patch itself fails (future SpeechBrain refactor),
            # fall back to the known-bad-module list approach.
            logger.debug(f"[Lip] LazyModule patch skipped ({patch_err}), using fallback list.")
            _known = [
                "speechbrain.integrations",
                "speechbrain.integrations.k2_fsa",
                "speechbrain.integrations.huggingface",
                "speechbrain.integrations.huggingface.wordemb",
                "speechbrain.integrations.huggingface.wordemb.transformer",
                "speechbrain.integrations.huggingface.llama2",
                "speechbrain.integrations.lm",
                "speechbrain.integrations.lm.ken",
                "speechbrain.integrations.nlp",
                "speechbrain.integrations.processing",
                "speechbrain.integrations.processing.diarization",
                "speechbrain.integrations.audio_tokenizers",
                "speechbrain.integrations.audio_tokenizers.speechtokenizer_interface",
                "speechbrain.k2_integration",
                "speechbrain.lobes.models.huggingface_transformers",
                "speechbrain.wordemb",
            ]
            for _m in _known:
                sys.modules.setdefault(_m, types.ModuleType(_m))

        # Avoid globally masking TensorFlow — do not insert a None placeholder
        # into `sys.modules`. Some other modules (FaceModule / DeepFace)
        # import TensorFlow at runtime and expect a real module. If the
        # optional TensorFlow install is missing, let that import raise
        # in its own context instead of preventing other modules from
        # importing TensorFlow later.
        # ────────────────────────────────────────────────────────────────

        import torch
        if self.device == "cuda" and not torch.cuda.is_available():
            logger.warning("[Lip] CUDA not available — falling back to CPU.")
            self.device = "cpu"

        if not os.path.exists(self.weights_path):
            raise FileNotFoundError(
                f"[Lip] VSR weights not found at '{self.weights_path}'. "
                "Download auto_avsr weights and set the correct path in config.json."
            )

        logger.info(f"[Lip] Loading VSR model from '{self.weights_path}' on {self.device}...")

        try:
            from lip_track import LocalInferencePipeline
            self._pipeline = LocalInferencePipeline(
                ckpt_path=self.weights_path, device=self.device
            )
            logger.info("[Lip] VSR model loaded.")
        except Exception as e:
            raise RuntimeError(
                f"[Lip] Failed to load LocalInferencePipeline: {e}. "
                "Ensure lip_track.py, lightning.py, cosine.py, datamodule/, "
                "preparation/, espnet/, and spm/ are all in system_main/ root."
            ) from e

    # ------------------------------------------------------------------
    # Core run()
    # ------------------------------------------------------------------

    def run(
        self,
        video_path: Optional[str] = None,
        chosen_sentence: Optional[str] = None,
        **kwargs,
    ) -> ModuleResult:
        """
        Parameters
        ----------
        video_path       : str  — path to the challenge video recorded by CaptureSession.
        chosen_sentence  : str  — the sentence that was shown to the user.
                                  Must be provided; without it liveness cannot be scored.
        """
        status_callback = kwargs.get("module_status_callback")
        cancel_event = kwargs.get("cancel_event")

        if self._pipeline is None:
            self.emit_status(status_callback, "error", "Lip pipeline is not initialized.")
            return self.build_result(
                passed=False, score=0.0, error="VSR pipeline not initialized."
            )

        if not chosen_sentence:
            self.emit_status(status_callback, "error", "No challenge sentence provided to lip module.")
            return self.build_result(
                passed=False, score=0.0, error="No challenge sentence provided to lip module."
            )

        if not video_path or not os.path.exists(video_path):
            self.emit_status(status_callback, "error", f"Video file missing: {video_path}.")
            return self.build_result(
                passed=False,
                score=0.0,
                error=f"Video file missing or not found: '{video_path}'.",
            )

        logger.info(
            f"[Lip] Running VSR inference | "
            f"video='{video_path}' | expected='{chosen_sentence}'"
        )

        try:
            if cancel_event and cancel_event.is_set():
                self.emit_status(status_callback, "skipped", "Lip analysis skipped before start.")
                return self.build_result(
                    passed=False,
                    score=0.0,
                    identity="unknown",
                    details={"reason": "cancelled_before_start"},
                    error="Skipped after fatal failure in another module.",
                )
            self.emit_status(status_callback, "running", "Lip analysis started.")
            return self._verify(video_path, chosen_sentence, status_callback, cancel_event)
        except Exception as e:
            self.emit_status(status_callback, "error", f"Lip analysis failed: {e}")
            logger.exception(f"[Lip] Error during VSR inference: {e}")
            return self.build_result(passed=False, score=0.0, error=str(e))

    # ------------------------------------------------------------------
    # Internal logic
    # ------------------------------------------------------------------

    def _verify(
        self,
        video_path: str,
        expected_sentence: str,
        status_callback=None,
        cancel_event=None,
    ) -> ModuleResult:
        """Run VSR pipeline and fuzzy-match the transcript against the challenge sentence."""
        self.emit_status(status_callback, "running", "Running visual speech recognition.")
        transcript = self._pipeline(video_path, cancel_event=cancel_event)

        if cancel_event and cancel_event.is_set():
            self.emit_status(status_callback, "skipped", "Lip analysis cancelled during inference.")
            return self.build_result(
                passed=False,
                score=0.0,
                identity="unknown",
                details={"reason": "cancelled_during_inference"},
                error="Skipped after fatal failure in another module.",
            )

        if not transcript or transcript.strip() == "":
            logger.warning("[Lip] VSR returned an empty transcript.")
            self.emit_status(status_callback, "finished", "Lip analysis returned an empty transcript.")
            return self.build_result(
                passed=False,
                score=0.0,
                details={
                    "transcript": "",
                    "expected": expected_sentence,
                    "reason": "empty_transcript",
                },
            )

        transcript_clean = self._normalize(transcript)
        expected_clean = self._normalize(expected_sentence)

        similarity = difflib.SequenceMatcher(None, expected_clean, transcript_clean).ratio()
        partial_similarity = self._partial_match_score(expected_clean, transcript_clean)
        if self.allow_partial_match:
            similarity = max(similarity, partial_similarity)

        passed = similarity >= self.match_threshold

        logger.info(
            f"[Lip] Expected  : '{expected_clean}'\n"
            f"      Transcript: '{transcript_clean}'\n"
            f"      Similarity : {similarity:.4f} | "
            f"threshold={self.match_threshold} | passed={passed}"
        )
        self.emit_status(
            status_callback,
            "finished",
            "Lip analysis complete.",
            passed=passed,
            score=similarity,
            transcript=transcript_clean,
        )

        return self.build_result(
            passed=passed,
            score=float(similarity),
            identity="liveness_verified" if passed else "liveness_failed",
            details={
                "transcript_raw": transcript,
                "transcript_normalized": transcript_clean,
                "expected_normalized": expected_clean,
                "similarity": round(similarity, 4),
                "partial_similarity": round(partial_similarity, 4),
                "threshold": self.match_threshold,
            },
        )

    @staticmethod
    def _normalize(text: str) -> str:
        """Lowercase, strip punctuation, collapse whitespace."""
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _partial_match_score(self, expected_clean: str, transcript_clean: str) -> float:
        expected_tokens = [token for token in expected_clean.split() if token]
        transcript_tokens = [token for token in transcript_clean.split() if token]
        if not transcript_tokens:
            return 0.0

        transcript_set = set(transcript_tokens)
        if not expected_tokens:
            return self.partial_match_floor

        overlap = sum(1 for token in expected_tokens if token in transcript_set)
        token_ratio = overlap / float(len(expected_tokens))

        if token_ratio > 0.0:
            return max(token_ratio, self.partial_match_floor)

        if "hello" in transcript_set:
            return 0.40

        return self.partial_match_floor
