"""
main.py
TriGuard Security System entry point.
"""

import argparse
import logging
import sys
from datetime import datetime
from typing import Any, Dict

from core.capture_session import CaptureSession
from core.config_loader import load_config
from core.logger import setup_logger
from core.orchestrator import AuthOrchestrator

logger = logging.getLogger(__name__)


def build_modules(config: Dict[str, Any], exit_on_error: bool = True) -> Dict[str, Any]:
    """
    Instantiate only modules that are enabled in config.json.
    A broken dependency in a disabled module never blocks startup.
    """
    modules = {}
    mcfg = config["modules"]

    failed_modules = []

    if mcfg["face"]["enabled"]:
        try:
            from modules.face_module import FaceModule

            modules["face"] = FaceModule(mcfg["face"])
            logger.info("[Main] FaceModule ready.")
        except Exception as e:
            logger.error(f"[Main] FaceModule failed to load: {e}")
            if exit_on_error:
                sys.exit(1)
            failed_modules.append("face")

    if mcfg["voice"]["enabled"]:
        try:
            from modules.voice_module import VoiceModule

            modules["voice"] = VoiceModule(mcfg["voice"])
            logger.info("[Main] VoiceModule ready.")
        except Exception as e:
            logger.error(f"[Main] VoiceModule failed to load: {e}")
            if exit_on_error:
                sys.exit(1)
            failed_modules.append("voice")

    if mcfg["lip"]["enabled"]:
        try:
            from modules.lip_module import LipModule

            modules["lip"] = LipModule(mcfg["lip"])
            logger.info("[Main] LipModule ready.")
        except Exception as e:
            logger.error(f"[Main] LipModule failed to load: {e}")
            if exit_on_error:
                sys.exit(1)
            failed_modules.append("lip")

    if not modules:
        logger.error("[Main] No modules loaded - enable at least one in config.json.")
        if exit_on_error:
            sys.exit(1)
        raise RuntimeError("No modules loaded - enable at least one in config.json.")

    if failed_modules and not exit_on_error:
        _apply_runtime_weights(config, modules)
        logger.warning(
            "[Main] Continuing without failed module(s): %s",
            ", ".join(failed_modules),
        )

    return modules


def _apply_runtime_weights(config: Dict[str, Any], modules: Dict[str, Any]) -> None:
    """Redistribute weights across only the modules that actually loaded."""
    active_names = list(modules.keys())
    if not active_names:
        return

    configured_total = sum(config["modules"][name]["weight"] for name in active_names)
    if configured_total <= 0:
        even_weight = 1.0 / len(active_names)
        for name, module in modules.items():
            module.effective_weight = even_weight
            module.config["effective_weight"] = even_weight
        return

    for name, module in modules.items():
        effective_weight = config["modules"][name]["weight"] / configured_total
        module.effective_weight = round(effective_weight, 6)
        module.config["effective_weight"] = module.effective_weight


def run_auth_session(config: Dict[str, Any], modules: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run the full authentication session:
    1. Shared capture
    2. Build the inputs dict for the orchestrator
    3. Execute all modules in parallel
    4. Return the result dict
    """
    sys_cfg = config["system"]
    lip_cfg = config["modules"]["lip"]
    face_cfg = config["modules"]["face"]

    session = CaptureSession(
        system_cfg=sys_cfg,
        lip_cfg=lip_cfg,
        face_cfg=face_cfg,
    )

    capture = session.run()

    if capture.error:
        logger.error(f"[Main] Capture session failed: {capture.error}")
        print(f"\n  [ERROR] Hardware capture failed: {capture.error}")
        sys.exit(1)

    if capture.face_abort:
        logger.warning("[Main] Authentication aborted - multiple faces detected during capture.")
        print("\n" + "=" * 60)
        print("  AUTHENTICATION ABORTED")
        print("  Reason: More than one face detected during the challenge.")
        print("  Only the registered user should be in front of the camera.")
        print("=" * 60 + "\n")
        session.cleanup(capture)
        return {
            "authenticated": False,
            "identity": "unknown",
            "final_score": 0.0,
            "results": {},
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "summary": "ABORTED - multiple faces",
        }

    inputs: Dict[str, Any] = {
        "frames": capture.frames,
        "face_abort": capture.face_abort,
        "audio_path": capture.audio_path,
        "video_path": capture.video_path,
        "chosen_sentence": capture.chosen_sentence,
    }

    logger.info(
        f"[Main] Capture complete | sentence='{capture.chosen_sentence}' | "
        f"frames={len(capture.frames)} | "
        f"audio='{capture.audio_path}' | video='{capture.video_path}'"
    )

    orchestrator = AuthOrchestrator(config, modules)
    result = orchestrator.authenticate(inputs)
    session.cleanup(capture)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TriGuard - Multi-Modal Security Authentication System"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.json",
        help="Path to config.json (default: config.json)",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    sys_cfg = config["system"]
    setup_logger(
        log_dir=sys_cfg.get("log_dir", "logs/"),
        log_level=sys_cfg.get("log_level", "INFO"),
    )

    logger.info(
        f"[Main] {sys_cfg.get('name', 'TriGuard')} "
        f"v{sys_cfg.get('version', '1.1.0')} starting..."
    )
    logger.info(f"[Main] Auth threshold: {sys_cfg.get('auth_pass_threshold', 0.70)}")

    modules = build_modules(config, exit_on_error=True)
    result = run_auth_session(config, modules)
    sys.exit(0 if result["authenticated"] else 1)


if __name__ == "__main__":
    main()
