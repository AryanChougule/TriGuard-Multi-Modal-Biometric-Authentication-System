"""
core/orchestrator.py
Runs all enabled modules in parallel, collects ModuleResults, computes the
final weighted score, and makes the authentication decision.
"""

import concurrent.futures
import logging
import threading
from datetime import datetime
from typing import Any, Dict

from core.base_module import ModuleResult

logger = logging.getLogger(__name__)


class AuthOrchestrator:
    """
    Coordinates parallel execution of all enabled authentication modules.
    Combines weighted scores and produces the final auth verdict.
    """

    def __init__(self, config: Dict[str, Any], modules: Dict[str, Any]):
        self.config = config
        self.system_cfg = config["system"]
        self.modules = modules
        self.pass_threshold: float = self.system_cfg.get("auth_pass_threshold", 0.70)

    def authenticate(self, inputs: Dict[str, Any], status_callback=None) -> Dict[str, Any]:
        logger.info("=" * 60)
        logger.info("Authentication session started.")
        logger.info("Active modules: %s", list(self.modules.keys()))

        results: Dict[str, ModuleResult] = {}
        cancel_event = threading.Event()
        fatal_error: str | None = None
        future_to_name: Dict[concurrent.futures.Future, str] = {}

        executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=len(self.modules),
            thread_name_prefix="auth_module",
        )
        try:
            future_to_name = {
                executor.submit(
                    self._run_module,
                    name,
                    mod,
                    inputs,
                    status_callback,
                    cancel_event,
                ): name
                for name, mod in self.modules.items()
            }

            for future in concurrent.futures.as_completed(future_to_name):
                name = future_to_name[future]
                try:
                    result = future.result()
                    results[name] = result
                    logger.info(str(result))
                    if self._is_fatal_result(result):
                        fatal_error = result.error or f"Fatal failure in module '{name}'."
                        cancel_event.set()
                        self._mark_skipped_modules(
                            results=results,
                            future_to_name=future_to_name,
                            completed_module=name,
                            status_callback=status_callback,
                            reason=f"Skipped after fatal failure in {name}.",
                        )
                        break
                except Exception as exc:
                    logger.error("Module '%s' raised an unhandled exception: %s", name, exc)
                    results[name] = ModuleResult(
                        module_name=name,
                        passed=False,
                        score=0.0,
                        weighted_score=0.0,
                        error=str(exc),
                    )
                    fatal_error = str(exc)
                    cancel_event.set()
                    self._mark_skipped_modules(
                        results=results,
                        future_to_name=future_to_name,
                        completed_module=name,
                        status_callback=status_callback,
                        reason=f"Skipped after fatal exception in {name}.",
                    )
                    break
        finally:
            if cancel_event.is_set():
                for future in future_to_name:
                    future.cancel()
                executor.shutdown(wait=False, cancel_futures=True)
            else:
                executor.shutdown(wait=True)

        if fatal_error:
            logger.warning("Authentication aborted early due to fatal module failure: %s", fatal_error)
            return self._build_response(
                results=results,
                final_score=0.0,
                authenticated=False,
                identity="unknown",
                liveness_ok=False,
                fatal_error=fatal_error,
            )

        final_score = sum(r.weighted_score for r in results.values())
        identity = self._resolve_identity(results)

        liveness_ok = True
        if "lip" in results:
            liveness_ok = results["lip"].passed
            if not liveness_ok:
                logger.warning("Liveness check FAILED: lip module did not pass.")

        authenticated = (final_score >= self.pass_threshold) and liveness_ok
        return self._build_response(
            results=results,
            final_score=final_score,
            authenticated=authenticated,
            identity=identity,
            liveness_ok=liveness_ok,
        )

    def _build_response(
        self,
        results: Dict[str, ModuleResult],
        final_score: float,
        authenticated: bool,
        identity: str,
        liveness_ok: bool,
        fatal_error: str | None = None,
    ) -> Dict[str, Any]:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        verdict = "AUTHENTICATED" if authenticated else "ACCESS DENIED"

        summary_lines = [
            "",
            "=" * 60,
            f"  AUTHENTICATION RESULT - {timestamp}",
            "=" * 60,
        ]
        if fatal_error:
            summary_lines.append(f"  EARLY ABORT   : {fatal_error}")

        for name, result in results.items():
            bar = self._score_bar(result.score)
            status = "PASS" if result.passed else "FAIL"
            err_note = f"  ! {result.error}" if result.error else ""
            summary_lines.append(
                f"  [{name.upper():6s}]  {status}  {bar}  "
                f"score={result.score:.3f}  weighted={result.weighted_score:.3f}  "
                f"id={result.identity}{err_note}"
            )

        summary_lines += [
            "-" * 60,
            f"  Final Score   : {final_score:.4f}  (threshold: {self.pass_threshold})",
            f"  Liveness OK   : {liveness_ok}",
            f"  Identity      : {identity}",
            f"  {verdict}",
            "=" * 60,
            "",
        ]
        summary = "\n".join(summary_lines)
        logger.info(summary)

        return {
            "results": results,
            "final_score": final_score,
            "authenticated": authenticated,
            "identity": identity,
            "timestamp": timestamp,
            "summary": summary,
        }

    def _run_module(
        self,
        name: str,
        module,
        inputs: Dict[str, Any],
        status_callback=None,
        cancel_event: threading.Event | None = None,
    ) -> ModuleResult:
        logger.debug("[%s] Starting module execution...", name)
        if callable(status_callback):
            status_callback(name, "starting", "Module thread started.", {})

        module_inputs = dict(inputs)
        module_inputs["cancel_event"] = cancel_event
        if callable(status_callback):
            module_inputs["module_status_callback"] = (
                lambda stage, message, details=None, module_name=name: status_callback(
                    module_name,
                    stage,
                    message,
                    details or {},
                )
            )

        result = module.run(**module_inputs)

        if callable(status_callback):
            stage = "failed" if result.error else "finished"
            status_callback(
                name,
                stage,
                f"Module completed with {'PASS' if result.passed else 'FAIL'}.",
                {
                    "passed": result.passed,
                    "score": result.score,
                    "identity": result.identity,
                    "error": result.error,
                },
            )
        logger.debug("[%s] Module execution complete.", name)
        return result

    def _mark_skipped_modules(
        self,
        results: Dict[str, ModuleResult],
        future_to_name: Dict[concurrent.futures.Future, str],
        completed_module: str,
        status_callback=None,
        reason: str = "Skipped.",
    ) -> None:
        for future, module_name in future_to_name.items():
            if module_name == completed_module or module_name in results:
                continue

            results[module_name] = ModuleResult(
                module_name=module_name,
                passed=False,
                score=0.0,
                weighted_score=0.0,
                identity="unknown",
                details={"reason": "skipped_due_to_fatal_failure"},
                error="Skipped after fatal failure in another module.",
            )
            if callable(status_callback):
                status_callback(module_name, "skipped", reason, {})
            future.cancel()

    @staticmethod
    def _is_fatal_result(result: ModuleResult) -> bool:
        return bool(result.error)

    def _resolve_identity(self, results: Dict[str, ModuleResult]) -> str:
        preference_order = ["face", "voice", "lip"]
        votes: Dict[str, float] = {}

        for name, result in results.items():
            if result.passed and result.identity and result.identity != "unknown":
                votes[result.identity] = votes.get(result.identity, 0.0) + result.weighted_score

        if not votes:
            return "unknown"

        best = max(votes, key=lambda key: votes[key])
        top_score = votes[best]
        top_candidates = [key for key, value in votes.items() if value == top_score]
        if len(top_candidates) == 1:
            return best

        for preferred_module in preference_order:
            if preferred_module in results and results[preferred_module].identity in top_candidates:
                return results[preferred_module].identity

        return best

    @staticmethod
    def _score_bar(score: float, width: int = 10) -> str:
        filled = int(round(score * width))
        return f"[{'#' * filled}{'.' * (width - filled)}]"
