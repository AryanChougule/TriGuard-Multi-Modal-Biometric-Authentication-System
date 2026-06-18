"""
core/base_module.py
Abstract base class that every authentication module must implement.
Enforces a consistent interface so the orchestrator can call all modules uniformly.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional


@dataclass
class ModuleResult:
    """
    Standardized result returned by every module.

    Fields
    ------
    module_name     : str   — Which module produced this result.
    passed          : bool  — Did this module's own threshold check pass?
    score           : float — Raw confidence score in [0.0, 1.0].
    weighted_score  : float — score * effective_weight (set by orchestrator).
    identity        : str   — The identity string the module matched, or "unknown".
    details         : dict  — Free-form extra info (distances, transcripts, etc.).
    error           : str   — Non-empty if the module raised an exception.
    """
    module_name: str
    passed: bool = False
    score: float = 0.0
    weighted_score: float = 0.0
    identity: str = "unknown"
    details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        err = f" | ERROR: {self.error}" if self.error else ""
        return (
            f"[{self.module_name.upper()}] {status} | "
            f"Score: {self.score:.4f} | "
            f"Weighted: {self.weighted_score:.4f} | "
            f"Identity: {self.identity}{err}"
        )


class BaseModule(ABC):
    """
    Every authentication module (Face, Voice, Lip) must inherit this class
    and implement the `run()` method.
    """

    def __init__(self, module_config: Dict[str, Any]):
        self.config = module_config
        self.enabled: bool = module_config.get("enabled", True)
        self.weight: float = module_config.get("weight", 0.0)
        self.effective_weight: float = module_config.get("effective_weight", self.weight)

    @abstractmethod
    def run(self, **kwargs) -> ModuleResult:
        """
        Execute the module's authentication logic.

        Parameters are passed as keyword arguments so each module can declare
        only what it needs (frame=, audio_path=, video_path=, etc.).

        Must return a ModuleResult instance.
        """
        raise NotImplementedError

    def build_result(
        self,
        passed: bool,
        score: float,
        identity: str = "unknown",
        details: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> ModuleResult:
        """Helper to construct a ModuleResult with this module's name/weight already set."""
        return ModuleResult(
            module_name=self.module_name,
            passed=passed,
            score=score,
            weighted_score=score * self.effective_weight,
            identity=identity,
            details=details or {},
            error=error,
        )

    def emit_status(
        self,
        callback: Optional[Callable[..., None]],
        stage: str,
        message: str,
        **details: Any,
    ) -> None:
        """Emit a lightweight progress update if a callback is available."""
        if callable(callback):
            callback(stage=stage, message=message, details=details)

    @property
    def module_name(self) -> str:
        return self.__class__.__name__.lower().replace("module", "")
