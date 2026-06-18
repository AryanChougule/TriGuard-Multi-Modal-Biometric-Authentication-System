import logging
import sys
import threading
from datetime import datetime
from typing import Any, Dict, Optional

import cv2
try:
    from modules.face_module import preload_face_runtime

    preload_face_runtime()
except Exception:
    # Face module will report a clearer runtime error if its dependencies still fail.
    pass

from PyQt5.QtCore import QObject, Qt, QThread, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QFont, QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QVBoxLayout,
    QWidget,
)

from core.capture_session import CaptureSession
from core.config_loader import load_config
from core.logger import setup_logger
from core.orchestrator import AuthOrchestrator
from main import build_modules


class CameraPreviewWorker(QObject):
    frame_ready = pyqtSignal(object)
    status_changed = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, camera_index: int = 0):
        super().__init__()
        self.camera_index = camera_index
        self._stop_event = threading.Event()

    @pyqtSlot()
    def run(self) -> None:
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            self.status_changed.emit("Preview camera unavailable.")
            self.finished.emit()
            return

        self.status_changed.emit("Live preview ready.")
        try:
            while not self._stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    continue
                self.frame_ready.emit(frame.copy())
                QThread.msleep(33)
        finally:
            cap.release()
            self.finished.emit()

    def stop(self) -> None:
        self._stop_event.set()


class AuthWorker(QObject):
    preview_ready = pyqtSignal(object)
    system_status = pyqtSignal(str, str)
    capture_status = pyqtSignal(str, str, object)
    module_status = pyqtSignal(str, str, str, object)
    result_ready = pyqtSignal(object)
    failed = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, config_path: str = "config.json", preloaded_config=None, preloaded_modules=None):
        super().__init__()
        self.config_path = config_path
        self._stop_event = threading.Event()
        self._preloaded_config = preloaded_config
        self._preloaded_modules = preloaded_modules

    def stop(self) -> None:
        self._stop_event.set()
        self.system_status.emit("Stopping", "Stop requested.")

    @pyqtSlot()
    def run(self) -> None:
        session = None
        capture = None

        try:
            config = self._preloaded_config
            modules = self._preloaded_modules

            if config is None or modules is None:
                self.system_status.emit("Loading", "Loading configuration.")
                config = load_config(self.config_path)

                sys_cfg = config["system"]
                setup_logger(
                    log_dir=sys_cfg.get("log_dir", "logs/"),
                    log_level=sys_cfg.get("log_level", "INFO"),
                )

                self.system_status.emit("Loading", "Initializing authentication modules.")
                modules = build_modules(config, exit_on_error=False)
                for module_name in modules.keys():
                    self.module_status.emit(module_name, "Ready", "Module loaded.", {})
            else:
                self.system_status.emit("Ready", "Preloaded models and configuration are ready.")

            session = CaptureSession(
                system_cfg=config["system"],
                lip_cfg=config["modules"]["lip"],
                face_cfg=config["modules"]["face"],
                status_callback=self._handle_capture_status,
                preview_callback=self._handle_preview_frame,
                stop_event=self._stop_event,
            )

            self.system_status.emit("Capture", "Waiting for countdown and recording.")
            capture = session.run()

            if capture.cancelled:
                self.result_ready.emit(
                    {
                        "authenticated": False,
                        "identity": "cancelled",
                        "final_score": 0.0,
                        "summary": capture.error or "Capture cancelled by user.",
                        "results": {},
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "cancelled": True,
                    }
                )
                return

            if capture.error:
                raise RuntimeError(capture.error)

            if capture.face_abort:
                self.result_ready.emit(
                    {
                        "authenticated": False,
                        "identity": "unknown",
                        "final_score": 0.0,
                        "results": {},
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "summary": "Authentication aborted: multiple faces detected.",
                    }
                )
                return

            inputs: Dict[str, Any] = {
                "frames": capture.frames,
                "face_abort": capture.face_abort,
                "audio_path": capture.audio_path,
                "video_path": capture.video_path,
                "chosen_sentence": capture.chosen_sentence,
            }

            self.system_status.emit("Processing", "Running face, voice, and lip modules.")
            orchestrator = AuthOrchestrator(config, modules)
            result = orchestrator.authenticate(inputs, status_callback=self._handle_module_status)
            self.result_ready.emit(result)
        except BaseException as exc:
            logging.getLogger(__name__).exception("GUI authentication failed: %s", exc)
            self.failed.emit(str(exc))
        finally:
            if session and capture:
                session.cleanup(capture)
            self.finished.emit()

    def _handle_preview_frame(self, frame) -> None:
        self.preview_ready.emit(frame)

    def _handle_capture_status(self, stage: str, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        self.capture_status.emit(stage, message, details or {})

    def _handle_module_status(
        self,
        module_name: str,
        stage: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.module_status.emit(module_name, stage, message, details or {})


class StartupWorker(QObject):
    status = pyqtSignal(str, str)
    ready = pyqtSignal(object, object)
    failed = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, config_path: str = "config.json"):
        super().__init__()
        self.config_path = config_path

    @pyqtSlot()
    def run(self) -> None:
        try:
            self.status.emit("Loading", "Loading configuration.")
            config = load_config(self.config_path)

            sys_cfg = config["system"]
            setup_logger(
                log_dir=sys_cfg.get("log_dir", "logs/"),
                log_level=sys_cfg.get("log_level", "INFO"),
            )

            self.status.emit("Loading", "Preloading face, voice, and lip models.")
            modules = build_modules(config, exit_on_error=False)
            self.ready.emit(config, modules)
        except BaseException as exc:
            logging.getLogger(__name__).exception("Startup preload failed: %s", exc)
            self.failed.emit(str(exc))
        finally:
            self.finished.emit()


class ModuleCard(QFrame):
    def __init__(self, title: str):
        super().__init__()
        self.setObjectName("moduleCard")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(8)

        self.title_label = QLabel(title)
        self.title_label.setObjectName("cardTitle")
        self.status_label = QLabel("Idle")
        self.status_label.setObjectName("cardStatus")
        self.message_label = QLabel("Waiting to start.")
        self.message_label.setWordWrap(True)
        self.message_label.setObjectName("cardMessage")

        layout.addWidget(self.title_label)
        layout.addWidget(self.status_label)
        layout.addWidget(self.message_label)

    def update_status(self, status: str, message: str) -> None:
        self.status_label.setText(status)
        self.message_label.setText(message)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TriGuard Auth Console")
        self.resize(1400, 900)

        self.preview_thread: Optional[QThread] = None
        self.preview_worker: Optional[CameraPreviewWorker] = None
        self.auth_thread: Optional[QThread] = None
        self.auth_worker: Optional[AuthWorker] = None
        self.current_capture_stage = "idle"
        self.startup_thread: Optional[QThread] = None
        self.startup_worker: Optional[StartupWorker] = None
        self.preloaded_config: Optional[Dict[str, Any]] = None
        self.preloaded_modules: Optional[Dict[str, Any]] = None
        self.models_ready = False

        self._build_ui()
        self._apply_styles()
        self._set_ready_state(False)
        self._start_startup_preload()
        self._start_idle_preview()

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)

        main_layout = QVBoxLayout(root)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(18)

        header_row = QHBoxLayout()
        header_row.setSpacing(16)
        title = QLabel("TriGuard Multi-Modal Authentication")
        title.setObjectName("appTitle")
        subtitle = QLabel("Live capture, parallel module tracking, and final auth verdict")
        subtitle.setObjectName("appSubtitle")

        title_col = QVBoxLayout()
        title_col.addWidget(title)
        title_col.addWidget(subtitle)
        header_row.addLayout(title_col)
        header_row.addStretch(1)

        self.start_button = QPushButton("Start Recording")
        self.stop_button = QPushButton("Stop Recording")
        self.stop_button.setEnabled(False)
        self.clear_button = QPushButton("Clear Events")

        self.start_button.clicked.connect(self.start_auth)
        self.stop_button.clicked.connect(self.stop_auth)

        header_row.addWidget(self.start_button)
        header_row.addWidget(self.stop_button)
        header_row.addWidget(self.clear_button)
        main_layout.addLayout(header_row)

        body = QGridLayout()
        body.setHorizontalSpacing(18)
        body.setVerticalSpacing(18)
        main_layout.addLayout(body, 1)

        self.camera_label = QLabel("Camera preview starting...")
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setMinimumSize(720, 420)
        self.camera_label.setObjectName("cameraFrame")
        body.addWidget(self.camera_label, 0, 0, 2, 2)

        system_group = QGroupBox("Session Status")
        system_layout = QVBoxLayout(system_group)
        system_layout.setSpacing(10)

        self.system_status_label = QLabel("Idle")
        self.capture_status_label = QLabel("Preview mode.")
        self.challenge_prompt_label = QLabel("Challenge sentence")
        self.challenge_prompt_label.setObjectName("challengePromptLabel")
        self.challenge_label = QLabel("not started")
        self.challenge_label.setObjectName("challengeLabel")
        self.result_identity_label = QLabel("Identity: -")
        self.result_score_label = QLabel("Final score: -")
        self.result_verdict_label = QLabel("Verdict: -")

        for widget in [
            self.system_status_label,
            self.capture_status_label,
            self.challenge_prompt_label,
            self.challenge_label,
            self.result_identity_label,
            self.result_score_label,
            self.result_verdict_label,
        ]:
            widget.setWordWrap(True)
            system_layout.addWidget(widget)

        body.addWidget(system_group, 0, 2)

        modules_group = QGroupBox("Module Activity")
        modules_layout = QVBoxLayout(modules_group)
        modules_layout.setSpacing(12)
        self.module_cards = {
            "face": ModuleCard("Face Module"),
            "voice": ModuleCard("Voice Module"),
            "lip": ModuleCard("Lip Module"),
        }
        for card in self.module_cards.values():
            modules_layout.addWidget(card)
        body.addWidget(modules_group, 1, 2)

        self.event_log = QPlainTextEdit()
        self.event_log.setReadOnly(True)
        self.event_log.setObjectName("eventLog")
        self.clear_button.clicked.connect(self.event_log.clear)
        body.addWidget(self.event_log, 2, 0, 1, 3)

    def _apply_styles(self) -> None:
        self.setFont(QFont("Bahnschrift", 10))
        self.setStyleSheet(
            """
            QMainWindow, QWidget {
                background: #f3efe6;
                color: #1b1e1f;
            }
            #appTitle {
                font-size: 28px;
                font-weight: 700;
                color: #13343b;
            }
            #appSubtitle {
                color: #5a6668;
                font-size: 13px;
            }
            QPushButton {
                background: #13343b;
                color: white;
                border: none;
                border-radius: 10px;
                padding: 10px 18px;
                font-weight: 600;
            }
            QPushButton:disabled {
                background: #93a1a1;
                color: #e6ecec;
            }
            QGroupBox {
                border: 1px solid #d5d0c5;
                border-radius: 14px;
                margin-top: 12px;
                padding-top: 12px;
                font-weight: 700;
                background: #fcfaf5;
            }
            QGroupBox::title {
                left: 12px;
                padding: 0 6px;
                color: #13343b;
            }
            #cameraFrame {
                background: #172225;
                border: 1px solid #294146;
                border-radius: 16px;
                color: #d8ece9;
            }
            #moduleCard {
                background: #f7f3ea;
                border: 1px solid #ddd7ca;
                border-radius: 12px;
            }
            #cardTitle {
                font-size: 16px;
                font-weight: 700;
                color: #13343b;
            }
            #cardStatus {
                font-size: 13px;
                font-weight: 600;
                color: #8e5b10;
            }
            #cardMessage {
                color: #4c5557;
            }
            #challengeLabel {
                font-size: 24px;
                font-weight: 800;
                color: #0f3941;
                padding: 14px 16px;
                border: 2px solid #c9d7d8;
                border-radius: 14px;
                background: #eef6f4;
            }
            #challengePromptLabel {
                font-size: 12px;
                font-weight: 800;
                letter-spacing: 1px;
                text-transform: uppercase;
                color: #5b6d70;
                margin-top: 4px;
            }
            #eventLog {
                background: #fffdf7;
                border: 1px solid #d8d0bf;
                border-radius: 14px;
                padding: 8px;
            }
            """
        )

    def append_event(self, text: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.event_log.appendPlainText(f"[{timestamp}] {text}")

    def start_auth(self) -> None:
        if not self.models_ready:
            self.append_event("Models are still loading. Please wait a moment.")
            return

        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self._stop_idle_preview()
        self._reset_module_cards()
        self.append_event("Authentication session requested.")

        self.auth_thread = QThread()
        self.auth_worker = AuthWorker(
            preloaded_config=self.preloaded_config,
            preloaded_modules=self.preloaded_modules,
        )
        self.auth_worker.moveToThread(self.auth_thread)

        self.auth_thread.started.connect(self.auth_worker.run)
        self.auth_worker.preview_ready.connect(self.update_preview)
        self.auth_worker.system_status.connect(self.on_system_status)
        self.auth_worker.capture_status.connect(self.on_capture_status)
        self.auth_worker.module_status.connect(self.on_module_status)
        self.auth_worker.result_ready.connect(self.on_result_ready)
        self.auth_worker.failed.connect(self.on_auth_failed)
        self.auth_worker.finished.connect(self.on_auth_finished)
        self.auth_worker.finished.connect(self.auth_thread.quit)
        self.auth_worker.finished.connect(self.auth_worker.deleteLater)
        self.auth_thread.finished.connect(self.auth_thread.deleteLater)
        self.auth_thread.finished.connect(self._finalize_auth_thread)

        self.auth_thread.start()

    def stop_auth(self) -> None:
        if self.auth_worker:
            self.auth_worker.stop()
            self.append_event("Stop requested by user.")
            self.stop_button.setEnabled(False)

    def on_system_status(self, status: str, message: str) -> None:
        self.system_status_label.setText(f"System: {status}")
        self.capture_status_label.setText(f"Activity: {message}")
        self.append_event(f"System {status.lower()}: {message}")

    def on_capture_status(self, stage: str, message: str, details: object) -> None:
        details = details or {}
        self.current_capture_stage = stage
        self.capture_status_label.setText(f"Activity: {message}")
        if isinstance(details, dict) and details.get("chosen_sentence"):
            self.challenge_label.setText(details["chosen_sentence"])
        self.append_event(f"Capture {stage}: {message}")

    def on_module_status(self, module_name: str, stage: str, message: str, details: object) -> None:
        card = self.module_cards.get(module_name)
        if card:
            card.update_status(stage.title(), message)
        self.append_event(f"{module_name.upper()} {stage}: {message}")

    def on_result_ready(self, result: object) -> None:
        result = result or {}
        self.result_identity_label.setText(f"Identity: {result.get('identity', '-')}")
        self.result_score_label.setText(f"Final score: {result.get('final_score', 0.0):.4f}")

        if result.get("cancelled"):
            verdict = "Cancelled"
        else:
            verdict = "Authenticated" if result.get("authenticated") else "Access Denied"
        self.result_verdict_label.setText(f"Verdict: {verdict}")

        summary = result.get("summary", "")
        if summary:
            self.append_event(summary.replace("\n", " | "))

    def on_auth_failed(self, error: str) -> None:
        self.append_event(f"Authentication failed: {error}")
        QMessageBox.critical(self, "Authentication Error", error)

    def on_auth_finished(self) -> None:
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.auth_worker = None
        self._start_idle_preview()

    def _finalize_auth_thread(self) -> None:
        self.auth_thread = None

    def _reset_module_cards(self) -> None:
        for card in self.module_cards.values():
            card.update_status("Idle", "Waiting to start.")

    def _set_ready_state(self, ready: bool) -> None:
        self.models_ready = ready
        self.start_button.setEnabled(ready)
        if ready:
            self.system_status_label.setText("System: Ready")
            self.capture_status_label.setText("Activity: Models loaded. Ready to record.")
        else:
            self.system_status_label.setText("System: Loading")
            self.capture_status_label.setText("Activity: Loading models and configuration.")

    def _start_startup_preload(self) -> None:
        self.startup_thread = QThread()
        self.startup_worker = StartupWorker()
        self.startup_worker.moveToThread(self.startup_thread)

        self.startup_thread.started.connect(self.startup_worker.run)
        self.startup_worker.status.connect(self.on_startup_status)
        self.startup_worker.ready.connect(self.on_startup_ready)
        self.startup_worker.failed.connect(self.on_startup_failed)
        self.startup_worker.finished.connect(self.startup_thread.quit)
        self.startup_worker.finished.connect(self.startup_worker.deleteLater)
        self.startup_thread.finished.connect(self.startup_thread.deleteLater)
        self.startup_thread.start()

    def on_startup_status(self, stage: str, message: str) -> None:
        self.system_status_label.setText(f"System: {stage}")
        self.capture_status_label.setText(f"Activity: {message}")
        self.append_event(f"Startup {stage.lower()}: {message}")

    def on_startup_ready(self, config: Dict[str, Any], modules: Dict[str, Any]) -> None:
        self.preloaded_config = config
        self.preloaded_modules = modules
        for module_name in modules.keys():
            self.module_cards[module_name].update_status("Ready", "Module loaded at startup.")
        self._set_ready_state(True)
        self.append_event("Models and modules loaded. Ready to record.")

    def on_startup_failed(self, error: str) -> None:
        self.append_event(f"Startup failed: {error}")
        QMessageBox.critical(self, "Startup Error", error)
        self._set_ready_state(False)

    def _start_idle_preview(self) -> None:
        try:
            config = self.preloaded_config or load_config("config.json")
            camera_index = config["system"].get("camera_index", 0)
        except Exception:
            camera_index = 0

        self.preview_thread = QThread()
        self.preview_worker = CameraPreviewWorker(camera_index=camera_index)
        self.preview_worker.moveToThread(self.preview_thread)

        self.preview_thread.started.connect(self.preview_worker.run)
        self.preview_worker.frame_ready.connect(self.update_preview)
        self.preview_worker.status_changed.connect(
            lambda message: self.append_event(f"Preview: {message}")
        )
        self.preview_worker.finished.connect(self.preview_thread.quit)
        self.preview_worker.finished.connect(self.preview_worker.deleteLater)
        self.preview_thread.finished.connect(self.preview_thread.deleteLater)
        self.preview_thread.start()

    def _stop_idle_preview(self) -> None:
        if self.preview_worker:
            self.preview_worker.stop()
        if self.preview_thread:
            self.preview_thread.quit()
            self.preview_thread.wait(1500)
        self.preview_worker = None
        self.preview_thread = None

    @pyqtSlot(object)
    def update_preview(self, frame: object) -> None:
        if frame is None:
            return

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channel = rgb.shape
        bytes_per_line = channel * width
        image = QImage(rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(image).scaled(
            self.camera_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.camera_label.setPixmap(pixmap)

    def closeEvent(self, event) -> None:
        if self.auth_worker:
            self.auth_worker.stop()
        self._stop_idle_preview()
        super().closeEvent(event)


def main() -> int:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
