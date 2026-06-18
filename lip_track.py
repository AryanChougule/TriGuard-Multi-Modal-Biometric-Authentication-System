import os
import cv2
import argparse
import torch
import torchvision
import sys
# Do NOT shadow the real TensorFlow module by inserting `None` into
# `sys.modules` here — other modules (e.g. DeepFace) import TensorFlow
# at runtime and expect the real package to be importable. If TensorFlow
# is truly missing in the environment, those imports should fail in
# their own context instead of being prevented globally.

# Import directly from your clean local repo files
from lightning import ModelModule
from datamodule.transforms import VideoTransform

class LocalInferencePipeline(torch.nn.Module):
    def __init__(self, ckpt_path, device="cuda"):
        super(LocalInferencePipeline, self).__init__()
        
        # Build arguments expected by the underlying framework
        parser = argparse.ArgumentParser()
        self.args, _ = parser.parse_known_args(args=[])
        setattr(self.args, 'modality', 'video')
        setattr(self.args, 'device', device)
        
        self.device = device
        self.video_transform = VideoTransform(subset="test")
        
        # Choose a detector implementation: prefer MediaPipe 'solutions' API if available,
        # otherwise fall back to the RetinaFace-based detector provided in preparation/.
        try:
            import mediapipe as _mp_check
            if hasattr(_mp_check, "solutions"):
                print("[Lip] Initializing MediaPipe face and landmark trackers...")
                from preparation.detectors.mediapipe.detector import LandmarksDetector
                from preparation.detectors.mediapipe.video_process import VideoProcess
                self.detector_backend = "mediapipe"
            else:
                raise ImportError("mediapipe has no 'solutions' API; falling back to RetinaFace")
        except Exception:
            print("[Lip] MediaPipe 'solutions' not available; falling back to RetinaFace detector.")
            from preparation.detectors.retinaface.detector import LandmarksDetector
            from preparation.detectors.retinaface.video_process import VideoProcess
            self.detector_backend = "retinaface"

        self.landmarks_detector = LandmarksDetector()
        self.video_process = VideoProcess(convert_gray=False)
        
        print(f"[Lip] Loading model weights from: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=torch.device(device))
        
        # Initialize the model structure and push it completely to the GPU
        self.modelmodule = ModelModule(self.args)
        self.modelmodule.model.load_state_dict(ckpt)
        self.modelmodule.to(device)
        self.modelmodule.eval()
        
    def load_video(self, data_filename):
        # Reads the source video into memory arrays via torchvision
        video = torchvision.io.read_video(data_filename, pts_unit="sec")[0].numpy()
        # Preserve the decode order exactly as stored in the video container.
        return video.copy()
        
    def forward(self, data_filename, cancel_event=None):
        data_filename = os.path.abspath(data_filename)
        if not os.path.isfile(data_filename):
            raise FileNotFoundError(f"Target video file not found at: {data_filename}")

        if cancel_event and cancel_event.is_set():
            return ""
            
        # 1. Capture raw input video frames
        video = self.load_video(data_filename)
        if video is None or len(video) == 0:
            return ""
        
        # 2. Extract mouth/lip spatial landmark coordinates
        print("[Lip] Tracking face landmarks and isolating the lip region...")
        landmarks = self.landmarks_detector(video, cancel_event=cancel_event)
        if cancel_event and cancel_event.is_set():
            return ""
        
        # 3. Crop out peripheral regions and isolate the lips
        video = self.video_process(video, landmarks)
        if cancel_event and cancel_event.is_set():
            return ""
        if video is None or len(video) == 0:
            return ""
        
        # 4. Perform matrix permutations and push tensor processing to the GPU
        video = torch.tensor(video, dtype=torch.float32)
        video = video.permute((0, 3, 1, 2))  # Converts format shape to (T, C, H, W)
        video = self.video_transform(video)
        video = video.to(self.device)  # Explicitly force video frames onto CUDA
        if cancel_event and cancel_event.is_set():
            return ""
        
        # 5. Run GPU-accelerated encoder-decoder matrix calculations
        print("[Lip] Decoding visual lip movements into text...")
        with torch.no_grad():
            transcript = self.modelmodule(video)
            
        return transcript

def overlay_captions(input_video_path, output_video_path, caption_text):
    """Reads the original video and writes a new version with text burned in."""
    print("[Lip] Burning prediction text into video...")
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("[Lip] Error: frame buffering stream failed.")
        return

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        font_thickness = 2
        text_color = (255, 255, 255)
        bg_color = (0, 0, 0)
        
        text_size = cv2.getTextSize(caption_text, font, font_scale, font_thickness)[0]
        text_x = (width - text_size[0]) // 2
        text_y = height - 40

        box_coords = ((text_x - 10, text_y + 10), (text_x + text_size[0] + 10, text_y - text_size[1] - 10))
        cv2.rectangle(frame, box_coords[0], box_coords[1], bg_color, cv2.FILLED)
        cv2.putText(frame, caption_text, (text_x, text_y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
        
        out.write(frame)

    cap.release()
    out.release()
    print(f"[Lip] Captioned video created at: {output_video_path}")

if __name__ == "__main__":
    # --- FILE SELECTION PATHS ---
    # Put your test video file inside the auto_avsr folder and update its name here
    INPUT_VIDEO_FILE = "bbbf9a.mpg" 
    OUTPUT_CAPTIONED_FILE = "final_captioned_output_2.mp4"
    WEIGHTS_PATH = "weights/vsr_trlrs2lrs3vox2avsp_base.pth"
    # -----------------------------

    print("[Lip] Initializing GPU environment diagnostics...")
    
    # Force check for CUDA capability
    if not torch.cuda.is_available():
        print("\n[Lip] Critical error: CUDA GPU acceleration is not accessible.")
        print("Please verify that:")
        print("1. You have an NVIDIA GPU installed.")
        print("2. You installed the CUDA-compatible PyTorch version, not the CPU-only version.")
        print("Fix command: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118\n")
        exit(1)
        
    print(f"[Lip] CUDA GPU verified. Using: {torch.cuda.get_device_name(0)}")
    runtime_device = "cuda"

    if not os.path.exists(INPUT_VIDEO_FILE):
        print(f"[Lip] Error: input video file '{INPUT_VIDEO_FILE}' was not found in your directory.")
        print("Please place an MP4 video file in this folder and update INPUT_VIDEO_FILE in the script.")
    else:
        try:
            # 1. Build pipeline natively inside GPU context
            pipeline = LocalInferencePipeline(ckpt_path=WEIGHTS_PATH, device=runtime_device)
            
            # 2. Run inference calculations via GPU tensors
            predicted_statement = pipeline(INPUT_VIDEO_FILE)
            
            if not predicted_statement or predicted_statement.strip() == "":
                predicted_statement = "[No speech detected]"

            print("\n" + "="*40)
            print(f"PREDICTED TRANSCRIPT: {predicted_statement}")
            print("="*40 + "\n")

            # 3. Burn the text onto the video frames
            overlay_captions(INPUT_VIDEO_FILE, OUTPUT_CAPTIONED_FILE, predicted_statement.upper())

        except Exception as error_instance:
            print(f"[Lip] Execution failure during runtime: {error_instance}")
