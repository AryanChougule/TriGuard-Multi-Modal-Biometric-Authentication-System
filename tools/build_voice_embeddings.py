# build_database.py

import os
import pickle
import subprocess
import tempfile
import torch
import torchaudio
from speechbrain.inference.speaker import SpeakerRecognition

# =====================================================
# CONFIGURATION
# =====================================================
MODEL_DIR = "./pretrained_models/spkrec-ecapa-voxceleb"
TRAIN_DIR = "data/voice_dataset/main_dataset"
OUTPUT_PATH = "data/voice_dataset/voice_embedding_db.pkl"

# The working path to your installed FFmpeg binary
FFMPEG_PATH = r"C:\Users\aryan\AppData\Local\ffmpegio\ffmpeg-downloader\ffmpeg\bin\ffmpeg.exe"

# Supported audio and video extensions
VALID_EXTENSIONS = (".wav", ".mp3", ".flac", ".ogg", ".mp4", ".mov", ".avi", ".mkv")

# Extensions that require FFmpeg processing beforehand
CONVERT_EXTENSIONS = (".mp4", ".ogg", ".mov", ".avi", ".mkv")

# =====================================================
# DEVICE & MODEL SETUP
# =====================================================
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using Device: {device}\n")

print("Loading ECAPA-TDNN Speaker Recognition Model...")
verification_model = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir=MODEL_DIR,
    run_opts={"device": device}
)
print("Model Loaded Successfully.\n")

# =====================================================
# AUDIO EXTRACTION UTILITY
# =====================================================
def convert_to_wav(media_path):
    """
    Converts MP4/OGG/etc. to a temporary 16kHz mono WAV file 
    using the local FFmpeg binary.
    """
    temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    temp_wav.close()
    output_path = temp_wav.name

    command = [
        FFMPEG_PATH,
        "-y",
        "-i", media_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        output_path
    ]

    result = subprocess.run(
        command, 
        stdout=subprocess.DEVNULL, 
        stderr=subprocess.DEVNULL
    )
    
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg processing failed for file: {media_path}")
        
    return output_path

# =====================================================
# EMBEDDING EXTRACTION
# =====================================================
def extract_embedding(audio_path):
    temporary_file = None

    try:
        # If it's a video file or .ogg, convert to raw WAV first
        if audio_path.lower().endswith(CONVERT_EXTENSIONS):
            temporary_file = convert_to_wav(audio_path)
            audio_path = temporary_file

        signal, fs = torchaudio.load(audio_path)

        # Convert stereo to mono
        if signal.shape[0] > 1:
            signal = signal.mean(dim=0, keepdim=True)

        # Resample to 16kHz if necessary
        if fs != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=fs, new_freq=16000)
            signal = resampler(signal)

        with torch.no_grad():
            embedding = verification_model.encode_batch(signal)

        # Flatten array to cleanly store it in the pickle file
        return embedding.cpu().numpy().flatten()

    finally:
        # Clean up temporary files safely
        if temporary_file is not None and os.path.exists(temporary_file):
            os.remove(temporary_file)

# =====================================================
# MAIN PROCESSING LOOP
# =====================================================
def main():
    speaker_database = {}

    if not os.path.exists(TRAIN_DIR):
        print(f"ERROR: Training directory '{TRAIN_DIR}' does not exist.")
        return

    print("====================================")
    print("STARTING DATABASE BUILDING PROCESS")
    print("====================================")

    for speaker_name in os.listdir(TRAIN_DIR):
        speaker_folder = os.path.join(TRAIN_DIR, speaker_name)

        if not os.path.isdir(speaker_folder):
            continue

        embeddings = []
        print(f"\nProcessing Speaker Folder: [{speaker_name}]")

        for file in os.listdir(speaker_folder):
            if file.lower().endswith(VALID_EXTENSIONS):
                path = os.path.join(speaker_folder, file)
                
                try:
                    emb = extract_embedding(path)
                    embeddings.append(emb)
                    print(f"  -> Processed: {file}")
                except Exception as e:
                    print(f"  -> ❌ Failed to process {file}. Error: {e}")

        # Guard rail: Only add speaker if they actually have processed embeddings
        if len(embeddings) > 0:
            speaker_database[speaker_name] = embeddings
            print(f"Successfully enrolled '{speaker_name}' with {len(embeddings)} samples.")
        else:
            print(f"⚠️ Warning: No valid data extracted for '{speaker_name}'. Skipping registration.")

    # Save data
    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(speaker_database, f)

    print("\n====================================")
    print(f"Success! Saved {len(speaker_database)} speakers to '{OUTPUT_PATH}'")
    print("====================================")

if __name__ == "__main__":
    main()
