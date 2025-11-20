import os
import sys
import json
import logging
import shutil
import subprocess
import tracemalloc
import gc
import psutil
import torch
from pathlib import Path
from faster_whisper import WhisperModel
from pydub import AudioSegment
from tqdm import tqdm
import argparse

# Global variables
LANGUAGE_CODE = "ta"  # Default language code (e.g., 'ta' for Tamil)
AUDIO_PATH = "input.wav"  # Default audio file path

# Setup logging
logging.basicConfig(
    filename="debug.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logging.info("Script started")

# Define cache directory
cache_dir = os.path.join(os.getcwd(), "model_cache")
os.makedirs(cache_dir, exist_ok=True)

# Verify ffmpeg
ffmpeg_path = "ffmpeg.exe"
try:
    subprocess.run([ffmpeg_path, "-version"], capture_output=True, check=True)
    print(f"✓ ffmpeg found at {ffmpeg_path}")
    AudioSegment.ffmpeg = ffmpeg_path
except Exception as e:
    logging.warning(f"ffmpeg not found at {ffmpeg_path}: {str(e)}. Trying system PATH.")
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        ffmpeg_path = "ffmpeg"
        print(f"✓ ffmpeg found in system PATH")
        AudioSegment.ffmpeg = ffmpeg_path
    except Exception as e:
        logging.error(f"ffmpeg not found: {str(e)}")
        print(f"❌ ffmpeg not found. Please ensure ffmpeg is installed and added to system PATH.")
        sys.exit(1)

# Clear existing cache
try:
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
        logging.info(f"Cleared existing cache directory {cache_dir}")
        print(f"✓ Cleared existing cache directory {cache_dir}")
    os.makedirs(cache_dir, exist_ok=True)
except Exception as e:
    logging.error(f"Failed to clear cache directory {cache_dir}: {str(e)}")
    print(f"❌ Failed to clear cache directory {cache_dir}: {str(e)}")
    sys.exit(1)

# Check cache directory permissions and disk space
try:
    test_file = os.path.join(cache_dir, "test.txt")
    with open(test_file, "w") as f:
        f.write("test")
    os.remove(test_file)
    disk_usage = shutil.disk_usage(os.getcwd())
    if disk_usage.free < 5e9:  # Less than 5GB free
        logging.error(f"Insufficient disk space: {disk_usage.free / 1e9:.2f} GB available")
        print(f"❌ Insufficient disk space: {disk_usage.free / 1e9:.2f} GB available")
        sys.exit(1)
    print(f"✓ Cache directory {cache_dir} is writable, {disk_usage.free / 1e9:.2f} GB free")
    logging.info(f"Cache directory {cache_dir} is writable, {disk_usage.free / 1e9:.2f} GB free")
except Exception as e:
    logging.error(f"Cache directory {cache_dir} issue: {str(e)}")
    print(f"❌ Cache directory {cache_dir} issue: {str(e)}")
    sys.exit(1)

# Check GPU availability and VRAM
model_size = "large-v3"  # For better Tamil accuracy
USE_GPU = True  # Prioritize GPU
device = "cuda" if USE_GPU and torch.cuda.is_available() else "cpu"
compute_type = "float32" if device == "cuda" else "int8"  # Use float32 for GPU, int8 for CPU

if device == "cuda":
    try:
        torch.cuda.init()
        gpu_info = torch.cuda.get_device_properties(0)
        print(f"✓ GPU: {gpu_info.name}, Total VRAM: {gpu_info.total_memory / 1e9:.2f} GB")
        logging.info(f"GPU: {gpu_info.name}, Total VRAM: {gpu_info.total_memory / 1e9:.2f} GB")
    except Exception as e:
        logging.error(f"Failed to initialize CUDA: {str(e)}. Falling back to CPU.")
        print(f"❌ Failed to initialize CUDA: {str(e)}. Falling back to CPU.")
        device = "cpu"
        compute_type = "int8"

# Clear memory before loading
gc.collect()
if device == "cuda":
    torch.cuda.empty_cache()
    print(f"✓ Cleared GPU memory before loading model")
print(f"Memory used before loading: {psutil.virtual_memory().used / 1e9:.2f} GB")

# Load model
model = None
try:
    logging.info(f"Downloading and caching {model_size} model to {cache_dir} with compute_type={compute_type} on {device}")
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    print(f"✓ Successfully cached {model_size} model to {cache_dir} on {device}")
    logging.info(f"Model {model_size} cached successfully on {device}")
except Exception as e:
    logging.error(f"Failed to cache {model_size} model on {device}: {str(e)}", exc_info=True)
    print(f"❌ Failed to cache {model_size} model on {device}: {str(e)}")
    if device == "cuda":
        logging.warning("Retrying with CPU and int8")
        device = "cpu"
        compute_type = "int8"
        try:
            model = WhisperModel(model_size, device=device, compute_type=compute_type)
            print(f"✓ Successfully cached {model_size} model on CPU with {compute_type}")
            logging.info(f"Model {model_size} cached successfully on CPU")
        except Exception as e:
            logging.error(f"Failed to cache {model_size} model on CPU: {str(e)}", exc_info=True)
            print(f"❌ Failed to cache {model_size} model on CPU: {str(e)}")
            sys.exit(1)
    else:
        sys.exit(1)

# Verify model is loaded
if model is None:
    logging.error("Model is None after caching attempt")
    print("❌ Model is None after caching attempt")
    sys.exit(1)

# Validate model functionality
try:
    if not hasattr(model, "transcribe"):
        raise AttributeError("Model object lacks 'transcribe' method")
    print(f"✓ Model {model_size} is valid and ready for transcription on {device}")
    logging.info(f"Model {model_size} is valid and ready for transcription on {device}")
except Exception as e:
    logging.error(f"Model validation failed: {str(e)}", exc_info=True)
    print(f"❌ Model validation failed: {str(e)}")
    sys.exit(1)

# Load from cache
try:
    logging.info(f"Loading {model_size} model with {compute_type} from cache on {device}")
    model = WhisperModel(model_size, device=device, compute_type=compute_type, local_files_only=True)
    print(f"✓ {model_size} loaded on {device} with {compute_type} from cache")
except Exception as e:
    logging.error(f"Failed to load {model_size} model from cache on {device}: {str(e)}", exc_info=True)
    print(f"❌ Failed to load {model_size} model from cache on {device}: {str(e)}")
    sys.exit(1)

# Final model check
if model is None or not hasattr(model, "transcribe"):
    logging.error("Model loading failed: model is None or lacks transcribe method")
    print("❌ Model loading failed: model is None or lacks transcribe method")
    sys.exit(1)

# --- Main Functions ---
def split_audio(audio_path=AUDIO_PATH, chunk_length_ms=20000):
    logging.info(f"Splitting {audio_path} with chunk_length_ms={chunk_length_ms}")
    chunks = []
    try:
        tracemalloc.start()
        temp_audio_path = os.path.join(os.getcwd(), "temp_input.wav")
        ffmpeg_command = [
            ffmpeg_path, "-i", audio_path, "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
            "-y", temp_audio_path
        ]
        result = subprocess.run(ffmpeg_command, capture_output=True, text=True)
        if result.returncode != 0:
            logging.error(f"ffmpeg failed for {audio_path}: {result.stderr}")
            print(f"❌ ffmpeg failed for {audio_path}: {result.stderr}")
            return chunks
        
        logging.info(f"Converted {audio_path} to {temp_audio_path}")
        audio = AudioSegment.from_file(temp_audio_path)
        duration_ms = len(audio)
        if duration_ms % chunk_length_ms != 0:
            padding_ms = chunk_length_ms - (duration_ms % chunk_length_ms)
            audio += AudioSegment.silent(duration=padding_ms)
            logging.info(f"Padded audio with {padding_ms}ms silence to align with {chunk_length_ms}ms chunks")
        
        temp_dir = Path(os.path.join(os.getcwd(), "temp_chunks"))
        temp_dir.mkdir(exist_ok=True)
        for i in range(0, len(audio), chunk_length_ms):
            chunk = audio[i:i + chunk_length_ms]
            chunk_path = temp_dir / f"{Path(audio_path).stem}_chunk{i//1000}.wav"
            try:
                chunk.export(chunk_path, format="wav")
                chunks.append((str(chunk_path), i / 1000.0))
                logging.info(f"Created chunk: {chunk_path}, offset: {i / 1000.0}s")
            except Exception as e:
                logging.error(f"Failed to export chunk {chunk_path}: {str(e)}")
        
        if Path(temp_audio_path).exists():
            os.remove(temp_audio_path)
            logging.info(f"Removed temporary file: {temp_audio_path}")
        
        tracemalloc.stop()
        return chunks
    except Exception as e:
        logging.error(f"Split audio error for {audio_path}: {str(e)}", exc_info=True)
        print(f"❌ Split audio error for {audio_path}: {str(e)}")
        return chunks

def transcribe_audio(chunk_path, model, device, language=LANGUAGE_CODE, beam_size=1, min_silence=1000):
    logging.info(f"Transcribing {chunk_path} with language={language}")
    if not Path(chunk_path).is_file():
        logging.error(f"Chunk file {chunk_path} not found")
        return {"error": f"Chunk file {chunk_path} not found"}
    try:
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
        segments, info = model.transcribe(
            chunk_path,
            language=language,
            beam_size=beam_size,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=min_silence),
            word_timestamps=True
        )
        transcript_segments = []
        full_text = ""
        word_timestamps = []
        for segment in segments:
            transcript_segments.append({
                "start": int(segment.start * 1000),
                "end": int(segment.end * 1000),
                "text": segment.text.strip()
            })
            full_text += segment.text.strip() + " "
            for word in segment.words:
                word_timestamps.append({
                    "word": word.word.strip(),
                    "start": int(word.start * 1000),
                    "end": int(word.end * 1000)
                })
        return {
            "segments": transcript_segments,
            "full_text": full_text.strip(),
            "word_timestamps": word_timestamps,
            "language": info.language,
            "confidence": info.language_probability
        }
    except Exception as e:
        logging.error(f"Transcription error for {chunk_path}: {str(e)}", exc_info=True)
        print(f"❌ Transcription error for {chunk_path}: {str(e)}")
        return {"error": str(e)}

# Command-line argument parsing
def parse_arguments():
    parser = argparse.ArgumentParser(description="Transcribe audio using faster-whisper.")
    parser.add_argument(
        "--audio_path",
        type=str,
        default=AUDIO_PATH,
        help="Path to the input audio file (default: input.wav)"
    )
    parser.add_argument(
        "--language",
        type=str,
        default=LANGUAGE_CODE,
        help="Language code for transcription (e.g., 'ta' for Tamil, 'en' for English) (default: ta)"
    )
    return parser.parse_args()

# Main execution
if __name__ == "__main__":
    args = parse_arguments()
    # Override global variables with command-line arguments if provided
    AUDIO_PATH = args.audio_path
    LANGUAGE_CODE = args.language

    # Verify audio file exists
    if not os.path.isfile(AUDIO_PATH):
        logging.error(f"Audio file {AUDIO_PATH} not found")
        print(f"❌ Audio file {AUDIO_PATH} not found")
        sys.exit(1)

    # Split audio into chunks
    chunks = split_audio(AUDIO_PATH)
    if not chunks:
        logging.error("No chunks created. Exiting.")
        print("❌ No chunks created. Exiting.")
        sys.exit(1)

    # Transcribe each chunk
    results = []
    for chunk_path, offset in tqdm(chunks, desc="Transcribing chunks"):
        result = transcribe_audio(chunk_path, model, device, LANGUAGE_CODE)
        if "error" not in result:
            result["offset"] = offset
            results.append(result)
        else:
            logging.error(f"Failed to transcribe chunk {chunk_path}: {result['error']}")
            print(f"❌ Failed to transcribe chunk {chunk_path}: {result['error']}")

    # Save results to JSON
    output_file = "transcription_results.json"
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"✓ Transcription results saved to {output_file}")
        logging.info(f"Transcription results saved to {output_file}")
    except Exception as e:
        logging.error(f"Failed to save results to {output_file}: {str(e)}")
        print(f"❌ Failed to save results to {output_file}: {str(e)}")
        sys.exit(1)

    # Clean up temporary chunk files
    temp_dir = Path(os.path.join(os.getcwd(), "temp_chunks"))
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
        logging.info(f"Removed temporary chunk directory: {temp_dir}")
        print(f"✓ Removed temporary chunk directory: {temp_dir}")