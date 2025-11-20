from pydub import AudioSegment
import json
import os
import logging
import sys
from multiprocessing import Pool, Manager
from logging.handlers import RotatingFileHandler
import json_log_formatter
from pathlib import Path
import subprocess
import time
from tenacity import retry, stop_after_attempt, wait_fixed

# Set up structured JSON logging with rotation
formatter = json_log_formatter.JSONFormatter()
log_file = 'audio_muting.log'
handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)  # 10MB per file, 5 backups
handler.setFormatter(formatter)
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

logging.basicConfig(
    level=logging.INFO,
    handlers=[handler, console_handler]
)
logger = logging.getLogger(__name__)

# --- Constants ---
AUDIO_EXTENSIONS = [".wav", ".mp3", ".m4a"]
FORMAT_MAP = {'.wav': 'wav', '.mp3': 'mp3', '.m4a': 'mp4'}
PROCESSED_TRACK_FILE = "processed_files.txt"  # <- NEW

# --- Validate FFmpeg ---
def validate_ffmpeg(ffmpeg_path=None):
    try:
        cmd = [ffmpeg_path or 'ffmpeg', '-version']
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        ffmpeg_version = result.stdout.split('\n')[0]
        logger.info(f"FFmpeg validated successfully: {ffmpeg_version}")
        return ffmpeg_path or 'ffmpeg'
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg validation failed: {e.stderr}")
        return None
    except FileNotFoundError:
        logger.error("FFmpeg executable not found. Please provide the path to ffmpeg.")
        return None
    except Exception as e:
        logger.error(f"Unexpected error validating FFmpeg: {str(e)}")
        return None

# --- Processed path tracking --- (NEW)
def load_processed_paths(track_file):
    if not os.path.exists(track_file):
        return set()
    with open(track_file, "r", encoding="utf-8") as f:
        return set(line.strip() for line in f if line.strip())

def append_processed_path(track_file, path):
    with open(track_file, "a", encoding="utf-8") as f:
        f.write(f"{path}\n")

# --- Extract mute ranges ---
def extract_mute_ranges(data_raw):
    try:
        if isinstance(data_raw, dict):
            data = [data_raw]
        elif isinstance(data_raw, list):
            data = data_raw
        else:
            raise ValueError("Unexpected JSON structure: not a dict or list")
        
        mute_ranges = []
        seen = set()
        for item in data:
            entities = item.get("entities", [])
            if not isinstance(entities, list):
                logger.error(f"Invalid 'entities' field: expected list, got {type(entities)}")
                continue
            for entity in entities:
                start = entity.get("start")
                end = entity.get("end")
                entity_type = entity.get("type")
                entity_name = entity.get("entity")
                if (start is not None and end is not None and 
                    isinstance(start, (int, float)) and isinstance(end, (int, float)) and 
                    entity_type in ["PERSON", "EMAIL", "PHONE"] and end > start):
                    key = (entity_name, entity_type, start, end)
                    if key not in seen:
                        mute_ranges.append({"start": int(start), "end": int(end)})
                        seen.add(key)
        return mute_ranges
    except Exception as e:
        logger.error(f"Error in extract_mute_ranges: {str(e)}")
        return []

# --- Mute audio function ---
@retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
def mute_audio_segment(input_file, output_file, mute_ranges):
    try:
        audio = AudioSegment.from_file(input_file)
        if len(audio) == 0:
            logger.error(f"Empty audio file: {input_file}")
            return False
        
        result = AudioSegment.empty()
        last_end = 0
        
        mute_ranges = sorted(mute_ranges, key=lambda x: x['start'])
        
        for mute_range in mute_ranges:
            start_ms = max(0, mute_range['start'])
            end_ms = min(len(audio), mute_range['end'])
            if start_ms >= end_ms:
                logger.warning(f"Invalid mute range {start_ms}ms to {end_ms}ms in {input_file}")
                continue
            logger.info(f"Muting segment {start_ms}ms to {end_ms}ms in {input_file}")
            if last_end < start_ms:
                result += audio[last_end:start_ms]
            mute_duration = end_ms - start_ms
            result += AudioSegment.silent(duration=mute_duration)
            last_end = end_ms
        
        if last_end < len(audio):
            result += audio[last_end:]
        
        format_ext = os.path.splitext(output_file)[1].lower()
        export_format = FORMAT_MAP.get(format_ext, format_ext[1:])
        result.export(output_file, format=export_format)
        logger.info(f"Muted audio saved to: {output_file}")
        return True
    except Exception as e:
        logger.error(f"Error muting {input_file}: {str(e)}")
        raise

# --- Process a single file ---
def process_file(args):
    root, file, input_folder, metrics, processed_paths = args
    input_json_path = os.path.abspath(os.path.join(root, file))
    
    try:
        if not input_json_path.startswith(os.path.abspath(input_folder)):
            logger.error(f"Path traversal detected: {input_json_path}")
            return False
        
        base_name = os.path.splitext(file)[0][:-4] if file.endswith('_pii.json') else os.path.splitext(file)[0]
        
        @retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
        def load_json():
            with open(input_json_path, 'r', encoding='utf-8') as f:
                return json.load(f, object_pairs_hook=lambda x: dict(x[:1000]))
        
        try:
            data_raw = load_json()
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {input_json_path}: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Failed to load JSON file '{input_json_path}': {str(e)}")
            return False
        
        audio_input_path = None
        for ext in AUDIO_EXTENSIONS:
            potential_audio = os.path.join(root, base_name + ext)
            if os.path.exists(potential_audio):
                audio_input_path = potential_audio
                break
        if not audio_input_path:
            logger.warning(f"Skipping {input_json_path}: No matching audio file found in {root}")
            return False
        
        audio_ext = os.path.splitext(audio_input_path)[1]
        output_audio_path = os.path.join(root, f"{base_name}_pii_muted{audio_ext}")
        
        # NEW: Skip if already processed
        if output_audio_path in processed_paths or os.path.exists(output_audio_path):
            logger.info(f"Skipping {input_json_path}: Already processed {output_audio_path}")
            return False
        
        logger.info(f"Processing PII JSON: {input_json_path} with audio {audio_input_path}")
        
        mute_ranges = extract_mute_ranges(data_raw)
        
        if not mute_ranges:
            logger.info(f"No mute ranges found in {file}; copying original audio as output.")
            try:
                AudioSegment.from_file(audio_input_path).export(output_audio_path, format=audio_ext[1:])
                with metrics["lock"]:
                    metrics["success_count"] += 1
                    metrics["total_mute_duration"] += 0
                append_processed_path(PROCESSED_TRACK_FILE, output_audio_path)  # NEW
                return True
            except Exception as e:
                logger.error(f"Error copying audio {audio_input_path}: {str(e)}")
                return False
        
        success = mute_audio_segment(audio_input_path, output_audio_path, mute_ranges)
        if success:
            with metrics["lock"]:
                metrics["success_count"] += 1
                metrics["total_mute_duration"] += sum(mr["end"] - mr["start"] for mr in mute_ranges)
            append_processed_path(PROCESSED_TRACK_FILE, output_audio_path)  # NEW
        return success
    except Exception as e:
        logger.error(f"Unexpected error processing {input_json_path}: {str(e)}")
        return False

# --- Recursive processing function with multiprocessing ---
def process_pii_folder(input_folder: str, workers: int):
    try:
        if not os.path.exists(input_folder):
            raise FileNotFoundError(f"Input folder not found: {input_folder}")
        
        processed_paths = load_processed_paths(PROCESSED_TRACK_FILE)  # NEW
        
        file_list = []
        for root, _, files in os.walk(input_folder):
            for file in files:
                if file.endswith('_pii.json'):
                    file_list.append((root, file, input_folder, metrics, processed_paths))  # MODIFIED
        
        total_files = len(file_list)    
        if total_files == 0:
            logger.warning(f"No _pii.json files found in {input_folder}")
            return

        # Process files with multiprocessing
        with Pool(processes=workers) as pool:
            results = pool.map(process_file, file_list)

        # Log summary
        success_count = metrics["success_count"]
        total_mute_duration = metrics["total_mute_duration"] / 1000  # Convert to seconds
        logger.info({
            "message": "Processing complete",
            "success_count": success_count,
            "total_files": total_files,
            "success_rate": success_count / total_files if total_files > 0 else 0,
            "total_mute_duration_seconds": total_mute_duration
        })

        if success_count < total_files:
            logger.error(f"Processing failed for {total_files - success_count} files")
            sys.exit(1)
    except Exception as e:
        logger.error(f"Error in process_pii_folder: {str(e)}")
        sys.exit(1)

# --- Main entry point ---
if __name__ == "__main__":
    # Hardcoded constants
    DEFAULT_WORKERS = 4
    INPUT_DIR = "multi_lang_audio_output_for_muting"
    # Prompt for FFmpeg path if not found
    ffmpeg_path = validate_ffmpeg()
    if not ffmpeg_path:
        ffmpeg_path = input("Enter the path to ffmpeg executable (e.g., C:\\path\\to\\ffmpeg.exe): ").strip()
        if not validate_ffmpeg(ffmpeg_path):
            sys.exit(1)

    AudioSegment.converter = ffmpeg_path
    logger.info(f"FFmpeg path set to: {ffmpeg_path}")

    # Initialize shared metrics for multiprocessing
    manager = Manager()
    metrics = manager.dict()
    metrics["success_count"] = 0
    metrics["total_mute_duration"] = 0
    metrics["lock"] = manager.Lock()

    # Run processing
    process_pii_folder(INPUT_DIR, DEFAULT_WORKERS)