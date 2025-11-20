import os, json, time, requests, subprocess, shutil, argparse
from dotenv import load_dotenv
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

load_dotenv()

API_BASE_URL = os.getenv("SONIOX_API_BASE_URL")
LANGUAGE_HINTS = ["en"]

# Used to store script name for the start and end time analysis in script_log.txt, It will help to figure out the start and end time if run the multiple scripts.
SCRIPT_NAME = "main"

TRANSACRIPTION_PROCESSSING_SLEEP_IN_SEC = 5
SONIOX_TRANSCRIPTION_MODEL = os.getenv("SONIOX_TRANSCRIPTION_MODEL")
# Used to provide the input folder path in which list of audios are placed.
INPUT_FOLDER_PATH = os.getenv("TRANSCRIPTION_INPUT_FOLDER_PATH")
# Used to dump the transcription folder which contain audio files and relevent transcription file
OUTPUT_FOLDER_PATH = os.getenv("TRANSCRIPTION_OUTPUT_FOLDER_PATH")
# Logs
LOG_FILE = os.getenv("TRANSCRIPTION_LOG_FILE")
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
PROCESSED_TRACK_FILE = os.getenv("TRANSCRIPTION_PROCESSED_TRACK_FILE")
os.makedirs(os.path.dirname(PROCESSED_TRACK_FILE), exist_ok=True)

MAX_CONCURRENT = os.cpu_count() * 4
MAX_TOTAL_GB = 9.5
MAX_TOTAL_BYTES = int(MAX_TOTAL_GB * 1024 ** 3)

MAX_THREAD_POOL = os.cpu_count() * 4

SESSION = requests.Session()

# ---------------------------
# Pool to manage concurrency + size limit
# ---------------------------
class ProcessingPool:
    """Manage concurrent requests and total size limit in bytes."""
    def __init__(self, max_concurrent: int, max_total_bytes: int):
        self.max_concurrent = max_concurrent
        self.max_total_bytes = max_total_bytes
        self.active_count = 0
        self.active_bytes = 0
        self.lock = threading.Lock()
        self.cond = threading.Condition(self.lock)

    def acquire(self, file_size: int):
        with self.cond:
            while self.active_count >= self.max_concurrent or self.active_bytes + file_size > self.max_total_bytes:
                self.cond.wait()
            self.active_count += 1
            self.active_bytes += file_size
            print(f"[POOL] Active count: {self.active_count}, Active size: {self.active_bytes/1024**3:.2f} GB")

    def release(self, file_size: int):
        with self.cond:
            self.active_count -= 1
            self.active_bytes -= file_size
            print(f"[POOL] Released: Active count: {self.active_count}, Active size: {self.active_bytes/1024**3:.2f} GB")
            self.cond.notify_all()

# ---------------------------
# Helper functions
# ---------------------------
def load_processed_paths(track_file):
    if not os.path.exists(track_file):
        return set()
    with open(track_file, "r", encoding="utf-8") as f:
        return set(line.strip() for line in f if line.strip())

def append_processed_path(track_file, path):
    with open(track_file, "a", encoding="utf-8") as f:
        f.write(f"{path}\n")

def log_time(message):
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{message}: {now}\n")

def error_log(file_data, error_type, status, error):
    log_entry = {
        "file_data": file_data,
        "error_type": error_type,
        "status": status,
        "error": str(error)
    }
    error_file_path = os.path.join(os.path.dirname(LOG_FILE), f'transcribe_error_{SCRIPT_NAME}.log')
    with open(error_file_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')

def poll_until_complete(transcription_id):
    while True:
        res = SESSION.get(f"{API_BASE_URL}/v1/transcriptions/{transcription_id}")
        res.raise_for_status()
        data = res.json()
        if data["status"] == "completed":
            return
        elif data["status"] == "error":
            raise Exception(f"Transcription failed: {data.get('error_message', 'Unknown error')}")
        time.sleep(TRANSACRIPTION_PROCESSSING_SLEEP_IN_SEC)

def convert_to_mp3(input_file, output_file):
    """Convert audio file to MP3 using FFmpeg for transcription."""
    try:
        subprocess.run([
            "ffmpeg", "-y", "-i", input_file,
            "-codec:a", "libmp3lame", "-qscale:a", "2", output_file
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error converting {input_file} to MP3: {str(e)}")
        return False

# ---------------------------
# Process a single file
# ---------------------------
def process_single_file(file_info, pool):
    index, file = file_info
    curr_file_path = os.path.join(INPUT_FOLDER_PATH, file)
    filename = os.path.splitext(file)[0]
    new_folder_path = os.path.join(OUTPUT_FOLDER_PATH, filename)
    os.makedirs(new_folder_path, exist_ok=True)

    processed_paths = load_processed_paths(PROCESSED_TRACK_FILE)
    output_json_file = os.path.join(new_folder_path, f'{filename}.json')
    if output_json_file in processed_paths or os.path.exists(output_json_file):
        print(f"Skipping {curr_file_path}: Already processed {output_json_file}")
        return f"Skipped: {file}"

    if not os.path.isfile(curr_file_path):
        return f"Not a file: {file}"

    # ---------------------------
    # Convert original audio to MP3 for transcription
    # ---------------------------
    if not curr_file_path.lower().endswith(".mp3"):
        mp3_file = os.path.join(new_folder_path, f"{filename}.mp3")
        if not convert_to_mp3(curr_file_path, mp3_file):
            return f"MP3 conversion failed: {file}"
    else:
        mp3_file = curr_file_path

    file_size = os.path.getsize(mp3_file)
    pool.acquire(file_size)

    try:
        print(f"Starting file upload: {mp3_file}")
        res = SESSION.post(f"{API_BASE_URL}/v1/files", files={"file": open(mp3_file, "rb")})
        try:
            res.raise_for_status()
        except Exception as e:
            error_log({"filename": file}, "uploading", res.status_code, e)
            return f"Upload error: {file}"

        file_id = res.json()["id"]

        res = SESSION.post(
            f"{API_BASE_URL}/v1/transcriptions",
            json={
                "file_id": file_id,
                "model": SONIOX_TRANSCRIPTION_MODEL,
                "language_hints": LANGUAGE_HINTS,
                "enable_speaker_diarization": True,
                "enable_language_identification": True
            }
        )
        try:
            res.raise_for_status()
        except Exception as e:
            error_log({"filename": file, "uploaded_file_id": file_id}, "start_transcription", res.status_code, e)
            return f"Transcription start error: {file}"

        transcription_id = res.json()["id"]

        try:
            poll_until_complete(transcription_id)
        except Exception as e:
            error_log({"filename": file, "uploaded_file_id": file_id, "transcription_id": transcription_id}, "transcription", res.status_code, e)
            return f"Transcription polling error: {file}"

        res = SESSION.get(f"{API_BASE_URL}/v1/transcriptions/{transcription_id}/transcript")
        try:
            res.raise_for_status()
        except Exception as e:
            error_log({"filename": file, "uploaded_file_id": file_id, "transcription_id": transcription_id}, "get_transcription", res.status_code, e)
            return f"Get transcript error: {file}"

        resp_json = res.json()
        with open(output_json_file, 'w', encoding='utf-8') as f:
            json.dump(resp_json, f, ensure_ascii=False, indent=2)
            f.write('\n')

        append_processed_path(PROCESSED_TRACK_FILE, output_json_file)

        # Used to move the audio file to newly created folder
        if curr_file_path.lower().endswith(".mp3") and os.path.exists(curr_file_path):
            shutil.move(mp3_file, os.path.join(OUTPUT_FOLDER_PATH, filename))
        # ---------------------------
        # Delete the wav file after transcription
        # ---------------------------
        if curr_file_path.lower().endswith(".wav") and os.path.exists(curr_file_path):
            os.remove(curr_file_path)
            print(f"Deleted WAV file after transcription: {curr_file_path}")

        # Cleanup on API
        SESSION.delete(url=f"{API_BASE_URL}/v1/transcriptions/{transcription_id}").raise_for_status()
        SESSION.delete(url=f"{API_BASE_URL}/v1/files/{file_id}").raise_for_status()

        print(f"{index + 1}. Transcript completed for the file : {file}")
        return f"Completed: {file}"
    finally:
        pool.release(file_size)

# ---------------------------
# Main
# ---------------------------
def main():
    log_time(f"{SCRIPT_NAME} > Start Timing")

    if not os.path.isdir(INPUT_FOLDER_PATH):
        raise NotADirectoryError(f"The Path {INPUT_FOLDER_PATH} is not a valid directory.")

    files_to_process = [(i, f) for i, f in enumerate(os.listdir(INPUT_FOLDER_PATH)) if os.path.isfile(os.path.join(INPUT_FOLDER_PATH, f))]
    if not files_to_process:
        print("No files to process.")
        log_time(f"{SCRIPT_NAME} > Complete Timing (No files)")
        return

    pool = ProcessingPool(max_concurrent=MAX_CONCURRENT, max_total_bytes=MAX_TOTAL_BYTES)

    print(f"Processing {len(files_to_process)} files with concurrency limit: {MAX_CONCURRENT} and total size limit: {MAX_TOTAL_GB} GB")
    with ThreadPoolExecutor(max_workers=min(len(files_to_process), MAX_THREAD_POOL)) as executor:
        futures = {executor.submit(process_single_file, file_info, pool): file_info for file_info in files_to_process}
        for future in as_completed(futures):
            index, file = futures[future]
            try:
                result = future.result()
                print(f"[{index + 1}/{len(files_to_process)}] {result}")
            except Exception as e:
                print(f"Error processing {file}: {str(e)}")

    log_time(f"{SCRIPT_NAME} > Complete Timing")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="API key arguments")
    parser.add_argument('--apiKey', required=True, help='Soniox API key')
    args = parser.parse_args()

    if args.apiKey:
        SESSION.headers["Authorization"] = f"Bearer {args.apiKey}"
        main()
    else:
        print("Soniox API key is required to run the code.")