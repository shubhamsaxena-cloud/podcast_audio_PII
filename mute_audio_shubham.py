from pydub import AudioSegment
import json
import os

json_file_path = r"C:\code_lab\Vikas_code\transcripts_entities_with_timestamps.json"  

# Check if file exists
if not os.path.exists(json_file_path):
    raise FileNotFoundError(f"JSON file not found: {json_file_path}")

# Open and load the JSON
with open(json_file_path, 'r', encoding='utf-8') as f:
    data_raw = json.load(f)  # Load raw data

# Handle different JSON structures
if isinstance(data_raw, dict):
    # If single object: {"path": "...", "entities": [...]}
    data = [data_raw]
elif isinstance(data_raw, list):
    # If list of objects
    data = data_raw
else:
    raise ValueError("Unexpected JSON structure: not a dict or list")

# Extract ranges (filter out invalid/None timestamps)
mute_ranges = []
for item in data:
    entities = item.get("entities", [])
    for entity in entities:
        start = entity.get("start")
        end = entity.get("end")
        if start is not None and end is not None and isinstance(start, (int, float)) and isinstance(end, (int, float)):
            mute_ranges.append({"start": int(start), "end": int(end)})

# Output the fetched ranges (for verification)
print("Fetched mute ranges:")
for rng in mute_ranges:
    print(rng)

def mute_audio_segment(input_file, output_file, mute_ranges):
    """
    Mute specified timestamp ranges in an audio file.
    mute_ranges: List of dicts with 'start' and 'end' in milliseconds.
    """
    # Load the audio file
    audio = AudioSegment.from_file(input_file)
  
    # Create a silent audio segment for muting
    silence = AudioSegment.silent(duration=1)  # 1ms of silence
  
    # Process each mute range
    result = AudioSegment.empty()
    last_end = 0
  
    for mute_range in mute_ranges:
        start_ms = mute_range['start']
        end_ms = mute_range['end']
      
        # Add unmuted portion before this mute range
        if last_end < start_ms:
            result += audio[last_end:start_ms]
      
        # Add silent portion for the mute range
        mute_duration = end_ms - start_ms
        result += AudioSegment.silent(duration=mute_duration)
      
        last_end = end_ms
  
    # Add any remaining audio after the last mute
    if last_end < len(audio):
        result += audio[last_end:]
  
    # Export the resulting audio
    result.export(output_file, format=output_file.split('.')[-1])

# Example usage
if __name__ == "__main__":
    # Example input
   
   
    print("mute_ranges = [")
    for i, rng in enumerate(mute_ranges):
        start = rng["start"]
        end = rng["end"]
        if i == 0:
            # First item: Use your original example comment style
            print(f'    {{"start": {start}, # Original example')
            print(f'     "end": {end}}}')
        else:
            # Subsequent items: Add custom comments (adjust seconds as needed)
            start_sec = start // 1000
            end_sec = end // 1000
            print(f',   {{"start": {start}, # New: Mute from {start_sec} seconds to {end_sec} seconds')
            print(f'     "end": {end}}}')
    print("]")
  
    input_audio = "input_audio.wav"  # Replace with your input audio file
    output_audio = "output_audio.mp3"  # Replace with desired output file
  
    mute_audio_segment(input_audio, output_audio, mute_ranges)