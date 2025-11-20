import os
import re
import json
import logging
from openai import OpenAI
import time


# --- Logging setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def log_progress(file_path: str, tokens_used: int, log_file: str = "process.txt"):
    """Append file processing summary to process.txt"""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"{timestamp} | Processed: {os.path.basename(file_path)} | Tokens used: {tokens_used}\n")


# --- Initialize OpenAI client ---
def init_client(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key)

# --- Extract word timestamps from token data ---
def extract_word_timestamps(data):
    """
    Groups tokens into properly spaced words and returns word–timestamp mapping.
    Handles both dict and list input structures.
    """

    def group_tokens_by_spaces(tokens):
        words = []
        current_word = ''
        current_start = None
        current_end = None

        for token in tokens:
            raw_text = token.get('text', '')
            text = raw_text.strip()
            start = token.get('start_ms')
            end = token.get('end_ms')

            if not text:
                continue

            # Start a new word if token begins with space or no current word yet
            if raw_text.startswith(' ') or current_word == '':
                if current_word:
                    words.append({
                        'word': current_word,
                        'start': current_start,
                        'end': current_end
                    })
                current_word = text
                current_start = start
                current_end = end
            else:
                # Continuation of the same word
                current_word += text
                current_end = end

            # End a word if token ends with space
            if raw_text.endswith(' '):
                words.append({
                    'word': current_word,
                    'start': current_start,
                    'end': current_end
                })
                current_word = ''
                current_start = None
                current_end = None

        if current_word:
            words.append({
                'word': current_word,
                'start': current_start,
                'end': current_end
            })

        return words

    # Handle dict or list of dicts
    results = []
    if isinstance(data, list):
        for record in data:
            if not isinstance(record, dict):
                continue
            tokens = record.get('tokens', [])
            words = group_tokens_by_spaces(tokens)
            results.extend(words)
    elif isinstance(data, dict):
        tokens = data.get('tokens', [])
        results = group_tokens_by_spaces(tokens)
    else:
        raise ValueError("Unsupported data format")

    return {'word_timestamps': results}


# --- GPT Batch Processing Function ---
# --- GPT Batch Processing Function ---
def run_gpt_in_batches(client: OpenAI, text: str, model: str = "gpt-4o-mini") -> tuple[list, int]:
    """
    Batches text into ~500–800 token chunks (roughly 3000–3200 characters)
    to minimize API calls and avoid rate-limit (429) errors.
    """
    MAX_CHARS_PER_BATCH = 2000  # ≈ 800 tokens
    all_responses = []
    total_tokens_used = 0

    # Split by sentence punctuation
    sentences = re.split(r'(?<=[।.!?])\s+', text.strip())

    # --- Group sentences into bigger chunks ---
    chunks = []
    current_chunk = ""

    for sent in sentences:
        if len(current_chunk) + len(sent) + 1 <= MAX_CHARS_PER_BATCH:
            current_chunk += (" " + sent)
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sent

    if current_chunk:
        chunks.append(current_chunk.strip())

    logging.info(f"📦 Total chunks formed: {len(chunks)} (~{MAX_CHARS_PER_BATCH} chars each)")

    # --- Send each chunk to GPT ---
    for idx, chunk in enumerate(chunks, 1):
        if not chunk.strip():
            continue

        prompt = f"""
         आप एक NER सिस्टम हैं जो हिंदी टेक्स्ट से केवल **Proper Nouns (संज्ञा नाम)** निकालता है।
सिर्फ PERSON, ORG और LOCATION entities निकालें जो वास्तव में **विशेष नाम** हैं।

⚠️ इन शब्दों को कभी entity न मानें — ये सामान्य शब्द हैं:
'भाई', 'भाई साहब', 'मैडम', 'मैम', 'सर', 'जी', 'बच्चा', 'गुरु', 'पंडित', 'संत', 'बाबा', 
'लड़का', 'लड़की', 'आदमी', 'औरत', 'बंदा', 'टीचर', 'स्टूडेंट', 'स्कूल', 'क्लास', 'शहर', 
'गांव', 'मंदिर', 'कंपनी', 'ऑफिस', 'टीचर', आदि।

Entities को ठीक वैसा ही निकालें जैसा text में दिखता है — 
कोई बदलाव नहीं, कोई transliteration नहीं।

केवल JSON array में उत्तर दें, कोई explanation नहीं:
[
  {{"entity": "<नाम>", "type": "PERSON"}},
  {{"entity": "<संगठन>", "type": "ORG"}},
  {{"entity": "<स्थान>", "type": "LOCATION"}}
]
Text (part {idx}):
{chunk}
"""

        try:
            response = client.chat.completions.create(
                model=model,
                temperature=0,
                messages=[{"role": "user", "content": prompt}]
            )

            tokens_used = response.usage.total_tokens
            total_tokens_used += tokens_used

            gpt_output = response.choices[0].message.content
            logging.info(f"✅ Chunk {idx}/{len(chunks)} processed ({tokens_used} tokens)")

            try:
                parsed = json.loads(gpt_output)
            except json.JSONDecodeError:
                match = re.search(r'\[\s*{.*?}\s*(?:,\s*{.*?}\s*)*\]', gpt_output, re.DOTALL)
                parsed = json.loads(match.group()) if match else []
                if not parsed:
                    logging.warning(f"⚠️ Invalid GPT JSON for chunk {idx}: {gpt_output[:120]}...")

            all_responses.extend(parsed)

        except Exception as e:
            logging.error(f"❌ Error in chunk {idx}: {e}")

    return all_responses, total_tokens_used


# --- Main execution ---
if __name__ == "__main__":
    API_KEY = ""
    INPUT_FOLDER = r"C:\code_lab\Vikas_code\multi_lang_audio_output"

    client = init_client(API_KEY)

    for root, dirs, files in os.walk(INPUT_FOLDER):
        for file in files:
            if not file.endswith(".json"):
                continue

            # Skip any previously generated output files
            if any(suffix in file for suffix in [
                "_word_timestamps",
                "_sentence_timestamps",
                "_pii",
                "_person_timestamps"
            ]):
                continue

            input_json_path = os.path.join(root, file)
            output_json_path = os.path.join(root, f"{os.path.splitext(file)[0]}_pii.json")

            logging.info(f"📂 Processing: {input_json_path}")

            with open(input_json_path, "r", encoding="utf-8") as f:
                data_raw = json.load(f)


                # ✅ Extract word timestamps first
                result = extract_word_timestamps(data_raw)
                print(json.dumps(result, ensure_ascii=False, indent=2))

                # ✅ Save word–timestamp mapping to a separate JSON file
                word_ts_output_path = os.path.join(root, f"{os.path.splitext(file)[0]}_word_timestamps.json")
                with open(word_ts_output_path, "w", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                logging.info(f"💾 Word–timestamp mapping saved → {word_ts_output_path}")


                # ✅ Then process with GPT (optional)
                text_content = data_raw.get("text") or data_raw.get("full_text") or ""
                if not text_content.strip():
                    logging.warning(f"⚠️ No text found in {file}")
                    continue

                entities, total_tokens = run_gpt_in_batches(client, text_content)

                # --- Collect PERSON names ---
                person_names_set = {
                    ent.get("entity", "").strip()
                    for ent in entities
                    if isinstance(ent, dict)
                    and ent.get("type") == "PERSON"
                    and ent.get("entity")
                }
                person_names_set.discard("")

                # --- Match PERSON entities to word timestamps ---
                person_entities_with_time = []

                

                for wt in result["word_timestamps"]:
                    word = wt.get("word", "").strip()
                    start = wt.get("start")
                    end = wt.get("end")

                    # If the current word matches (exactly or partially) a detected PERSON name
                    for name in person_names_set:
                        name_parts = name.split()
                        if any(part == word or part in word for part in name_parts):
                            person_entities_with_time.append({
                                "entity": name,
                                "type": "PERSON",
                                "start": start,
                                "end": end
                            })
                            break  # move to next word after first match


                # --- Save in compact output format ---
                output_data = [
                    {
                        "path": input_json_path.replace("/", "\\"),
                        "entities": person_entities_with_time
                    }
                ]

                output_json_path = os.path.join(root, f"{os.path.splitext(file)[0]}_pii.json")
                with open(output_json_path, "w", encoding="utf-8") as f:
                    json.dump(output_data, f, ensure_ascii=False, indent=2)

                logging.info(f"✅ Clean PII JSON saved → {output_json_path}")


                # --- Match detected PERSON names to word timestamps ---
                
                person_time_ranges = []

                for wt in result["word_timestamps"]:
                    word = wt.get("word", "").strip()
                    start = wt.get("start")
                    end = wt.get("end")

                    # If word is one of the detected person names
                    if word in person_names_set:
                        person_time_ranges.append((word, start, end))

                # Optional: remove duplicates if same name repeats at same time
                person_time_ranges = list(set(person_time_ranges))

                logging.info(f"🧍 Matched PERSON name time ranges: {person_time_ranges}")

                # --- Sentence-wise JSON generation ---
                logging.info("🧩 Building sentence-level timestamp JSON")

                # Split text into sentences
                sentences = re.split(r'(?<=[।.!?])\s+', text_content.strip())
                sentence_blocks = []

                for sent in sentences:
                    sent_clean = sent.strip()
                    if not sent_clean:
                        continue

                    # find matching words in this sentence
                    matching_words = [w for w in result["word_timestamps"] if w["word"] and w["word"] in sent_clean]

                    if not matching_words:
                        continue

                    start_ms = matching_words[0]["start"]
                    end_ms = matching_words[-1]["end"]

                    # find person entities within this range
                    persons_in_sent = [
                        {"name": name, "start_ms": s, "end_ms": e}
                        for (name, s, e) in person_time_ranges
                        if s >= start_ms and e <= end_ms
                    ]

                    sentence_blocks.append({
                        "start_ms": start_ms,
                        "end_ms": end_ms,
                        "text": sent_clean,
                        "persons": persons_in_sent
                    })

                # Save to file
                sent_output_path = os.path.join(root, f"{os.path.splitext(file)[0]}_sentence_entities.json")
                with open(sent_output_path, "w", encoding="utf-8") as f:
                    json.dump({"content": sentence_blocks}, f, ensure_ascii=False, indent=2)

                log_progress(input_json_path, total_tokens)


                logging.info(f"💾 Sentence-level entity JSON saved → {sent_output_path}")



