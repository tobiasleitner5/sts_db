import os
import pandas as pd
import json
import logging
import argparse
from openai import OpenAI
from utils import extract_random_sentences_from_gzipped_csv
from system_prompt import get_system_prompt, get_system_prompt_version

# Parse command line arguments
parser = argparse.ArgumentParser(description='Generate STS sentence pairs using OpenAI Batch API')
parser.add_argument('--api-key', type=str, required=True, help='OpenAI API key')
parser.add_argument('--data-folder', type=str, required=True, help='Path to folder containing gzipped CSV files')
parser.add_argument('--filename-filter', type=str, default=None, help='Substring to filter filenames')
parser.add_argument('--num-sentences', type=int, default=500, help='Number of sentences to process (default: 500)')
parser.add_argument('--mode', type=str, choices=['create', 'status', 'download'], default='create',
                    help='Mode: create batch, check status, or download results')
parser.add_argument('--batch-id', type=str, help='Batch ID for status/download modes')
parser.add_argument('--model', type=str, required=True, help='OpenAI model to use')
parser.add_argument('--prompt-type', type=str, choices=['positive', 'negative', 'both'], default='both',
                    help='Which prompt types to use (default: both)')
parser.add_argument('--output-folder', type=str, default='/Volumes/Samsung PSSD T7 Media/data/ouput/sts_db',
                    help='Path to output folder (default: /Volumes/Samsung PSSD T7 Media/data/ouput/sts_db)')
args = parser.parse_args()


# Create output directories
os.makedirs(os.path.join(args.output_folder, 'logs'), exist_ok=True)
os.makedirs(os.path.join(args.output_folder, 'output'), exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(os.path.join(args.output_folder, 'logs/sts_batch_generation.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize the client
client = OpenAI(api_key=args.api_key)

# Load prompts CSV
df = pd.read_csv('prompts/prompts.csv', sep=';')
positive_prompts = df[df['Prompt type'] == 'Positive']
hard_negative_prompts = df[df['Prompt type'] == 'Hard negative']


DEFAULT_TRACKING_COLUMNS = [
    "batch_id",
    "filename_filter",
    "num_sentences",
    "created_at",
    "downloaded",
    "system_prompt_version",
]


def _read_tracking_header(tracking_file):
    if not os.path.exists(tracking_file):
        return []
    with open(tracking_file, 'r') as f:
        first_line = f.readline().strip()
    if not first_line:
        return []
    return first_line.split(';')


def _normalize_tracking_file(tracking_file):
    if not os.path.exists(tracking_file):
        return
    with open(tracking_file, 'r') as f:
        lines = [line.rstrip('\n') for line in f]
    if not lines:
        return
    rows = [line.split(';') for line in lines if line != ""]
    if not rows:
        return
    header = rows[0]
    max_fields = max(len(r) for r in rows)
    if max_fields <= len(header) and all(len(r) == len(header) for r in rows[1:]):
        return
    new_header = header[:]
    if len(new_header) < max_fields:
        if len(new_header) == 5 and max_fields == 6 and "system_prompt_version" not in new_header:
            new_header.append("system_prompt_version")
        else:
            for i in range(len(new_header), max_fields):
                new_header.append(f"col_{i+1}")
    fixed_rows = [new_header]
    for r in rows[1:]:
        if len(r) < len(new_header):
            r = r + [""] * (len(new_header) - len(r))
        elif len(r) > len(new_header):
            r = r[:len(new_header)]
        fixed_rows.append(r)
    with open(tracking_file, 'w') as f:
        for r in fixed_rows:
            f.write(";".join(r) + "\n")


def create_batch_request(custom_id, row, text_input):
    """Create a single batch request entry."""
    prompt_instruction = row['Prompt']
    prompt_type = row['Prompt type']
    
    system_content = get_system_prompt(prompt_instruction, prompt_type)
    
    body = {
        "model": args.model,
        "input": [
            {"role": "system", "content": system_content},
            {"role": "user", "content": text_input}
        ]
    }
    if args.model == "gpt-5.2":
        body["reasoning"] = {"effort": "medium"}
    else:
        body["temperature"] = 0.7

    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/responses",
        "body": body
    }


def create_batch():
    """Create batch requests file and submit to OpenAI."""
    # Get sentences
    sentences = extract_random_sentences_from_gzipped_csv(
        args.data_folder,
        num_sentences=args.num_sentences,
        filename_filter=args.filename_filter
    )
    
    logger.info(f"Creating batch | SENTENCES={len(sentences)}")
    
    # Store metadata for later processing
    metadata = {}
    
    # Create JSONL file with all requests in a temp location first
    requests = []
    for idx, input_sentence in enumerate(sentences, 1):
        # Select prompt type
        if args.prompt_type == 'positive':
            row = positive_prompts.sample(1).iloc[0]
        elif args.prompt_type == 'negative':
            row = hard_negative_prompts.sample(1).iloc[0]
        else:
            # Alternate between Positive and Hard negative
            if idx % 2 == 1:
                row = positive_prompts.sample(1).iloc[0]
            else:
                row = hard_negative_prompts.sample(1).iloc[0]
        
        custom_id = f"request-{idx}"
        request = create_batch_request(custom_id, row, input_sentence)
        requests.append(request)
        
        # Store metadata to merge with results later
        metadata[custom_id] = {
            "input_sentence": input_sentence,
            "prompt_type": row['Prompt type'],
            "prompt_instruction": row['Prompt']
        }
    
    # Write requests to a temporary file for upload
    temp_batch_file = os.path.join(args.output_folder, "output/batch_requests_temp.jsonl")
    with open(temp_batch_file, 'w') as f:
        for request in requests:
            f.write(json.dumps(request) + '\n')
    
    logger.info(f"Created batch file | FILE={temp_batch_file}")
    
    # Upload file to OpenAI
    with open(temp_batch_file, 'rb') as f:
        uploaded_file = client.files.create(file=f, purpose="batch")
    
    logger.info(f"Uploaded file | FILE_ID={uploaded_file.id}")
    
    # Create batch job
    batch = client.batches.create(
        input_file_id=uploaded_file.id,
        endpoint="/v1/responses",
        completion_window="24h"
    )
    
    # Create batch-specific folder and save metadata + requests there
    batch_dir = os.path.join(args.output_folder, "output", batch.id)
    os.makedirs(batch_dir, exist_ok=True)
    
    # Move requests file into batch folder
    batch_file = os.path.join(batch_dir, "batch_requests.jsonl")
    os.rename(temp_batch_file, batch_file)
    
    # Save metadata in batch folder
    with open(os.path.join(batch_dir, "batch_metadata.json"), 'w') as f:
        json.dump(metadata, f)
    
    prompt_version = get_system_prompt_version()
    logger.info(
        "Batch created | BATCH_ID=%s | STATUS=%s | SYSTEM_PROMPT=%s | MODEL=%s | PROMPT_TYPE=%s",
        batch.id,
        batch.status,
        prompt_version,
        args.model,
        args.prompt_type,
    )
    logger.info(f"Batch files saved | DIR={batch_dir}")
    logger.info(f"Run with --mode status --batch-id {batch.id} to check progress")
    
    # Append batch job info to tracking file
    tracking_file = os.path.join(args.output_folder, "output/batch_jobs.csv")
    tracking_columns = _read_tracking_header(tracking_file)
    if not tracking_columns:
        tracking_columns = DEFAULT_TRACKING_COLUMNS[:]
        write_header = True
    else:
        write_header = False
    row_values = {
        "batch_id": batch.id,
        "filename_filter": args.filename_filter,
        "num_sentences": args.num_sentences,
        "created_at": pd.Timestamp.now().isoformat(),
        "downloaded": "no",
        "system_prompt_version": prompt_version,
    }
    with open(tracking_file, 'a') as f:
        if write_header:
            f.write(";".join(tracking_columns) + "\n")
        row = [str(row_values.get(col, "")) for col in tracking_columns]
        f.write(";".join(row) + "\n")
    
    logger.info(f"Batch info appended | FILE={tracking_file}")
    
    return batch.id


def check_status(batch_id):
    """Check the status of a batch job."""
    batch = client.batches.retrieve(batch_id)
    
    logger.info(f"Batch status | ID={batch_id}")
    logger.info(f"  STATUS={batch.status}")
    logger.info(f"  TOTAL={batch.request_counts.total}")
    logger.info(f"  COMPLETED={batch.request_counts.completed}")
    logger.info(f"  FAILED={batch.request_counts.failed}")
    
    if batch.status == "completed":
        logger.info(f"Run with --mode download --batch-id {batch_id} to get results")
    
    return batch.status


def download_results(batch_id):
    """Download and process batch results."""
    batch = client.batches.retrieve(batch_id)
    
    if batch.status != "completed":
        logger.error(f"Batch not complete | STATUS={batch.status}")
        return
    
    # Load metadata from batch-specific folder
    batch_dir = os.path.join(args.output_folder, 'output', batch_id)
    with open(os.path.join(batch_dir, "batch_metadata.json"), 'r') as f:
        metadata = json.load(f)
    
    # Download results
    result_file_id = batch.output_file_id
    result_content = client.files.content(result_file_id).text
    
    # Process results
    results_database = []
    total_input_tokens = 0
    total_output_tokens = 0

    def extract_output_text(response_body):
        if isinstance(response_body, dict) and response_body.get("output_text"):
            return response_body["output_text"]
        output_items = response_body.get("output", []) if isinstance(response_body, dict) else []
        parts = []
        for item in output_items:
            if item.get("type") != "message":
                continue
            for content in item.get("content", []):
                if content.get("type") in ("output_text", "text") and "text" in content:
                    parts.append(content["text"])
        return "".join(parts)

    def extract_usage(usage):
        if not isinstance(usage, dict):
            return 0, 0
        input_tokens = usage.get("input_tokens", usage.get("prompt_tokens", 0))
        output_tokens = usage.get("output_tokens", usage.get("completion_tokens", 0))
        return input_tokens, output_tokens
    
    for line in result_content.strip().split('\n'):
        result = json.loads(line)
        custom_id = result['custom_id']
        
        if result['response']['status_code'] == 200:
            response_body = result['response']['body']
            content = extract_output_text(response_body)
            
            try:
                parsed_result = json.loads(content)
                # Add metadata
                meta = metadata[custom_id]
                parsed_result['input_sentence'] = meta['input_sentence']
                parsed_result['prompt_type'] = meta['prompt_type']
                parsed_result['prompt_instruction'] = meta['prompt_instruction']
                
                results_database.append(parsed_result)
                
                # Track tokens
                usage = response_body.get('usage', {})
                input_tokens, output_tokens = extract_usage(usage)
                total_input_tokens += input_tokens
                total_output_tokens += output_tokens
                
            except json.JSONDecodeError as e:
                logger.error(f"JSON parse error | ID={custom_id} | ERROR={e}")
        else:
            logger.error(f"Request failed | ID={custom_id} | ERROR={result['response']}")
    
    # Save results
    batch_output_dir = os.path.join(args.output_folder, 'output', batch_id)
    os.makedirs(batch_output_dir, exist_ok=True)
    output_file = os.path.join(batch_output_dir, 'sts_database.jsonl')
    with open(output_file, 'w') as f:
        for entry in results_database:
            f.write(json.dumps(entry) + '\n')

    # Save results as Excel
    excel_file = os.path.join(batch_output_dir, 'sts_database.xlsx')
    results_df = pd.DataFrame(results_database)
    results_df.to_excel(excel_file, index=False)
    
    logger.info(f"Results saved | TOTAL_ENTRIES={len(results_database)} | JSONL={output_file} | EXCEL={excel_file}")
    logger.info(f"Token usage | INPUT={total_input_tokens} | OUTPUT={total_output_tokens} | TOTAL={total_input_tokens + total_output_tokens}")
    
    # Mark batch as downloaded in tracking file
    tracking_file = os.path.join(args.output_folder, "output/batch_jobs.csv")
    if os.path.exists(tracking_file):
        _normalize_tracking_file(tracking_file)
        tracking_df = pd.read_csv(tracking_file, sep=';')
        tracking_df.loc[tracking_df['batch_id'] == batch_id, 'downloaded'] = 'yes'
        tracking_df.to_csv(tracking_file, sep=';', index=False)
        logger.info(f"Tracking file updated | FILE={tracking_file}")


# Main execution
if args.mode == 'create':
    create_batch()
elif args.mode == 'status':
    if not args.batch_id:
        logger.error("--batch-id required for status mode")
    else:
        check_status(args.batch_id)
elif args.mode == 'download':
    if not args.batch_id:
        logger.error("--batch-id required for download mode")
    else:
        download_results(args.batch_id)
