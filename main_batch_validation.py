import os
import pandas as pd
import json
import logging
import argparse
from openai import OpenAI
from utils import extract_random_sentences_from_gzipped_csv
from system_prompt import get_system_prompt

# Parse command line arguments
parser = argparse.ArgumentParser(description='Validation: Generate STS pairs for every prompt × N sentences using OpenAI Batch API')
parser.add_argument('--api-key', type=str, required=True, help='OpenAI API key')
parser.add_argument('--data-folder', type=str, required=True, help='Path to folder containing gzipped CSV files')
parser.add_argument('--filename-filter', type=str, default=None, help='Substring to filter filenames')
parser.add_argument('--num-sentences', type=int, default=20, help='Number of sentences to process (default: 20)')
parser.add_argument('--mode', type=str, choices=['create', 'status', 'download'], default='create',
                    help='Mode: create batch, check status, or download results')
parser.add_argument('--batch-id', type=str, help='Batch ID for status/download modes')
parser.add_argument('--output-folder', type=str, default='/Volumes/Samsung PSSD T7 Media/data/ouput/sts_db',
                    help='Path to output folder (default: /Volumes/Samsung PSSD T7 Media/data/ouput/sts_db)')
args = parser.parse_args()

# Create output directories
os.makedirs(os.path.join(args.output_folder, 'logs'), exist_ok=True)
os.makedirs(os.path.join(args.output_folder, 'validation'), exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(os.path.join(args.output_folder, 'logs/sts_batch_validation.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize the client
client = OpenAI(api_key=args.api_key)

# Load prompts CSV — use ALL prompts (no sampling)
df = pd.read_csv('prompts/prompts.csv', sep=';')
all_prompts = df.reset_index(drop=True)


def create_batch_request(custom_id, row, text_input):
    """Create a single batch request entry."""
    prompt_instruction = row['Prompt']
    system_content = get_system_prompt(prompt_instruction)

    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": text_input}
            ],
            "temperature": 0.7
        }
    }


def create_batch():
    """Create validation batch: every prompt × every sentence."""
    # Get sentences
    sentences = extract_random_sentences_from_gzipped_csv(
        args.data_folder,
        num_sentences=args.num_sentences,
        filename_filter=args.filename_filter
    )

    total_requests = len(sentences) * len(all_prompts)
    logger.info(f"Creating validation batch | SENTENCES={len(sentences)} | PROMPTS={len(all_prompts)} | TOTAL_REQUESTS={total_requests}")

    # Store metadata for later processing
    metadata = {}
    requests = []
    request_idx = 0

    for sent_idx, input_sentence in enumerate(sentences, 1):
        for prompt_idx, row in all_prompts.iterrows():
            request_idx += 1
            custom_id = f"val-s{sent_idx}-p{prompt_idx}"
            request = create_batch_request(custom_id, row, input_sentence)
            requests.append(request)

            metadata[custom_id] = {
                "input_sentence": input_sentence,
                "sentence_idx": sent_idx,
                "prompt_idx": prompt_idx,
                "prompt_type": row['Prompt type'],
                "prompt_instruction": row['Prompt'],
                "prompt_source": row['Source']
            }

    # Write requests to a temporary file for upload
    temp_batch_file = os.path.join(args.output_folder, "validation/batch_requests_temp.jsonl")
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
        endpoint="/v1/chat/completions",
        completion_window="24h"
    )

    # Create batch-specific folder and save metadata + requests there
    batch_dir = os.path.join(args.output_folder, "validation", batch.id)
    os.makedirs(batch_dir, exist_ok=True)

    # Move requests file into batch folder
    batch_file = os.path.join(batch_dir, "batch_requests.jsonl")
    os.rename(temp_batch_file, batch_file)

    # Save metadata in batch folder
    with open(os.path.join(batch_dir, "batch_metadata.json"), 'w') as f:
        json.dump(metadata, f)

    logger.info(f"Batch created | BATCH_ID={batch.id} | STATUS={batch.status}")
    logger.info(f"Batch files saved | DIR={batch_dir}")
    logger.info(f"Run with --mode status --batch-id {batch.id} to check progress")

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
    """Download and process validation batch results."""
    batch = client.batches.retrieve(batch_id)

    if batch.status != "completed":
        logger.error(f"Batch not complete | STATUS={batch.status}")
        return

    # Load metadata from batch-specific folder
    batch_dir = os.path.join(args.output_folder, 'validation', batch_id)
    with open(os.path.join(batch_dir, "batch_metadata.json"), 'r') as f:
        metadata = json.load(f)

    # Download results
    result_file_id = batch.output_file_id
    result_content = client.files.content(result_file_id).text

    # Process results
    results_database = []
    total_input_tokens = 0
    total_output_tokens = 0

    for line in result_content.strip().split('\n'):
        result = json.loads(line)
        custom_id = result['custom_id']

        if result['response']['status_code'] == 200:
            response_body = result['response']['body']
            content = response_body['choices'][0]['message']['content']

            try:
                parsed_result = json.loads(content)

                # Add metadata
                meta = metadata[custom_id]
                parsed_result['input_sentence'] = meta['input_sentence']
                parsed_result['sentence_idx'] = meta['sentence_idx']
                parsed_result['prompt_idx'] = meta['prompt_idx']
                parsed_result['prompt_type'] = meta['prompt_type']
                parsed_result['prompt_instruction'] = meta['prompt_instruction']
                parsed_result['prompt_source'] = meta['prompt_source']

                results_database.append(parsed_result)

                # Track tokens
                usage = response_body['usage']
                total_input_tokens += usage['prompt_tokens']
                total_output_tokens += usage['completion_tokens']

            except json.JSONDecodeError as e:
                logger.error(f"JSON parse error | ID={custom_id} | ERROR={e}")
        else:
            logger.error(f"Request failed | ID={custom_id} | ERROR={result['response']}")

    # Save results as JSONL
    output_file = os.path.join(batch_dir, 'sts_validation.jsonl')
    with open(output_file, 'w') as f:
        for entry in results_database:
            f.write(json.dumps(entry) + '\n')

    # Save results as Excel
    excel_file = os.path.join(batch_dir, 'sts_validation.xlsx')
    results_df = pd.DataFrame(results_database)
    results_df.to_excel(excel_file, index=False)

    logger.info(f"Results saved | TOTAL_ENTRIES={len(results_database)} | JSONL={output_file} | EXCEL={excel_file}")
    logger.info(f"Token usage | INPUT={total_input_tokens} | OUTPUT={total_output_tokens} | TOTAL={total_input_tokens + total_output_tokens}")


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
