import os
import pandas as pd
import json
import logging
import argparse
from openai import OpenAI
from utils import extract_random_sentences_from_gzipped_csv
from system_prompt import get_system_prompt

# Create output directories
os.makedirs('logs', exist_ok=True)
os.makedirs('output', exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler('logs/sts_generation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Parse command line arguments
parser = argparse.ArgumentParser(description='Generate STS sentence pairs using OpenAI')
parser.add_argument('--api-key', type=str, required=True, help='OpenAI API key')
parser.add_argument('--data-folder', type=str, required=True, help='Path to folder containing gzipped CSV files')
parser.add_argument('--filename-filter', type=str, default=None, help='Substring to filter filenames')
parser.add_argument('--num-sentences', type=int, default=500, help='Number of sentences to process (default: 500)')
args = parser.parse_args()

# Initialize the client
client = OpenAI(api_key=args.api_key)

# Load your cleaned CSV
df = pd.read_csv('prompts.csv', sep=';')

# Separate prompts by type
positive_prompts = df[df['Prompt type'] == 'Positive']
hard_negative_prompts = df[df['Prompt type'] == 'Hard negative']


def generate_sts_pair(row, text_input):
    prompt_instruction = row['Prompt']
    prompt_type = row['Prompt type']

    system_content = get_system_prompt(prompt_instruction)

    logger.info(f"PROMPT_TYPE={prompt_type} | INSTRUCTION={prompt_instruction[:80]}...")
    logger.info(f"INPUT={text_input[:100]}...")

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": text_input}
            ],
            temperature=0.7 # Slight randomness helps with STS diversity
        )
        
        # Parse the response to ensure it's valid JSON
        result = response.choices[0].message.content
        parsed_result = json.loads(result)
        
        # Add prompt metadata to the result
        parsed_result['prompt_type'] = prompt_type
        parsed_result['prompt_instruction'] = prompt_instruction
        parsed_result['input_sentence'] = text_input
        
        # Extract token usage
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        
        logger.info(f"OUTPUT={parsed_result.get('output_sentence', '')[:100]}...")
        return parsed_result, input_tokens, output_tokens
    
    except Exception as e:
        logger.error(f"ERROR={e}")
        return None, 0, 0

# Load input sentences from gzipped CSV files (first sentence from each Body)
results_database = []

# Get random sentences from gzipped files
sentences = extract_random_sentences_from_gzipped_csv(
    args.data_folder, 
    num_sentences=args.num_sentences, 
    filename_filter=args.filename_filter
)

logger.info(f"Starting STS generation | SENTENCES={args.num_sentences} | FILE_FILTER={args.filename_filter}")

# Token tracking
total_input_tokens = 0
total_output_tokens = 0

# Process each sentence
for idx, input_sentence in enumerate(sentences, 1):
    logger.info(f"--- Processing {idx}/{len(sentences)} ---")
    
    # Alternate between Positive and Hard negative
    if idx % 2 == 1:
        row = positive_prompts.sample(1).iloc[0]
    else:
        row = hard_negative_prompts.sample(1).iloc[0]
    
    output, input_tokens, output_tokens = generate_sts_pair(row, input_sentence)
    total_input_tokens += input_tokens
    total_output_tokens += output_tokens
    
    if output:
        results_database.append(output)

# Save database
with open('output/sts_database.jsonl', 'w') as f:
    for entry in results_database:
        f.write(json.dumps(entry) + '\n')

logger.info(f"Database generation complete | TOTAL_ENTRIES={len(results_database)} | OUTPUT_FILE=output/sts_database.jsonl")
logger.info(f"Token usage | INPUT_TOKENS={total_input_tokens} | OUTPUT_TOKENS={total_output_tokens} | TOTAL_TOKENS={total_input_tokens + total_output_tokens}")