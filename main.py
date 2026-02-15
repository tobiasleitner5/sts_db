import pandas as pd
import json
import logging
import argparse
from openai import OpenAI
from utils import extract_random_sentences_from_gzipped_csv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler('sts_generation.log'),
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

    # Your strategy: System Prompt + Instructions
    system_content = (
        f"System Prompt: {prompt_instruction} "
        f"Your output must always be a valid JSON object. "
        f"Do not include any conversational text, explanations, or markdown code blocks. "
        f"The JSON must follow this schema: "
        "{"
        '  "input_sentence": <input>,'
        '  "output_sentence": <output>'
        "}"
    )

    logger.info(f"PROMPT_TYPE={prompt_type} | INSTRUCTION={prompt_instruction[:80]}...")
    logger.info(f"INPUT={text_input[:100]}...")

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
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
        
        logger.info(f"OUTPUT={parsed_result.get('output_sentence', '')[:100]}...")
        return parsed_result
    
    except Exception as e:
        logger.error(f"ERROR={e}")
        return None

# Load input sentences from gzipped CSV files (first sentence from each Body)
results_database = []

# Get random sentences from gzipped files
sentences = extract_random_sentences_from_gzipped_csv(
    args.data_folder, 
    num_sentences=args.num_sentences, 
    filename_filter=args.filename_filter
)

logger.info(f"Starting STS generation | SENTENCES={args.num_sentences} | FILE_FILTER={args.filename_filter}")

# Process each sentence
for idx, input_sentence in enumerate(sentences, 1):
    logger.info(f"--- Processing {idx}/{len(sentences)} ---")
    
    # Alternate between Positive and Hard negative
    if idx % 2 == 1:
        row = positive_prompts.sample(1).iloc[0]
    else:
        row = hard_negative_prompts.sample(1).iloc[0]
    
    output = generate_sts_pair(row, input_sentence)
    if output:
        results_database.append(output)

# Save your new database
with open('sts_database.jsonl', 'w') as f:
    for entry in results_database:
        f.write(json.dumps(entry) + '\n')

logger.info(f"Database generation complete | TOTAL_ENTRIES={len(results_database)} | OUTPUT_FILE=sts_database.jsonl")