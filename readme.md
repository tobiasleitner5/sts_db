# STS Database Generator

A tool for generating Semantic Textual Similarity (STS) sentence pairs using OpenAI's GPT-4o model. The generator creates positive paraphrases and hard negative examples from input sentences extracted from gzipped CSV files.

## Overview

This project generates sentence pairs for training STS models by:

1. **Extracting sentences** from gzipped CSV files (first sentence from each article body)
2. **Applying diverse prompts** that alternate between positive (paraphrase) and hard negative (contradicting) transformations
3. **Generating output sentences** using OpenAI's GPT-4o model
4. **Saving results** in JSONL format for downstream training

## Features

- Random sampling of input sentences from large datasets
- Alternating between positive and hard negative prompt types
- Configurable number of sentences to process
- Filename filtering for targeted data extraction
- Comprehensive logging with token usage tracking
- Support for both real-time and batch processing

## Installation

```bash
pip install -r requirements.txt
```

## Files

| File | Description |
|------|-------------|
| `main.py` | Real-time processing script (synchronous API calls) |
| `main_batch.py` | Batch processing script (OpenAI Batch API - 50% cheaper) |
| `utils.py` | Utility functions for data extraction |
| `prompts.csv` | Pool of prompts for positive and hard negative generation |
| `sts_database.jsonl` | Output file with generated sentence pairs |

## Usage

### Real-Time Processing (`main.py`)

For immediate results with synchronous API calls:

```bash
python3 main.py \
  --api-key "sk-your-openai-api-key" \
  --data-folder "/path/to/gzipped/csv/files" \
  --filename-filter "2000" \
  --num-sentences 500
```

#### Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--api-key` | Yes | - | Your OpenAI API key |
| `--data-folder` | Yes | - | Path to folder containing gzipped CSV files |
| `--filename-filter` | No | None | Only process files containing this substring |
| `--num-sentences` | No | 500 | Number of random sentences to process |

---

## Batch Processing (`main_batch.py`)

For large-scale processing with **50% cost savings**. The OpenAI Batch API processes requests within 24 hours at half the price.

### How It Works

The Batch API workflow consists of three steps:

1. **Create**: Prepare all requests, upload to OpenAI, and start batch job
2. **Status**: Poll the batch job to check progress
3. **Download**: Retrieve results once processing is complete

### Step 1: Create and Submit Batch

```bash
python3 main_batch.py \
  --api-key "sk-your-openai-api-key" \
  --data-folder "/path/to/gzipped/csv/files" \
  --filename-filter "2000" \
  --num-sentences 5000 \
  --mode create
```

This will output a batch ID (e.g., `batch_abc123`).

### Step 2: Check Status

```bash
python3 main_batch.py \
  --api-key "sk-your-openai-api-key" \
  --mode status \
  --batch-id "batch_abc123"
```

### Step 3: Download Results

Once the batch is complete:

```bash
python3 main_batch.py \
  --api-key "sk-your-openai-api-key" \
  --mode download \
  --batch-id "batch_abc123"
```

### Batch Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--api-key` | Yes | - | Your OpenAI API key |
| `--data-folder` | Yes* | - | Path to folder containing gzipped CSV files |
| `--filename-filter` | No | None | Only process files containing this substring |
| `--num-sentences` | No | 500 | Number of random sentences to process |
| `--mode` | No | create | Mode: `create`, `status`, or `download` |
| `--batch-id` | Yes** | - | Batch ID for status/download modes |

\* Required only for `create` mode  
\** Required only for `status` and `download` modes

### When to Use Each Script

| Use Case | Script | Why |
|----------|--------|-----|
| Quick testing (< 100 sentences) | `main.py` | Immediate results |
| Production runs (500+ sentences) | `main_batch.py` | 50% cost savings |
| Real-time applications | `main.py` | Low latency |
| Large-scale dataset creation | `main_batch.py` | Higher rate limits |

---

## Output Format

Results are saved to `sts_database.jsonl` with one JSON object per line:

```json
{
  "output_sentence": "A musician strums his guitar outdoors.",
  "input_sentence": "A man is playing a guitar in a park.",
  "prompt_type": "Positive",
  "prompt_instruction": "Please paraphrase the input sentence..."
}
```

## Prompt Types

The generator uses two types of prompts from `prompts.csv`:

- **Positive**: Generate paraphrases or entailed sentences (same meaning)
- **Hard Negative**: Generate contradicting or altered sentences (different meaning)

These alternate for balanced training data.

## Logging

Both scripts log to console and file:
- `main.py` → `sts_generation.log`
- `main_batch.py` → `sts_batch_generation.log`

Logs include timestamps, progress tracking, and token usage summaries.

## Data Requirements

Input gzipped CSV files must have a `Body` column containing article text. The script extracts the first sentence from each body and filters to sentences with more than 3 words.