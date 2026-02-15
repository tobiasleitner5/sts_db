import gzip
import csv
import random
from pathlib import Path
import nltk
from nltk.tokenize import sent_tokenize

# Download punkt tokenizer for sentence splitting (only needed once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


def get_first_sentence(text):
    """Extract the first sentence from a text."""
    sentences = sent_tokenize(text)
    return sentences[0] if sentences else text


def extract_random_sentences_from_gzipped_csv(data_folder, num_sentences, text_column="Body", filename_filter=None):
    """Extract random first sentences from 'Body' column of a gzipped CSV file.
    
    Args:
        data_folder: Path to folder containing gzipped CSV files
        num_sentences: Number of random sentences to return
        text_column: The column name to extract text from
        filename_filter: Substring to filter filenames (only process files containing this string)
    
    Returns:
        List of random first sentences
    """
    gz_files = list(Path(data_folder).glob("**/*.gz"))
    
    # Filter to files containing the substring
    if filename_filter:
        gz_files = [f for f in gz_files if filename_filter in f.name]
    
    print(f"Found {len(gz_files)} matching gzipped files")
    
    if not gz_files:
        print(f"No files found matching filter: {filename_filter}")
        return []

    # Only process the first matching file
    gz_file = gz_files[0]
    print(f"Processing file: {gz_file}")
    
    # First, collect all valid sentences
    all_sentences = []
    with gzip.open(gz_file, 'rt', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if text_column in row and row[text_column]:
                sentence = get_first_sentence(row[text_column])
                # Only include sentences with more than 3 words
                if len(sentence.split()) > 3:
                    all_sentences.append(sentence)
    
    print(f"Found {len(all_sentences)} sentences in file")
    
    # Return random sample
    if len(all_sentences) <= num_sentences:
        return all_sentences
    
    return random.sample(all_sentences, num_sentences)