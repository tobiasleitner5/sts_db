import gzip
import csv
import random
from pathlib import Path
import spacy
from spacy.language import Language

_NLP = None

_ABBREVIATIONS = {
    "inc.", "co.", "corp.", "ltd.", "plc.", "u.s.", "u.k.", "fed.", "sec.", "ftc.",
    "no.", "q1", "q2", "q3", "q4", "yr.", "est.", "approx.",
}


@Language.component("finance_sentencizer")
def _finance_sentencizer(doc):
    for token in doc[:-1]:
        if token.text.lower() in _ABBREVIATIONS and doc[token.i + 1].is_sent_start:
            doc[token.i + 1].is_sent_start = False
    return doc


def _get_nlp():
    global _NLP
    if _NLP is None:
        _NLP = spacy.load("en_core_web_sm")
        if "finance_sentencizer" not in _NLP.pipe_names:
            _NLP.add_pipe("finance_sentencizer", before="parser")
    return _NLP


def get_first_sentence(text):
    """Extract the first sentence from a text."""
    doc = _get_nlp()(text)
    for sent in doc.sents:
        return sent.text
    return text


def extract_random_sentences_from_gzipped_csv(data_folder, num_sentences, text_column="Body", filename_filter=None, seed=None):
    """Extract random first sentences from 'Body' column of a gzipped CSV file.
    
    Args:
        data_folder: Path to folder containing gzipped CSV files
        num_sentences: Number of random sentences to return
        text_column: The column name to extract text from
        filename_filter: Substring to filter filenames (only process files containing this string)
        seed: Random seed for reproducible sampling (default: None = non-deterministic)
    
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
    total_rows = 0
    missing_body = 0
    short_sentence = 0
    with gzip.open(gz_file, 'rt', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_rows += 1
            if text_column not in row or not row[text_column]:
                missing_body += 1
                continue
            sentence = get_first_sentence(row[text_column])
            # Only include sentences with more than 3 words
            if len(sentence.split()) > 3:
                all_sentences.append(sentence)
            else:
                short_sentence += 1
    
    print(f"Found {len(all_sentences)} sentences in file")
    print(f"Non-valid sentences | total_rows={total_rows} missing_body={missing_body} short_sentence={short_sentence}")
    
    # Return random sample
    if len(all_sentences) <= num_sentences:
        return all_sentences
    
    if seed is not None:
        random.seed(seed)
    
    return random.sample(all_sentences, num_sentences)
