import argparse
import glob
import os
import re
import spacy
import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet
import sys

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

def add_phone_component(nlp):
    """Adds a component to recognize phone numbers."""
    pattern = r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
    def phone_recognizer(doc):
        for match in re.finditer(pattern, doc.text):
            span = doc.char_span(match.start(), match.end(), label="PHONE")
            if span is not None:
                doc.ents += (span,)
        return doc
    return phone_recognizer

# Add phone recognizer to pipeline
nlp.add_pipe(add_phone_component(nlp), last=True)

def get_related_words(concept):
    """Fetch related words for a given concept using WordNet."""
    synonyms = set()
    for syn in wordnet.synsets(concept):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name().lower().replace('_', ' '))
    return synonyms

def censor_text(doc, entity_types, concept_words):
    """Redacts sensitive information based on specified flags."""
    redacted_text = doc.text

    # Censor entity types (names, dates, phones, addresses)
    for ent in doc.ents:
        if ent.label_ in entity_types:
            redacted_text = redacted_text.replace(ent.text, "█" * len(ent.text))

    # Censor concepts (whole sentences containing related words)
    for sent in doc.sents:
        if any(word in concept_words for word in sent.text.lower().split()):
            redacted_text = redacted_text.replace(sent.text, "█" * len(sent.text))

    return redacted_text

def log_statistics(censored_items, filepath, stats_output):
    """Logs statistics of redacted items."""
    stats = f"File: {filepath}\n"
    for item, count in censored_items.items():
        stats += f"{item}: {count}\n"
    stats += "\n"

    if stats_output == 'stderr':
        sys.stderr.write(stats)
    elif stats_output == 'stdout':
        print(stats)
    else:
        with open(stats_output, 'a') as f:
            f.write(stats)

def censor_file(filepath, entity_types, concept_words, output_dir, stats_output):
    """Reads and processes the file to censor sensitive information."""
    with open(filepath, 'r') as f:
        text = f.read()

    # Run SpaCy NER and custom censoring
    doc = nlp(text)
    redacted_text = censor_text(doc, entity_types, concept_words)

    # Write censored output
    output_path = os.path.join(output_dir, os.path.basename(filepath) + ".censored")
    with open(output_path, 'w') as f:
        f.write(redacted_text)

    # Log statistics
    censored_items = {
        "NAMES": sum(1 for ent in doc.ents if ent.label_ == "PERSON" and "NAMES" in entity_types),
        "DATES": sum(1 for ent in doc.ents if ent.label_ == "DATE" and "DATES" in entity_types),
        "PHONES": sum(1 for ent in doc.ents if ent.label_ == "PHONE" and "PHONES" in entity_types),
        "ADDRESS": sum(1 for ent in doc.ents if ent.label_ == "GPE" and "ADDRESS" in entity_types),  # Adjust if you want a different label
        "CONCEPTS": sum(1 for sent in doc.sents if any(word in concept_words for word in sent.text.lower().split()))
    }
    log_statistics(censored_items, filepath, stats_output)

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Censor sensitive items in text files")
    parser.add_argument('--input', nargs='+', required=True, help='Input file pattern')
    parser.add_argument('--output', required=True, help='Directory to save censored files')
    parser.add_argument('--names', action='store_true', help='Censor names')
    parser.add_argument('--dates', action='store_true', help='Censor dates')
    parser.add_argument('--phones', action='store_true', help='Censor phone numbers')
    parser.add_argument('--address', action='store_true', help='Censor addresses')
    parser.add_argument('--concept', action='append', help='Concepts to censor')
    parser.add_argument('--stats', default='stderr', help='Output for statistics')
    args = parser.parse_args()

    # Set up directory
    os.makedirs(args.output, exist_ok=True)

    # Define entity types based on flags
    entity_types = set()
    if args.names:
        entity_types.add("PERSON")
    if args.dates:
        entity_types.add("DATE")
    if args.phones:
        entity_types.add("PHONE")
    if args.address:
        entity_types.add("GPE")

    # Prepare concept words if any
    concept_words = set()
    if args.concept:
        for concept in args.concept:
            concept_words.update(get_related_words(concept))

    # Process each file
    for pattern in args.input:
        for filepath in glob.glob(pattern):
            try:
                censor_file(filepath, entity_types, concept_words, args.output, args.stats)
            except Exception as e:
                print(f"Error processing file {filepath}: {e}")

if __name__ == '__main__':
    main()
