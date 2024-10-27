import spacy
import re
import argparse
import os
from glob import glob
import sys
import spacy.cli
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.corpus import wordnet

# Load SpaCy model for English language
nlp = spacy.load("en_core_web_sm")

# Helper function to redact sensitive information
def redact_entities(doc, entities):
    redacted_text = doc.text
    stats = []
    
    for ent in entities:
        if ent.label_ in ['PERSON', 'DATE', 'GPE', 'ORG']:  # Handles names, dates, addresses, etc.
            # Replace with censor character █
            redacted_text = redacted_text.replace(ent.text, '█' * len(ent.text))
            stats.append(f"Censored {ent.label_}: {ent.text} at {ent.start_char}-{ent.end_char}")
    
    return redacted_text, stats

# Helper function to detect and redact phone numbers using regex
def redact_phone_numbers(text):
    phone_pattern = re.compile(r'(\(?\+?[0-9]{1,3}\)?[\s.-]?[0-9]{1,4}[\s.-]?[0-9]{1,4}[\s.-]?[0-9]{1,9})')
    redacted_text = re.sub(phone_pattern, lambda x: '█' * len(x.group()), text)
    matches = phone_pattern.findall(text)
    stats = [f"Censored PHONE: {match}" for match in matches]
    return redacted_text, stats

def get_related_words(concept):
    """Fetch related words for a given concept using WordNet."""
    synonyms = set()
    for syn in wordnet.synsets(concept):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name().lower().replace('_', ' '))
    return synonyms


# Helper function for censoring concept-based sentences
def redact_concept_sentences(doc, concepts):
    redacted_text = doc.text
    stats = []
    
    # for sent in doc.sents:
    #     if any(concept.lower() in sent.text.lower() for concept in concepts):
    #         redacted_text = redacted_text.replace(sent.text, '█' * len(sent.text))
    #         stats.append(f"Censored CONCEPT: {sent.text.strip()}")
    for sent in doc.sents:
        if any(word in concepts for word in sent.text.lower().split()):
            redacted_text = redacted_text.replace(sent.text, "█" * len(sent.text))
            stats.append(f"Censored CONCEPT: {sent.text.strip()}")
    return redacted_text, stats

# Function to process each file
def process_file(file, output_dir, redact_flags, concepts):
    with open(file, 'r', encoding='utf-8') as f:
        text = f.read()

    # Process the text using SpaCy
    doc = nlp(text)
    stats = []
    
    if 'names' in redact_flags or 'dates' in redact_flags or 'address' in redact_flags:
        # Redact names, dates, and addresses
        entities_to_redact = []
        if 'names' in redact_flags:
            entities_to_redact.extend([ent for ent in doc.ents if ent.label_ == 'PERSON'])
        if 'dates' in redact_flags:
            entities_to_redact.extend([ent for ent in doc.ents if ent.label_ == 'DATE'])
        if 'address' in redact_flags:
            entities_to_redact.extend([ent for ent in doc.ents if ent.label_ in ['GPE', 'ORG']])
        
        text, redaction_stats = redact_entities(doc, entities_to_redact)
        stats.extend(redaction_stats)
    
    if 'phones' in redact_flags:
        # Redact phone numbers
        text, phone_stats = redact_phone_numbers(text)
        stats.extend(phone_stats)
    
    if concepts:
        # Redact concept-based sentences
        doc = nlp(text)  # Re-run SpaCy since text may have changed
        concept_words = set()
        
        for concept in concepts:
            concept_words.update(get_related_words(concept))
        text, concept_stats = redact_concept_sentences(doc, concept_words)
        stats.extend(concept_stats)

    # Save the redacted text to a new file
    censored_filename = os.path.join(output_dir, os.path.basename(file) + '.censored')
    with open(censored_filename, 'w', encoding='utf-8') as out_file:
        out_file.write(text)

    return stats


def main():
    parser = argparse.ArgumentParser(description="Redact sensitive information from text files.")
    parser.add_argument('--input', nargs='+', help='Glob pattern for input files', required=True)
    parser.add_argument('--output', help='Directory to save redacted files', required=True)
    parser.add_argument('--names', action='store_true', help='Redact names')
    parser.add_argument('--dates', action='store_true', help='Redact dates')
    parser.add_argument('--phones', action='store_true', help='Redact phone numbers')
    parser.add_argument('--address', action='store_true', help='Redact addresses')
    parser.add_argument('--concept', action='append', help='Redact specific concept sentences')
    parser.add_argument('--stats', help='File or location to write statistics (stderr, stdout, or file path)', required=True)
    
    args = parser.parse_args()

    # Collect all flags
    redact_flags = []
    if args.names:
        redact_flags.append('names')
    if args.dates:
        redact_flags.append('dates')
    if args.phones:
        redact_flags.append('phones')
    if args.address:
        redact_flags.append('address')

    # Output directory check
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    # Process all files using glob
    stats = []
    for pattern in args.input:
        files = glob(pattern)
        for file in files:
            file_stats = process_file(file, args.output, redact_flags, args.concept)
            stats.extend(file_stats)

    # Write stats to the appropriate place (stderr, stdout, or file)
    if args.stats == 'stderr':
        import sys
        print("\n".join(stats), file=sys.stderr)
    elif args.stats == 'stdout':
        print("\n".join(stats))
    else:
        with open(args.stats, 'w') as stats_file:
            stats_file.write("\n".join(stats))

if __name__ == '__main__':
    main()