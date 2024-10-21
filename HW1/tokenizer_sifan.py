import re
import sys

pattern_filename = r'^>>>.*<<<$' # regex pattern to match the file name


def read_abbreviations(file_path):
    with open(file_path, 'r') as f:
        abbre = [line.strip() for line in f.readlines()]# Read abbreviations and strip whitespace
    return abbre

def open_html_text(file_path):
    with open(file_path, 'r') as f:
        text = f.readlines()
    return text

def match_abbreviation(match, abbreviations):
    """Helper function to check if a token with a period is an abbreviation."""
    token = match.group(0) # Get the matched token
    # Check if the token without the period is in the abbreviations list
    if token[:-1] in abbreviations:
        return token
    return token.replace('.', '') # Remove the period if it is not an abbreviation

def tokenize_text(text, abbre):
    filtered_lines = [line for line in text if not re.match(pattern_filename, line.strip())]# filter out the file name
    combined_text = " ".join(filtered_lines) # combine the text
    
    text_no_space = re.sub(r'\s*-\s*', '-', combined_text) # remove spaces around hyphens, like " - " to "-"
    text_adding_space = " " + text_no_space + " " # Add spaces around the text for better handling of punctuation
    text_remove_space = re.sub(r'\s+', ' ', text_adding_space) # \s+: This regex pattern matches one or more whitespace characters (\s includes spaces, tabs, and newlines)
    
    pattern_date_1 = r'(\d{1,2}\.\s*\w+\s\d{4})'
    pattern_date_2 = r'(\d{1,2}\.\s+[A-Za-zäöüÄÖÜ]+)'
    pattern_date_3 = r'(Stand:\s*\d{2}\.\d{2}\.\d{4}\s+\d{2}:\d{2}\s+Uhr)'
    pattern_text_3 = r'(\(\w[\w\s.,-]*\))' # for (text/hyphen/period/comma/space)
    
    date_matches = re.findall(pattern_date_1, text_remove_space) # Find all date matches
    date_matches.extend(re.findall(pattern_date_2, text_remove_space))
    date_matches.extend(re.findall(pattern_date_3, text_remove_space))
    date_matches.extend(re.findall(pattern_text_3, text_remove_space))
    # print(date_matches)

    
    for idx, match in enumerate(date_matches): # index, date_match
        dateholder = f"__DATE_HOLDER_{idx}__" # Create a placeholder for each date match
        text_remove_space = text_remove_space.replace(match, dateholder) # ex: Replace the '13.Oktober 2024' with the __DATE_HOLDER_0__
        # print(date_matches)
        # print(text_remove_space)

    pattern_abbre = r'\b(?:[A-Za-z0-9]+\.-?)+\b' # regex pattern to match abbreviations
    processed_text = re.sub(pattern_abbre, lambda match: match_abbreviation(match, abbre), text_remove_space)

    sentences = re.split(r'(?<=[.!?])\s+', processed_text.strip()) # splited sentences with placeholders
    # print(sentences)
    processed_sentences = []
    for sentence in sentences:
        for idx, match in enumerate(date_matches):
            placeholder = f"__DATE_HOLDER_{idx}__"
            sentence = sentence.replace(placeholder, match) # Replace placeholders for dates back in each sentence
        

        processed_sentences.append(sentence)
    # print(processed_sentences)

    for sentence in processed_sentences:
        tokens = sentence.split()
        print(" ".join(tokens))


abbre = read_abbreviations('/Users/sifan/Library/Mobile Documents/com~apple~CloudDocs/MA/CL/EET/1/abbreviations.txt')
html = open_html_text('/Users/sifan/Library/Mobile Documents/com~apple~CloudDocs/MA/CL/EET/1/text.txt')


tokenize_text(html, abbre)
