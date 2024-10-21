import re
import sys
import requests


def load_abbreviations(file_path):
    """load abbreviations list file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        abbr_ls = [line.strip() for line in f if line.strip()]
        return abbr_ls

def tokenize_text(text_file, abbr_ls):
    with open(text_file, 'r') as f:
        text = f.read()
    abbr_pattern = r'\b(?:' + '|'.join(re.escape(abbr) for abbr in abbr_ls) + r')\b'
    text_with_abbr = re.sub(abbr_pattern, lambda x: f'ABBR_{x.group(0)}', text)
    pattern = r"\w+(?:'\w+)?|[^\w\s]"
    tokens = re.findall(pattern, text_with_abbr)
    tokens = [token.replace('ABBR_', '') if token.startswith('ABBR_') else token for token in tokens]

    # modify abbreviation with more than one words
    for abbr in abbr_ls:
        if ' ' in abbr:
            abbr_tokens = abbr.split()
            for token in abbr_tokens:
                if token in tokens:
                    tokens.remove(token)
            tokens.append(abbr)
    return tokens

def split_sentence(tokens):
    sentences = []
    current_sentence = []
    for token in tokens:
        current_sentence.append(token)
        # 检查当前 token 是否为句子的结束符号
        if token in {'.', '!', '?'}:
            sentences.append(current_sentence)
            current_sentence = []

    if current_sentence:
        sentences.append(current_sentence)

    return sentences

if __name__ == '__main__':
    abbr_file = sys.argv[1]
    text_file = sys.argv[2]
    abbr_ls = load_abbreviations(abbr_file)
    tokens = tokenize_text(text_file, abbr_ls)
    sentences = split_sentence(tokens)
    for sentence in sentences:
        print(' '.join(sentence))



