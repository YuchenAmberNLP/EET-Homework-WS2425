import os
import json
import sys
import nltk
from nltk.tokenize import word_tokenize
from collections import defaultdict
import string


def text_tokenize(text):
    tokens = word_tokenize(text)
    filtered_tokens = [word for word in tokens if word not in string.punctuation]
    return filtered_tokens


def get_word_freq(dir_name):
    word_frequency = defaultdict(int)
    total_word = 0

    for file_name in os.listdir(dir_name):
        if file_name == 'Summary.txt':
            continue
        file_path = os.path.join(dir_name, file_name)
        with open(file_path, 'r', encoding='ISO-8859-1') as f:
            text = f.read()
            words = text_tokenize(text)
            for word in words:
                word_frequency[word] += 1
                total_word += 1

    return word_frequency, total_word


class NaiveBayesClassifier:
    def __init__(self):
        self.word_freq_per_class = defaultdict(lambda: defaultdict(int))
        self.total_word_per_class = defaultdict(int)
        self.total_doc_per_class = defaultdict(int)
        self.class_prior = {}
        self.vocab = set()
        self.discount = {}

    def calculate_discount(self, word_freq):
        N1 = sum(1 for count in word_freq.values() if count == 1)
        N2 = sum(1 for count in word_freq.values() if count == 2)
        if N1 + 2 * N2 == 0:
            return 0
        return N1 / (N1 + 2 * N2)

    def train(self, train_dir):
        for class_name in os.listdir(train_dir):
            class_dir = os.path.join(train_dir, class_name)
            if not os.path.isdir(class_dir):
                continue

            word_freq, total_words = get_word_freq(class_dir)
            self.word_freq_per_class[class_name] = word_freq
            self.total_word_per_class[class_name] = total_words
            self.total_doc_per_class[class_name] = len(os.listdir(class_dir))
            self.vocab.update(word_freq.keys())

            self.discount[class_name] = self.calculate_discount(word_freq)

        total_docs = sum(self.total_doc_per_class.values())
        for class_name in self.total_doc_per_class:
            self.class_prior[class_name] = self.total_doc_per_class[class_name] / total_docs

    def save_params(self, paramfile_name):
        param_data = {
            'word_freq_per_class': {k: dict(v) for k, v in self.word_freq_per_class.items()},
            'total_word_per_class': dict(self.total_word_per_class),
            'class_prior': dict(self.class_prior),
            'vocab': list(self.vocab),
            'discount': dict(self.discount)
        }
        with open(paramfile_name, 'w', encoding='utf-8') as f:
            json.dump(param_data, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    nltk.download('punkt_tab')
    # nltk.download('punkt')
    train_dir = sys.argv[1]
    paramfile_name = sys.argv[2]
    classifier = NaiveBayesClassifier()
    classifier.train(train_dir)
    classifier.save_params(paramfile_name)
    print('paramfile saved successfully.')