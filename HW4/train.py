import os
import json
import sys
from collections import defaultdict
import string


def text_tokenize(text):
    tokens = text.split()
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
        self.discount = None
        self.apriori_prob = {}
        self.r_w_given_c = defaultdict(lambda: defaultdict(float))
        self.r_w_given_c_sum = {}
        self.alpha_c = {}

    def import_dirs(self, train_dir):
        for class_name in os.listdir(train_dir):
            class_dir = os.path.join(train_dir, class_name)
            if not os.path.isdir(class_dir):
                continue

            word_freq, total_words = get_word_freq(class_dir)

            # self.total_doc_per_class[class_name] += len(os.listdir(class_dir))
            # self.total_word_per_class[class_name] += total_words
            # for word, freq in word_freq.items():
            #     self.word_freq_per_class[class_name][word] += freq
            #     self.vocab.add(word)


            self.word_freq_per_class[class_name] = word_freq
            self.total_word_per_class[class_name] = total_words
            self.total_doc_per_class[class_name] = len(os.listdir(class_dir))
            self.vocab.update(word_freq.keys())

        total_docs = sum(self.total_doc_per_class.values())
        self.class_prior = {c: count / total_docs for c, count in self.total_doc_per_class.items()}

    def calculate_discount_and_smoothing(self):
        N1 = 0
        N2 = 0
        for c, words in self.word_freq_per_class.items():
            for w, freq in words.items():
                if freq == 1:
                    N1 += 1
                elif freq == 2:
                    N2 += 1
        if N1 + 2 * N2 > 0:
            self.discount = N1 / (N1 + 2 * N2)
        else:
            self.discount = 0
        # print("discount calculated,", self.discount)

        for class_name, word_freq in self.word_freq_per_class.items():
            total_words = self.total_word_per_class[class_name]
            self.r_w_given_c[class_name] = {
                word: (max(0, freq - self.discount)/total_words) for word, freq in word_freq.items()
            } if total_words > 0 else 0
            self.r_w_given_c_sum[class_name] = sum(self.r_w_given_c[class_name].values())
            self.alpha_c[class_name] = (1 - self.r_w_given_c_sum[class_name])
        #     print("alpha-c calculated, ", alpha_c)
        # print("alpha and discount calculated.")

    def calculate_apriori_prob(self):
        total_count = sum(sum(words.values()) for words in self.word_freq_per_class.values())
        for w in self.vocab:
            freq_w = sum(self.word_freq_per_class[c][w] for c in self.word_freq_per_class)
            self.apriori_prob[w] = freq_w / total_count if total_count > 0 else 0

    def calculate_prob_with_smoothing(self, word, class_name):
        r_w_given_c = self.r_w_given_c[class_name].get(word, 0)
        p_w_backoff = self.apriori_prob.get(word, 0)
        prob_with_smoothing = r_w_given_c + self.alpha_c[class_name] * p_w_backoff

        return prob_with_smoothing


        
    def save_params(self, paramfile_name):
        word_prob_given_class = {}
        for class_name in self.word_freq_per_class:
            word_prob_given_class[class_name] = {
                word: self.calculate_prob_with_smoothing(word, class_name)
                for word in self.vocab
            }
            # 'word_freq_per_class': {k: dict(v) for k, v in self.word_freq_per_class.items()},
            # 'total_word_per_class': dict(self.total_word_per_class),
            # 'class_prior': dict(self.class_prior),
            # 'vocab': list(self.vocab),
            # 'discount': dict(self.discount),
        param_data = {
            'prob_per_class': self.class_prior,
            'word_prob_given_class': word_prob_given_class
        }
        with open(paramfile_name, 'w', encoding='utf-8') as f:
            json.dump(param_data, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    train_dir = sys.argv[1]
    paramfile_name = sys.argv[2]
    classifier = NaiveBayesClassifier()
    classifier.import_dirs(train_dir)
    classifier.calculate_discount_and_smoothing()
    classifier.calculate_apriori_prob()
    classifier.save_params(paramfile_name)
    print('paramfile saved successfully.')