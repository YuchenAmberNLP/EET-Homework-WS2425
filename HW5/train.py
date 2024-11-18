import os
import json
import sys
from collections import defaultdict
import string
import random
import math


def text_tokenize(text):
    tokens = text.split()
    filtered_tokens = [word for word in tokens if word not in string.punctuation]
    # print(tokens)
    return filtered_tokens

def get_word_freq(text):
    word_frequency = defaultdict(int)

    words = text_tokenize(text)
    for word in words:
        word_frequency[word] += 1
    return word_frequency


def load_mail_data(mail_dir):
    class_data = {}
    for class_name in os.listdir(mail_dir):
        class_dir = os.path.join(mail_dir, class_name)
        if os.path.isdir(class_dir):
            emails = []
            for filename in os.listdir(class_dir):
                file_path = os.path.join(class_dir, filename)
                if os.path.isfile(file_path):
                    with open(file_path, 'r', encoding='ISO-8859-1') as f:
                        text = f.read()
                        emails.append(text)
            class_data[class_name] = emails
    return class_data


class LogLinearClassifier:
    def __init__(self, classes, learning_rate=0.1):
        self.classes = classes
        self.weights = {c: defaultdict(float) for c in self.classes}
        self.learning_rate = learning_rate
        self.best_epoch = None
        self.best_learning_rate = None
        self.best_dev_f1 = 0.0

    def get_features(self, text):
        vector = defaultdict(float)
        for word in text_tokenize(text):
            vector[word] += 1.0
        return vector

    def logsumexp(self, scores):
        # prevent overflow
        max_score = max(scores)
        sum_exp = sum(math.exp(score - max_score) for score in scores)
        return max_score + math.log(sum_exp)

    def predict(self, features):
        scores = {}
        for c in self.classes:
            scores[c] = sum(features[word] * self.weights[c][word] for word in features)

        logZ = self.logsumexp(scores.values())
        probs = {c: math.exp(scores[c] - logZ) for c in self.classes}
        return probs

    def update_weights(self, features, true_class):
        predicted_probs = self.predict(features)

        for c in self.classes:
            for word, count in features.items():
                if c == true_class:
                    gradient = count * (1 - predicted_probs[c])
                else:
                    gradient = -count * predicted_probs[c]
                self.weights[c][word] += self.learning_rate * gradient

    def train(self, texts, labels, epochs=10):
        for epoch in range(epochs):
            samples = list(zip(texts, labels))
            random.shuffle(samples)  # 打亂數據集
            for email, label in samples:
                features = self.get_features(email)
                self.update_weights(features, label)


    def save_params(self, paramfile_path):
        weights_dict = {c: dict(self.weights[c]) for c in self.weights}
        params = {
            "classes": self.classes,
            "weights": weights_dict
        }

        with open(paramfile_path, 'w', encoding='utf-8') as f:
            json.dump(params, f, ensure_ascii=False, indent=4)
        print(f"Model parameters saved to {paramfile_path}")


    # def calculate_f1(self, dev_data, dev_labels):
    #     predictions = [self.classify(email) for email in dev_data]
    #     return f1_score(dev_labels, predictions, average='macro')

    # def train(self, texts, labels, dev_data, dev_labels, epochs=10, learning_rates=[0.01, 0.1, 1.0]):
    #     for lr in learning_rates:
    #         self.learning_rate = lr
    #         print(f"Training with learning rate: {lr}")
    #         for epoch in range(epochs):
    #             samples = list(zip(texts, labels))
    #             random.shuffle(samples)
    #             for text, label in samples:
    #                 features = self.get_features(text)
    #                 self.update_weights(features, label)
    #                 f1 = self.calculate_f1(dev_data, dev_labels)
    #                 print(f"Epoch {epoch + 1}, Learning rate {lr}, F1 Score: {f1}")
    #
    #                 if f1 > self.best_dev_f1:
    #                     self.best_dev_f1 = f1
    #                     self.best_learning_rate = lr
    #                     self.best_epoch = epoch + 1
    #                     print(f"New best model found: Learning rate = {lr}, Epoch = {epoch + 1}, F1 Score = {f1}")
    #
    #     print(f"Best Model: Learning rate = {self.best_learning_rate}, Epoch = {self.best_epoch}, F1 Score = {self.best_f1}")
    #


if __name__ == "__main__":
    train_dir = sys.argv[1]
    paramfile_name = sys.argv[2]
    train_data = load_mail_data(train_dir)
    texts = []
    labels = []
    for class_name, email_list in train_data.items():
        for email in email_list:
            texts.append(email)
            labels.append(class_name)
    classifier = LogLinearClassifier(list(train_data.keys()))
    classifier.train(texts=texts, labels=labels)
    classifier.save_params(paramfile_name)
    print('paramfile saved successfully.')




