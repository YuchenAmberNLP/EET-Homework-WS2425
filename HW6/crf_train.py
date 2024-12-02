import sys
import math
from collections import defaultdict
import json
import random
import numpy

def load_data(filepath):
    sentences = []
    with open(filepath, 'r', encoding='utf-8') as file:
        sentence = []
        for line in file:
            if line.strip():
                word, tag = line.strip().split('\t')  # 分割單詞和標籤
                sentence.append((word, tag))
            else:
                if sentence:
                    sentences.append(sentence)
                    sentence = []
        if sentence:
            sentences.append(sentence)

    return sentences


class LCCRFTagger:
    def __init__(self, sentences, learning_rate=0.1, num_epochs=2):
        self.sentences = sentences
        tags = set()
        for sentence in self.sentences:
            for _, tag in sentence:
                tags.add(tag)
        tags.add('<s>')
        self.tags = tags
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.weights = defaultdict(dict)

    def log_sum_exp(self, scores):
        # prevent overflow
        max_score = max(scores)
        sum_exp = sum(math.exp(score - max_score) for score in scores)
        return max_score + math.log(sum_exp)

    def extract_word_features(self, sentence, i, prev_tag=None):
        """5 features in total"""
        word = sentence[i][0]
        features = {
            'word': word,
            'is_capitalized': word[0].upper() == word[0],
            #suffix of the word
            'suffix_2': word[-2:] if len(word) > 1 else word,
            'suffix_4': word[-4:] if len(word) > 3 else word
        }
        #extracting previous word
        if prev_tag is not None:
            features["prev_tag"] = prev_tag
        else:
            features["prev_tag"] = "<s>"
        return features


    def compute_feature_score(self, word_features, tag):
        score = 0.0
        for feature_name, feature_value in word_features.items():
            feature_key = f"{feature_name}={feature_value}"
            if tag in self.weights and feature_key in self.weights[tag]:
                score += self.weights[tag][feature_key]
        return score

    def forward(self, sentence):
        """
        forward alg for alpha
        """
        n = len(sentence)
        alpha = [{} for _ in range(n)]

        for tag in self.tags:
            word_features = self.extract_word_features(sentence, 0, None)
            alpha[0][tag] = self.compute_feature_score(word_features, tag)  # <s> 表示句首标签

        for i in range(1, n):
            for tag in self.tags:
                scores = [
                    alpha[i - 1][prev_tag] + self.compute_feature_score(self.extract_word_features(sentence, i, prev_tag), tag)
                    for prev_tag in self.tags
                ]
                alpha[i][tag] = self.log_sum_exp(scores)

        scores = [alpha[n - 1][tag] for tag in self.tags]
        z = self.log_sum_exp(scores)
        return alpha, z

    def backward(self, sentence):
        """
        backward alg for beta
        """
        n = len(sentence)
        beta = [{} for _ in range(n)]

        for tag in self.tags:
            beta[n - 1][tag] = 0  # log(1) = 0

        for i in range(n - 2, -1, -1):
            for tag in self.tags:
                word_features = self.extract_word_features(sentence, i, tag)
                scores = [
                    beta[i + 1][next_tag] + self.compute_feature_score(word_features, next_tag)
                    for next_tag in self.tags
                ]
                beta[i][tag] = self.log_sum_exp(scores)

        word_features = self.extract_word_features(sentence, 0, None)
        scores = [
            beta[0][tag] + self.compute_feature_score(word_features, tag)
            for tag in self.tags
        ]
        z = self.log_sum_exp(scores)
        return beta, z



    def compute_gradient(self, sentence, alpha, beta, z_log):
        gradient = defaultdict(lambda: defaultdict(float))
        n = len(sentence)

        # 提取真实标签序列
        true_tags = [tag for _, tag in sentence]        # beobachten hfk
        for i in range(n):
            word = sentence[i][0]
            prev_tag = true_tags[i - 1] if i > 0 else "<s>"
            current_tag = true_tags[i]

            word_features = self.extract_word_features(sentence, i, prev_tag)
            for feature_name, feature_value in word_features.items():
                feature_key = f"{feature_name}={feature_value}"
                gradient[current_tag][feature_key] += 1

        for i in range(n):
            word_features = self.extract_word_features(sentence, i, None)

            for t in self.tags:  # 当前标签
                for t_prev in self.tags:
                    log_marginal_prob = (
                            (alpha[i - 1][t_prev] if i > 0 else 0)
                            + self.compute_feature_score(word_features, t)
                            + beta[i][t]
                            - z_log
                    )
                    log_marginal_prob = max(log_marginal_prob, -103)
                    clipped_log_prob = min(log_marginal_prob, 700)
                    marginal_prob = math.exp(clipped_log_prob)

                    for feature_name, feature_value in word_features.items():
                        feature_key = f"{feature_name}={feature_value}"
                        # Subtract the expected counts from the model
                        gradient[t][feature_key] -= marginal_prob

        return gradient

    def update_weights(self, gradient):
        for tag, feature_gradients in gradient.items():
            for feature_key, grad_value in feature_gradients.items():
                if feature_key not in self.weights[tag]:
                    self.weights[tag][feature_key] = 0.0

                self.weights[tag][feature_key] += self.learning_rate * grad_value


    def train(self):
        sentences = self.sentences
        self.weights = defaultdict(lambda: defaultdict(float))

        for epoch in range(self.num_epochs):
            random.shuffle(sentences)

            for sentence in sentences:
                alpha, z_log_a = self.forward(sentence)
                beta, z_log_b = self.backward(sentence)
                # assert math.isclose(z_log_a, z_log_b, rel_tol=1e-6), "Forward and Backward results do not match!"
                # print(z_log_a, z_log_b)
                gradient = self.compute_gradient(sentence, alpha, beta, z_log_a)

                self.update_weights(gradient)

            print(f"Epoch {epoch + 1}/{self.num_epochs} completed")


    def save_params(self, filepath):

        param_weights = defaultdict(dict)

        for tag, features in self.weights.items():
            for feature, value in features.items():
                if "=" in feature:
                    feature_type, feature_value = feature.split("=", 1)
                    key = f"{feature_value}-{tag}"  # 生成新的键
                    param_weights[f"{feature_type}-tag"][key] = value
                else:
                    raise ValueError(f"Unexpected feature format: {feature}")
        params = {
            "tags": list(self.tags),
            "weights": dict(param_weights)
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(params, f, ensure_ascii=False, indent=4)
        print(f"Model parameters saved to {filepath}")






if __name__ == "__main__":
    train_file = sys.argv[1]
    param_file = sys.argv[2]
    train_sentences = load_data(train_file)
    print("train_sentences loaded")


    # initialize model
    sub_train_len = int(len(train_sentences) * 0.1)  #
    sub_train_sentences = random.sample(train_sentences, sub_train_len)
    model = LCCRFTagger(sub_train_sentences, num_epochs=2)
    print(len(model.tags))
    # print(model.tags)
    # print(len(model.tags))
    model.train()
    model.save_params(param_file)
