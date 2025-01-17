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
                    sentences.append(list(zip(*sentence)))
                    sentence = []
        if sentence: # add the last sentence
            sentences.append(list(zip(*sentence)))

    return sentences


def read_paramfile(filepath):
    with open(filepath, 'r') as f:
        params = json.load(f)

    tags = params["tags"]

    weights = defaultdict(dict)

    feature_mappings = [
        ("word-tag", "word"),
        ("suffix-tag", "suffix"),
        ("prev_tag-tag", "prev_tag")
    ]
    for weights_key, feature_type in feature_mappings:
        feature_weights = params["weights"].get(weights_key, {})
        # print(feature_weights)
        for key, value in feature_weights.items():
            feature, tag = key.rsplit('-', 1)
            # print(feature)
            feature_key = f"{feature_type}={feature}"
            weights[tag][feature_key] = value
    
    print(f"Weights loaded from {filepath}: {dict(weights)}")

    return tags, weights


class LCCRFTagger:
    def __init__(self, sentences=None, load_paramfile=None):
        self.best_weights = defaultdict(dict)
        # cache
        self.feature_cache = {}
        self.score_cache = {}

        # initialize or load tags and weights
        if sentences != None and load_paramfile == None:
            self.sentences = sentences
            tags = set(tag for words, tags in sentences for tag in tags)
            self.tags = tags
            self.weights = defaultdict(lambda: defaultdict(lambda: random.uniform(-0.01, 0.01)))
        elif load_paramfile != None and sentences == None:
            print("paramfile loaded.")
            self.tags, self.weights = read_paramfile(load_paramfile)


    def log_sum_exp(self, scores):
        # prevent overflow
        max_score = max(scores)
        sum_exp = sum(math.exp(score - max_score) for score in scores)
        return max_score + math.log(sum_exp)


    def extract_context_features(self, prev_tag='<s>'):
        # print(words)
        return ([f'prev_tag={prev_tag}'])

    def extract_lex_features(self, words, i):

        if i == len(words):
            return (['word=<s>'])
        else:
            word = words[i]
            # print(words)
            return ([f'word={word}',
                     f'is_capitalized={word[0].isupper()}'] +
                    [f'suffix={word[-l:]}' for l in range(1,max(len(word), 6))])



    def extract_word_features(self, words, i, prev_tag='<s>'):
        cache_key = (i, prev_tag)
        if cache_key in self.feature_cache:
            return self.feature_cache[cache_key]

        features = self.extract_lex_features(words, i) + self.extract_context_features(prev_tag)
        # if i == len(words):
        #     return (['word=<s>',
        #              f'prev_tag={prev_tag}'])
        # else:
        #     word = words[i]
        #     # print(words)
        #     features = [f'word={word}',
        #                 f'is_capitalized={word[0].isupper()}',
        #                 f'prev_tag={prev_tag}'] + [f'suffix={word[-l:]}' for l in range(1,max(len(word), 6))]

        self.feature_cache[cache_key] = features
        return features


    def clear_cache(self):
        self.feature_cache.clear()
        self.score_cache.clear()

    def compute_feature_score(self, features, tag):
        # use cache
        total_score = 0.0
        for feature in features:
            cache_key = (feature, tag)

            if cache_key in self.score_cache:
                feature_score = self.score_cache[cache_key]
            else:
                feature_score = self.weights[tag].get(feature, 0.0)
                self.score_cache[cache_key] = feature_score

            total_score += feature_score

        return total_score

    def forward(self, sentence, threshold):
        """
        forward alg for alpha
        """
        words = sentence[0]
        n = len(words)
        alpha = [{} for _ in range(n+2)]

        alpha[0]['<s>'] = 0.0

        for i in range(1, n+2):
            max_score = float("-inf")
            scores = {}
            for tag in self.tags if i < n+1 else ['<s>']:
                # print(i-1)
                scores[tag] = self.log_sum_exp([
                    prev_score + self.compute_feature_score(
                        self.extract_word_features(words, i - 1, prev_tag), tag
                    )
                    for prev_tag, prev_score in alpha[i - 1].items()
                ])
                max_score = max(max_score, scores[tag])
            prune_threshold = max_score + math.log(threshold)
            alpha[i] = {tag: score for tag, score in scores.items() if score > prune_threshold}

        return alpha

    def backward(self, sentence, alpha):
        """
        backward alg for beta
        """
        words = sentence[0]
        n = len(words)
        beta = [{} for _ in range(n + 2)]  # +2 for <s> at the start and end

        # Initialize beta at the end position (n+1) with <s>
        beta[n + 1]['<s>'] = 0  # log(1) = 0

        # Iterate from the second to last position (n) down to the first position (0)
        for i in range(n, -1, -1):
            for tag in alpha[i] if i > 0 else ['<s>']:  # Use <s> for the last position
                word_features = self.extract_word_features(words, i, tag)
                scores = [
                    beta[i + 1][next_tag] + self.compute_feature_score(word_features, next_tag)
                    for next_tag, next_score in beta[i + 1].items()
                ]
                beta[i][tag] = self.log_sum_exp(scores)

        return beta



    def compute_gradient(self, sentence, alpha, beta, l1_lambda):
        gradient = defaultdict(lambda: defaultdict(float))

        # beobachten hfk
        words, tags = sentence
        n = len(words)
        # print(words)
        for i, (prev_tag, tag) in enumerate(zip(['<s>'] + list(tags), list(tags) + ['<s>'])):
            for feature in self.extract_word_features(words, i, prev_tag):
                gradient[tag][feature] += 1

        log_z = alpha[-1]['<s>']
        max_exp = 700  # Prevent Overflow with clipping, maximum value that exp can handle

        for i in range(1, n+2):
            for t, beta_score in beta[i].items():
                # print(i)
                # expected lexical feature values
                # print(i, words)
                lex_features = self.extract_lex_features(words, i-1)
                lex_score = self.compute_feature_score(lex_features, t)
                gamma = math.exp(alpha[i][t] + beta_score - log_z)
                for feature in lex_features:
                    gradient[t][feature] -= gamma

                for t_prev, alpha_score in alpha[i-1].items():
                    # expected context feature values
                    context_features = self.extract_context_features(t_prev)
                    context_score = self.compute_feature_score(context_features, t)
                    stable_value = alpha_score + lex_score + context_score + beta_score - log_z # log-sum-exp trick for preventing overflow -> subtracting a large constant to stabilize the computation.
                    gamma = math.exp(min(stable_value, max_exp))
                    for feature in context_features:
                        gradient[t][feature] -= gamma

        for tag, feature_weights in self.weights.items():
            for feature, weight in feature_weights.items():
                # L1 正则化对梯度的贡献
                l1_penalty = l1_lambda * (1 if weight > 0 else -1)
                gradient[tag][feature] = gradient[tag].get(feature, 0.0) - l1_penalty

        return gradient

    def update_weights(self, gradient, learning_rate, max_grad=5.0):
        for tag, feature_gradients in gradient.items():
            for feature_key, grad_value in feature_gradients.items():
                # Clip the gradient
                grad_value = max(min(grad_value, max_grad), -max_grad)
                self.weights[tag][feature_key] = self.weights[tag].get(feature_key, 0.0) + learning_rate * grad_value

    def train(self, train_sentences, dev_sentences, num_epochs=2, learning_rate=0.05, l1_lambda=0.1, threshold=0.001):
        sentences = train_sentences
        best_accuracy = 0
        self.weights = defaultdict(lambda: defaultdict(float))

        for epoch in range(num_epochs):
            random.shuffle(sentences)
            print(f"Epoch {epoch + 1}/{num_epochs}")

            for idx, sentence in enumerate(sentences):
                print(f"Training on sentence {idx + 1}/{len(train_sentences)}: {sentence}")
                alpha = self.forward(sentence, threshold)
                beta = self.backward(sentence, alpha)
                self.clear_cache() # clear cache

                gradient = self.compute_gradient(sentence, alpha, beta, l1_lambda)

                self.update_weights(gradient, learning_rate)

            accuracy = self.evaluate(dev_sentences)
            print(f"Epoch {epoch + 1}/{num_epochs}, Accuracy: {accuracy}")

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                self.best_weights = self.weights

            print(f"Epoch {epoch + 1}/{num_epochs} completed")

    def viterbi(self, words):
        print(f"Running Viterbi for words: {words}")
        n = len(words)
        dp = [{} for _ in range(n+2)]
        backpointer = [{} for _ in range(n+2)]
        dp[0]["<s>"] = 0.0  # start of a sentence, score = 0
        for tag in self.tags:
            features = self.extract_word_features(words, 0)
            score = self.compute_feature_score(features, tag)
            dp[1][tag] = score
            backpointer[1][tag] = "<s>"
        # from 2nd word in the sentence:
        for i in range(2, n + 1):
            for tag in self.tags:
                max_score = -math.inf
                best_prev_tag = None

                for prev_tag in self.tags:
                    prev_score = dp[i - 1][prev_tag]
                    features = self.extract_word_features(words, i-1, prev_tag)

                    # current score
                    score = prev_score + self.compute_feature_score(features, tag)

                    if score > max_score:
                        max_score = score
                        best_prev_tag = prev_tag

                dp[i][tag] = max_score
                backpointer[i][tag] = best_prev_tag

        dp[n + 1]["<s>"] = -math.inf  # initialize
        for tag in self.tags:
            prev_score = dp[n][tag]
            features = self.extract_word_features(words, n, tag)

            score = prev_score + self.compute_feature_score(features, '<s>')

            if score > dp[n + 1]["<s>"]:
                dp[n + 1]["<s>"] = score
                backpointer[n + 1]["<s>"] = tag

        best_last_tag = backpointer[n + 1]["<s>"]
        best_path = [best_last_tag]
        for i in range(n, 0, -1):
            best_last_tag = backpointer[i][best_last_tag]
            best_path.append(best_last_tag)
        best_path.reverse()
        # print(backpointer)
        return best_path[1:]
        # for i in range(1, n + 1):
        #     for tag in self.tags:
        #         max_score, best_prev_tag = max(
        #             (dp[i - 1][prev_tag] + self.compute_feature_score(self.extract_word_features(words, i - 1, prev_tag), tag), prev_tag)
        #             for prev_tag in dp[i - 1]
        #         )
        #         dp[i][tag] = max_score
        #         backpointer[i][tag] = best_prev_tag

        # best_last_tag = max(dp[n], key=dp[n].get)
        # best_path = [best_last_tag]

        # for i in range(n, 0, -1):
        #     best_last_tag = backpointer[i][best_last_tag]
        #     best_path.append(best_last_tag)

        # best_path.reverse()
        # print(f"Viterbi best path: {best_path[1:]}")
        # return best_path[1:]



    def evaluate(self, dev_sentences):
        total_tags = 0
        correct_tags = 0

        for sentence in dev_sentences:
            words, true_tags = sentence
            # prediction
            predicted_tags = self.viterbi(words)

            # calculate correct tags
            for pred, true in zip(predicted_tags, true_tags):
                if pred == true:
                    correct_tags += 1
                total_tags += 1

        accuracy = correct_tags / total_tags if total_tags > 0 else 0.0
        return accuracy

    def save_params(self, filepath):

            param_weights = defaultdict(dict)

            for tag, features in self.best_weights.items():
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
    dev_file = sys.argv[2]
    param_file = sys.argv[3]
    train_sentences = load_data(train_file)
    # print(train_sentences)
    dev_sentences = load_data(dev_file)
    print("train_sentences loaded")
    # print(train_sentences)

    print(f"Loaded {len(train_sentences)} training sentences:")
    print(train_sentences[:2])  # Print the first 2 sentences for verification

    dev_sentences = load_data(dev_file)
    print(f"Loaded {len(dev_sentences)} development sentences:")
    print(dev_sentences[:2])
    # initialize model
    sub_train_len = int(len(train_sentences) * 0.05)  # 10% of the training data
    sub_train_sentences = random.sample(train_sentences, sub_train_len)
    sub_dev_len = int(len(dev_sentences) * 0.1)  # 10% of the dev data
    sub_dev_sentences = random.sample(dev_sentences, sub_dev_len)
    model = LCCRFTagger(train_sentences)
    print(len(model.tags))
    # print(model.tags)
    # print(len(model.tags))
    model.train(sub_train_sentences, sub_dev_sentences, num_epochs=2, learning_rate=0.1)
    model.save_params(param_file)


    train_sentences = load_data(train_file)

