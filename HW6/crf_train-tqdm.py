import sys
import math
from collections import defaultdict
import json
import random
from tqdm import tqdm
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


# class WordTagSentences:
#     def __init__(self, filepath, data_size=None):
#         sentences = []
#         with open(filepath, 'r', encoding='utf-8') as file:
#             sentence = []
#             for line in file:
#                 if line.strip():
#                     word, tag = line.strip().split('\t')  # 分割單詞和標籤
#                     sentence.append((word, tag))
#                 else:
#                     if sentence:
#                         sentences.append(sentence)
#                         sentence = []
#             if sentence:
#                 sentences.append(sentence)
#         self.tags = None
#         if data_size:
#             self.sentences = sentences[:data_size]
#         else:
#             self.sentences = sentences
#         # self.word_tag_probabilities = defaultdict(dict)
#         # self.suffix_2_probabilities = defaultdict(dict)
#         # self.suffix_3_probabilities = defaultdict(dict)
#         # self.suffix_4_probabilities = defaultdict(dict)
#         # self.capitalized_probabilities = defaultdict(dict)
#         # self.tag_transition_probabilities = defaultdict(dict)
#         self.features = defaultdict(dict)


class LCCRFTagger:
    def __init__(self, sentences, learning_rate=0.1, num_epochs=2):
        self.sentences = sentences
        tags = set()
        for sentence in self.sentences:
            for _, tag in sentence:
                tags.add(tag)
        tags.add('<s>')# 添加标签到集合中
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
            # 'is_first': i == 0, #if the word is a first word
            # 'is_last': i == len(sentence) - 1,  #if the word is a last word
            'is_capitalized': word[0].upper() == word[0],
            # 'is_all_caps': word.upper() == word,      #word is in uppercase
            # 'is_all_lower': word.lower() == word,      #word is in lowercase
            #suffix of the word
            'suffix_2': word[-2:] if len(word) > 1 else word,
            # 'suffix_3': word[-3:] if len(word) > 2 else word,
            'suffix_4': word[-4:] if len(word) > 3 else word
        }
        #extracting previous word
        if prev_tag is not None:
            # 上一个标签作为上下文特征
            features["prev_tag"] = prev_tag
        else:
            # 特别处理句子开头，添加特殊标志 </s>
            features["prev_tag"] = "<s>"
        return features


    def compute_feature_score(self, word_features, tag):
        """
        动态计算特征分数
        :param word: 当前单词
        :param tag: 当前标签
        :param prev_tag: 前一个标签
        :param features: 特征字典
        :return: 特征分数（一个浮点数）
        """
        score = 0.0

        # 遍历特征字典，计算当前标签和前一个标签的总分数
        for feature_name, feature_value in word_features.items():
            feature_key = f"{feature_name}={feature_value}"

            # 从权重表中查找特征对应的权重
            if tag in self.weights and feature_key in self.weights[tag]:
                score += self.weights[tag][feature_key]

        return score

    def forward(self, sentence):
        """
        前向算法计算 alpha 值
        :param sentence: 一个句子（单词序列）
        :return: 前向表 alpha 和句子的归一化因子
        """
        n = len(sentence)
        alpha = [{} for _ in range(n)]  # 每个位置都有一个字典存储各个标签的前向分数

        # 初始化第一个单词
        for tag in self.tags:
            word_features = self.extract_word_features(sentence, 0, None)
            alpha[0][tag] = self.compute_feature_score(word_features, tag)  # <s> 表示句首标签

        # 递归计算
        for i in range(1, n):
            for tag in self.tags:
                scores = [
                    alpha[i - 1][prev_tag] + self.compute_feature_score(self.extract_word_features(sentence, i, prev_tag), tag)
                    for prev_tag in self.tags
                ]
                alpha[i][tag] = self.log_sum_exp(scores)

        # 计算归一化因子 Z(w_1^n) 使用 log-sum-exp
        scores = [alpha[n - 1][tag] for tag in self.tags]
        z = self.log_sum_exp(scores)
        return alpha, z

    def backward(self, sentence):
        """
        后向算法计算 beta 值
        :param sentence: 一个句子（单词序列）
        :return: 后向表 beta 和句子的归一化因子
        """
        n = len(sentence)
        beta = [{} for _ in range(n)]  # 每个位置都有一个字典存储各个标签的后向分数

        # 初始化最后一个单词
        for tag in self.tags:
            beta[n - 1][tag] = 0  # log(1) = 0

        # 递归计算
        for i in range(n - 2, -1, -1):
            for tag in self.tags:
                word_features = self.extract_word_features(sentence, i, tag)
                scores = [
                    beta[i + 1][next_tag] + self.compute_feature_score(word_features, next_tag)
                    for next_tag in self.tags
                ]
                beta[i][tag] = self.log_sum_exp(scores)

        # 计算归一化因子 Z(w_1^n) 使用 log-sum-exp
        word_features = self.extract_word_features(sentence, 0, None)
        scores = [
            beta[0][tag] + self.compute_feature_score(word_features, tag)
            for tag in self.tags
        ]
        z = self.log_sum_exp(scores)
        return beta, z



    def compute_gradient(self, sentence, alpha, beta, z_log):
        """
        使用 log_sum_exp 计算梯度
        :param sentence: 输入句子，格式为 [(word, tag), (word, tag), ...]
        :param alpha: 前向概率表（对数形式）
        :param beta: 后向概率表（对数形式）
        :param z_log: 对数归一化因子 log(Z(X))
        :return: 梯度字典 {tag: {feature_key: gradient_value}}
        """
        gradient = defaultdict(lambda: defaultdict(float))  # 存储每个特征对每个标签的梯度
        n = len(sentence)

        # 提取真实标签序列
        true_tags = [tag for _, tag in sentence]
        gamma = [{} for _ in range(n)]  # 存储每个位置和标签的 posterior 概率

        # Compute γt(i) = αt(i) * βt(i) for all positions and tags
        for i in range(n):
            for tag in self.tags:
                gamma[i][tag] = alpha[i].get(tag, 0) + beta[i].get(tag, 0) - z_log  # 计算对数后验概率
            # 对每个位置归一化 gamma[i]，确保其和为 1
            total_gamma = sum(math.exp(gamma[i][tag]) for tag in self.tags)  # 计算归一化因子
            for tag in self.tags:
                gamma[i][tag] -= math.log(total_gamma) if total_gamma != 0 else 0

        # 观测特征期望
        for i in range(n):
            word = sentence[i][0]  # 取出当前单词
            prev_tag = true_tags[i - 1] if i > 0 else "<s>"  # 前一个真实标签，句首用 <s>
            current_tag = true_tags[i]  # 当前真实标签

            word_features = self.extract_word_features(sentence, i, prev_tag)
            for feature_name, feature_value in word_features.items():
                feature_key = f"{feature_name}={feature_value}"
                gradient[current_tag][feature_key] += 1  # 累加观测特征

        # 模型特征期望
        for i in range(n):
            word = sentence[i][0]  # 取出当前单词
            word_features = self.extract_word_features(sentence, i, None)  # 当前单词的特征

            for t in self.tags:  # 当前标签
                for t_prev in self.tags:  # 前一个标签
                    # 计算边缘概率的对数值
                    log_marginal_prob = (
                            (alpha[i - 1][t_prev] if i > 0 else 0)  # 前向概率，句首为 0
                            + self.compute_feature_score(word_features, t)  # 当前分数
                            + beta[i][t]  # 后向概率
                            - z_log  # 对数归一化因子
                    )
                    # print(log_marginal_prob)
                    marginal_prob = numpy.exp(log_marginal_prob)  # 转回概率值

                    # 更新模型特征期望
                    for feature_name, feature_value in word_features.items():
                        feature_key = f"{feature_name}={feature_value}"
                        # Subtract the expected counts from the model
                        gradient[t][feature_key] -= math.exp(gamma[i].get(t, 0)) * marginal_prob  # 在对数空间中更新

        return gradient

    def update_weights(self, gradient):
        """
        更新权重
        :param gradient: 梯度字典 {tag: {feature_key: gradient_value}}
        """
        for tag, feature_gradients in gradient.items():
            for feature_key, grad_value in feature_gradients.items():
                # 如果权重不存在，初始化为 0
                if feature_key not in self.weights[tag]:
                    self.weights[tag][feature_key] = 0.0

                # 按梯度更新权重
                self.weights[tag][feature_key] += self.learning_rate * grad_value


    def train(self):
        """
        训练 LC-CRF 模型
        :param sentences: 训练数据，每个句子是 [(word, tag), (word, tag), ...] 的格式
        """
        # 初始化权重
        sentences = self.sentences
        self.weights = defaultdict(lambda: defaultdict(float))

        for epoch in range(self.num_epochs):
            random.shuffle(sentences)  # 随机打乱训练数据

            # 包装内层循环，显示句子处理进度
            with tqdm(total=len(sentences), desc=f"Epoch {epoch + 1}/{self.num_epochs}") as pbar:
                for sentence in sentences:
                    # 1. 计算前向和后向概率
                    alpha, z_log_a = self.forward(sentence)
                    beta, z_log_b = self.backward(sentence)
                    # assert math.isclose(z_log_a, z_log_b, rel_tol=1e-6), "Forward and Backward results do not match!"
                    # print(z_log_a, z_log_b)
                    # 2. 计算梯度
                    gradient = self.compute_gradient(sentence, alpha, beta, z_log_a)

                    # 3. 更新权重
                    self.update_weights(gradient)

                    pbar.update(1)

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
    sub_train_len = int(len(train_sentences) * 0.1)  # 计算前 10% 的样本数量
    sub_train_sentences = random.sample(train_sentences, sub_train_len)
    model = LCCRFTagger(sub_train_sentences, num_epochs=2)
    print(len(model.tags))
    # print(model.tags)
    # print(len(model.tags))
    model.train()
    model.save_params(param_file)
