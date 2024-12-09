import sys
import os


current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
crf_train = __import__("crf-train")


def load_only_words_data(filepath):
    sentences = []
    with open(filepath, 'r', encoding='utf-8') as file:
        sentence = []
        for line in file:
            if line.strip():
                if '\t' in line:
                    word, tag = line.strip().split('\t')  # 分割單詞和標籤
                else:
                    word = line.strip()
                sentence.append(word)
            else:
                if sentence:
                    sentences.append(sentence)
                    sentence = []
        if sentence: # add the last sentence
            sentences.append(sentence)

    return sentences


def save_sentences_to_file(sentences, tags, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        for words, sentence_tags in zip(sentences, tags):
            for word, tag in zip(words, sentence_tags):
                file.write(f"{word}\t{tag}\n")  # 写入单词和标签
            file.write("\n")

def predict_tags_for_sentences(sentences):
    for sentence in test_sentences:
        tag = tagger.viterbi(sentence)
        res_tags.append(tag)
    return res_tags

paramfile = sys.argv[1]
test_file = sys.argv[2]
test_sentences = load_only_words_data(test_file)

sub_test_sentences = test_sentences[:10]
test_res_file = 'test_res.txt'

print(test_sentences[:3])
res_tags = []

tagger = crf_train.LCCRFTagger(load_paramfile='paramfile')

# tags = predict_tags_for_sentences(sub_test_sentences)
tags = predict_tags_for_sentences(test_sentences)

save_sentences_to_file(test_sentences, tags, test_file)
# save_sentences_to_file(test_sentences, tags, test_res_file)