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
        for i, (words, true_tags) in enumerate(sentences):
            pred_tags = tags[i]
            for word, tag in zip(words, pred_tags):
                file.write(f"{word}\t{tag}\n")
            file.write("\n")


def predict_tags_for_sentences(sentences):
    res_tags = []
    for words, _ in sentences:
        tag = tagger.viterbi(words)
        res_tags.append(tag)
    # print(res_tags)
    return res_tags


paramfile = sys.argv[1]
test_file = sys.argv[2]
test_sentences = crf_train.load_data(test_file)

test_res_file = 'test_annotation.txt'

tagger = crf_train.LCCRFTagger(load_paramfile=paramfile)

tags = predict_tags_for_sentences(test_sentences)
accuracy = tagger.evaluate(test_sentences)
print("Accuracy on test file:", accuracy)

save_sentences_to_file(test_sentences, tags, test_res_file)
