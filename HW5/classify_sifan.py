import json
import string
from collections import defaultdict
import math
import sys
import os

def text_tokenize(text):
    tokens = text.split()
    filtered_tokens = [word for word in tokens if word not in string.punctuation]
    # print(tokens)
    return filtered_tokens

def load_params(paramfile): # Load the parameters from the paramfile
    with open(paramfile, 'r') as f:
        params = json.load(f)
    return params

def word_frq_email(file_path):
    # extract the word frequency from the email
    word_fq = defaultdict(int) # {word1: freq1, word2: freq2, ...}
    with open(file_path, 'r', encoding='ISO-8859-1') as f:
        text = f.read()
        words = text_tokenize(text)
        for word in words:
            word_fq[word] += 1
    
    return word_fq

def classify_email(file_path, params):
    # classify the email
    word_fq = word_frq_email(file_path)
    scores = {c: 0.0 for c in params['classes']} # initialize the score for each class, {class1: score1, class2: score2, ...}
    for c in params['classes']:
        for word, freq in word_fq.items():
            if word in params['weights'][c]: # if the word is in the class
                scores[c] += freq * params['weights'][c][word]
    max_score = max(scores.values()) # get the maximum score of all classes
    for c in params['classes']:
        scores[c] = math.exp(scores[c] - max_score) # calculate the probability of the class, {class1: prob1, class2: prob2, ...}
    Z_sum_exp = sum(scores.values()) # Normalisierungsterm, sum of all probabilities
    probs = {c: scores[c] / Z_sum_exp for c in params['classes']} # normalize the probabilities, {class1: prob1, class2: prob2, ...}
    return max(probs, key=probs.get) # return the class with the highest probability


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 classify.py <paramfile> <mail-dir>")
        sys.exit(1)
    
    paramfile = sys.argv[1] 
    mail_dir = sys.argv[2]
    params = load_params(paramfile)

    # ham = 0
    # spam = 0
    # tp_fn = 0

    for dir_name in os.listdir(mail_dir): # iterate all directories in the mail_dir
        sub_dir_name = os.path.join(mail_dir, dir_name) # join all pathes of the directory together
        if os.path.isdir(sub_dir_name):
            for file in os.listdir(sub_dir_name):
                file_path = os.path.join(sub_dir_name, file) # join all pathes of the file together
                if os.path.isfile(file_path):
                    # word_fq = word_frq_email(file_path) # Get word frequency for the file
                    probs = classify_email(file_path, params) # Classify the email based on word frequencies
                    # if probs == 'ham':
                    #     ham += 1
                    # else:
                    #     spam += 1
                    # if dir_name == probs:
                    #     tp_fn += 1
                    # print(f"{file_path}\t{probs}")
                    # print(f"TP+FN: {tp_fn}")

                    with open('results.txt', 'a') as f:
                        f.write(f"{file_path}\t{probs}\n")
                        # print(f"Ham: {ham}, Spam: {spam}")


### Test the classify.py
# predicted Ham: 1507 (actual: 1500) 
# predicted Spam: 4493 (actual: 4500)
# Correct predicted in total: 5869 (total: 6000)
# Accuracy: 5869 / (1500+4500)*100% = 97.82%

