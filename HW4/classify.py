import os
import sys
import json
import string
from collections import defaultdict
import math

def text_tokenize(text):
    tokens = text.split() # split with space
    filtered_tokens = [word for word in tokens if word not in string.punctuation]
    return filtered_tokens

def load_params(paramfile): # Load the parameters from the paramfile
    with open(paramfile, 'r') as f:
        params = json.load(f)
    return params

def word_frq_email(file_path):
    # extract the word frequency from the email
    word_fq = defaultdict(int)
    with open(file_path, 'r', encoding='ISO-8859-1') as f:
        text = f.read()
        words = text_tokenize(text)
        for word in words:
            word_fq[word] += 1
    
    return word_fq

def classify_email(word_fq, params):
    word_freq_per_class = params['word_prob_given_class'] # p(w|c)
    prob_per_class = params['prob_per_class'] # p(c)
    log_prob_per_class = {} # log(p(c)) + sum(log(p(w|c)))
    
    for c, prob in prob_per_class.items():
        log_prob = math.log(prob) # log(p(c))

        for w, fq in word_fq.items():
            if w in word_freq_per_class[c]:
                log_prob += fq * math.log(word_freq_per_class[c][w]) # += log(p(w|c))
            else:
                # log_prob += math.log(1e-10) # Small probability for unknown word (Laplace smoothing)
                continue # Skip the unknown word
    
        log_prob_per_class[c] = log_prob

    argmax = max(log_prob_per_class, key=log_prob_per_class.get) # Get the class with the highest log-probability

    return argmax

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 classify.py <paramfile> <mail-dir>")
        sys.exit(1)

    paramfile = sys.argv[1] 
    mail_dir = sys.argv[2]
    params = load_params(paramfile)
    
    for dir_name in os.listdir(mail_dir): # Iterate over each subdirectory in the 'ham' and 'spam' directories
        sub_dir_name = os.path.join(mail_dir, dir_name)  # Path to the 'ham' and 'spam' directories
        if os.path.isdir(sub_dir_name):
        # Iterate over each file in the subdirectory
            for file in os.listdir(sub_dir_name):
                file_full_path = os.path.join(sub_dir_name, file) # Path to the file
                if os.path.isfile(file_full_path):
                # Get word frequency for the file
                    word_fq = word_frq_email(file_full_path)
                # Classify the email based on word frequencies
                    classification = classify_email(word_fq, params)
                # Output the filename and assigned class, separated by a tab
                    print(f"{file}\t{classification}")


# Aus der Programmausgabe haben wir berechnet, wie häufig die E-Mails in den beiden Ordnern als Spam oder Ham klassifiziert wurden.
# Mit dem Befehl: python3 classify.py ... | cut -f2 | sort | uniq -c erhalten wir folgendes Ergebnis:
### predicted: 1446 ham （actual: 1500 ham）
### predicted: 4554 spam （actual: 4500 spam）

### Accuracy = (1446 + 4500)/(1446 + 4554) = 5946/6000 = 0.991
