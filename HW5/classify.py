from train import LogLinearClassifier
import json
from collections import defaultdict
import sys
import os


def load_params(paramfile_path):
    with open(paramfile_path, 'r', encoding='utf-8') as f:
        params = json.load(f)
    classes = params["classes"]
    weights = {c: defaultdict(float, params["weights"][c]) for c in classes}
    classifier = LogLinearClassifier(classes)
    classifier.weights = weights
    return classifier


def classify(features, classifier):
    probs = classifier.predict(features)
    # print(probs)
    predicted_class = max(probs, key=probs.get)
    return predicted_class


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 classify.py <paramfile> <mail-dir>")
        sys.exit(1)

    paramfile = sys.argv[1]
    mail_dir = sys.argv[2]

    classifier = load_params(paramfile)
    # total = 0
    # correct = 0
    results_file = "results.txt"
    with open(results_file, "w") as results:
        for class_name in os.listdir(mail_dir):
            class_dir = os.path.join(mail_dir, class_name)
            if os.path.isdir(class_dir):
                for filename in os.listdir(class_dir):
                    file_path = os.path.join(class_dir, filename)
                    with open(file_path, 'r', encoding='ISO-8859-1') as f:
                        # total += 1
                        text = f.read()
                        features = classifier.get_features(text)
                        prediction = classify(features, classifier)
                        # if prediction == class_name:
                            # correct += 1
                        results.write(f"{mail_dir}/{class_name}/{filename}\t{prediction}\n")


    print("results are saved")
    # print(f"total{total}, correct{correct}, acc{correct/total}")

# total: 6000, correct prediction: 5876
# Accuracy: 0.9793333333333333




