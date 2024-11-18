#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Importieren der notwendigen Bibliotheken
import os
import json
import sys
import math
from collections import defaultdict
import string

# Funktion zur Tokenisierung des Textes (Trennung von Wörtern)
def text_tokenize(text):
    tokens = text.split()  # Teilt den Text in Wörter (Tokens) auf
    filtered_tokens = [word for word in tokens if word not in string.punctuation]  # Entfernt Satzzeichen
    return filtered_tokens

# Funktion zur Extraktion von Merkmalen (Wörtern) aus dem Text
def get_features(text):
    vector = defaultdict(float)  # Erstellt einen Standardwertvektor für die Merkmale
    for word in text_tokenize(text):  # Für jedes Wort im tokenisierten Text
        vector[word] += 1.0  # Erhöht den Wert des Merkmals für das Wort um 1
    return vector

# Klasse für den Log-Linear-Klassifikator
class LogLinearClassifier:
    def __init__(self, paramfile_path):
        self.classes = []  # Liste der möglichen Klassen
        self.weights = {}  # Gewichtung der Merkmale für jede Klasse
        self.load_params(paramfile_path)  # Lädt die Parameter aus der Datei

    # Funktion zum Laden der Parameter aus einer JSON-Datei
    def load_params(self, paramfile_path):
        with open(paramfile_path, 'r', encoding='utf-8') as f:
            params = json.load(f)  # Lädt die JSON-Daten aus der Datei
            self.classes = params["classes"]  # Setzt die Klassen aus den Parametern
            # Setzt die Gewichte für jede Klasse und jedes Wort
            self.weights = {c: defaultdict(float, params["weights"][c]) for c in self.classes}

    # Log-Summe-Exponential-Funktion zur Normalisierung der Scores
    def logsumexp(self, scores):
        max_score = max(scores)  # Bestimme das größte Score
        sum_exp = sum(math.exp(score - max_score) for score in scores)  # Berechne die Exponentialsumme
        return max_score + math.log(sum_exp)  # Logarithmierte Summe der Exponentialwerte

    # Funktion zur Vorhersage der Wahrscheinlichkeiten für jede Klasse
    def predict(self, features):
        scores = {}  # Dictionary für die Scores der Klassen
        for c in self.classes:  # Für jede Klasse
            scores[c] = sum(features[word] * self.weights[c][word] for word in features)  # Berechne den Score

        logZ = self.logsumexp(scores.values())  # Normalisiere die Scores mit logsumexp
        # Berechne die Wahrscheinlichkeiten für jede Klasse
        probs = {c: math.exp(scores[c] - logZ) for c in self.classes}
        return probs

    # Funktion zur Klassifikation eines Textes
    def classify(self, text):
        features = get_features(text)  # Extrahiere die Merkmale aus dem Text
        probs = self.predict(features)  # Berechne die Wahrscheinlichkeiten für jede Klasse
        predicted_class = max(probs, key=probs.get)  # Wähle die Klasse mit der höchsten Wahrscheinlichkeit
        return predicted_class

# Funktion zum Laden der E-Mail-Daten aus einem Verzeichnis
def load_mail_data(mail_dir):
    emails = []  # Liste zum Speichern der E-Mails
    filenames = []  # Liste zum Speichern der Dateinamen
    for subdir in os.listdir(mail_dir):  # Durchlaufe alle Unterverzeichnisse
        subdir_path = os.path.join(mail_dir, subdir)  # Bestimme den Pfad des Unterverzeichnisses
        if os.path.isdir(subdir_path):  # Wenn es sich um ein Verzeichnis handelt
            for filename in os.listdir(subdir_path):  # Durchlaufe alle Dateien im Unterverzeichnis
                if filename.startswith('.'):  # Ignoriere versteckte Dateien
                    continue
                file_path = os.path.join(subdir_path, filename)  # Bestimme den vollständigen Dateipfad
                if os.path.isfile(file_path):  # Wenn es sich um eine Datei handelt
                    # Lese die E-Mail-Datei und füge sie zur Liste hinzu
                    with open(file_path, 'r', encoding='ISO-8859-1') as f:
                        emails.append((subdir, filename, f.read()))
    return emails

# Hauptprogramm
if __name__ == "__main__":
    # Überprüfen, ob die richtigen Argumente übergeben wurden
    if len(sys.argv) != 3:
        print("Usage: python3 classify.py <paramfile> <mail-dir>")  # Ausgabe der korrekten Nutzung
        sys.exit(1)

    paramfile_name = sys.argv[1]  # Der Pfad zur Parameterdatei
    mail_dir = sys.argv[2]  # Der Pfad zum Verzeichnis mit den E-Mails

    # Erstelle einen Klassifikator und lade die Parameter
    classifier = LogLinearClassifier(paramfile_name)
    # Lade die E-Mails aus dem angegebenen Verzeichnis
    emails = load_mail_data(mail_dir)

    # Datei zum Speichern der Klassifikationsergebnisse
    results_file = "results.txt"
    with open(results_file, "w") as results:
        for subdir, filename, email in emails:  # Für jede geladene E-Mail
            predicted_class = classifier.classify(email)  # Bestimme die Klasse der E-Mail
            results.write(f"{subdir}/{filename}\t{predicted_class}\n")  # Speichere das Ergebnis in der Datei

