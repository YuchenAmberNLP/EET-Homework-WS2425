import argparse
import sys
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import torch.optim as optim

# Funktion zum Einlesen der Daten
def read_data(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            label, text = line.strip().split('\t')
            tokens = text.split()
            data.append((int(label), tokens))
    return data

# Vokabular aufbauen
def build_vocab(data):
    texts = (tokens for _, tokens in data)
    vocab = build_vocab_from_iterator(texts, specials=["<unk>", "<pad>"])
    vocab.set_default_index(vocab["<unk>"])
    return vocab

# Collate-Funktion
def collate(batch, vocab):
    labels, sequences = zip(*batch)
    text_lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.long)
    word_ids = [torch.tensor([vocab[token] for token in seq], dtype=torch.long) for seq in sequences]
    padded_sequences = pad_sequence(word_ids, batch_first=True, padding_value=vocab["<pad>"])
    labels = torch.tensor(labels, dtype=torch.long)
    return labels, padded_sequences, text_lengths

# Dataset-Klasse
class TextDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# LSTM-Klassifikator-Klasse
class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_classes, dropout_rate):
        super(TextClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, text, text_lengths):
        embedded = self.dropout(self.embedding(text))
        packed_input = pack_padded_sequence(embedded, text_lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, (hidden, _) = self.lstm(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        pooled_output = output.mean(dim=1)
        logits = self.fc(self.dropout(pooled_output))
        return logits

    def save_params(self, file_path):
        torch.save(self.state_dict(), file_path)

    def load_params(self, file_path):
        self.load_state_dict(torch.load(file_path))

# Trainingsfunktion
def train(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for labels, texts, lengths in dataloader:
        labels, texts, lengths = labels.to(device), texts.to(device), lengths.to(device)
        optimizer.zero_grad()
        outputs = model(texts, lengths)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, predictions = torch.max(outputs, 1)
        correct += (predictions == labels).sum().item()
        total+=labels.size(0)
        
    accuracy = correct / total


    return total_loss / len(dataloader), accuracy

# Evaluierungsfunktion
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for labels, texts, lengths in dataloader:
            labels, texts, lengths = labels.to(device), texts.to(device), lengths.to(device)
            outputs = model(texts, lengths)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
    accuracy = correct / len(dataloader.dataset)
    return total_loss / len(dataloader), accuracy

# Hauptprogramm
if __name__ == "__main__":
    train_file = "/content/drive/MyDrive/sentiment.train.tsv"
    dev_file = "/content/drive/MyDrive/sentiment.dev.tsv"
    param_file = "/content/drive/MyDrive/paramfile"

    print("Loading data...")
    train_data = read_data(train_file)
    dev_data = read_data(dev_file)

    print("Building vocabulary...")
    vocab = build_vocab(train_data)

    train_dataset = TextDataset(train_data)
    dev_dataset = TextDataset(dev_data)

    collate_fn = lambda batch: collate(batch, vocab)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Initializing model...")
    model = TextClassifier(len(vocab), embed_size=256, hidden_size=256, num_classes=5, dropout_rate=0.5).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("Starting training...")
    for epoch in range(10):  # Number of epochs fixed for demonstration
        train_loss, train_accuracy = train(model, train_loader, criterion, optimizer, device)
        dev_loss, dev_accuracy = evaluate(model, dev_loader, criterion, device)
        print(f'Epoch {epoch + 1}: Train Loss = {train_loss:.4f}, Train Accuracy = {train_accuracy:.4f},  Dev Loss = {dev_loss:.4f}, Dev Accuracy = {dev_accuracy:.4f}')

    print("Saving model parameters...")
    model.save_params(param_file)

    print("Training complete.")
  
"""
Epoch 1: Train Loss = 1.5584, Train Accuracy=0.2944,  Dev Loss = 1.5354, Dev Accuracy = 0.3043
Epoch 2: Train Loss = 1.4878, Train Accuracy=0.3471,  Dev Loss = 1.5614, Dev Accuracy = 0.3061
Epoch 3: Train Loss = 1.4149, Train Accuracy=0.3851,  Dev Loss = 1.5062, Dev Accuracy = 0.3470
Epoch 4: Train Loss = 1.3200, Train Accuracy=0.4377,  Dev Loss = 1.5736, Dev Accuracy = 0.3252
Epoch 5: Train Loss = 1.2310, Train Accuracy=0.4830,  Dev Loss = 1.6577, Dev Accuracy = 0.3324
Epoch 6: Train Loss = 1.1442, Train Accuracy=0.5222,  Dev Loss = 1.5453, Dev Accuracy = 0.3551
Epoch 7: Train Loss = 1.0566, Train Accuracy=0.5652,  Dev Loss = 1.6711, Dev Accuracy = 0.3297
Epoch 8: Train Loss = 0.9829, Train Accuracy=0.5988,  Dev Loss = 1.7747, Dev Accuracy = 0.3588
Epoch 9: Train Loss = 0.9069, Train Accuracy=0.6310,  Dev Loss = 1.9525, Dev Accuracy = 0.3642
Epoch 10: Train Loss = 0.8338, Train Accuracy=0.6625,  Dev Loss = 2.1013, Dev Accuracy = 0.3406
"""

"""
# Hauptprogramm
This part needs to be checked: 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('train_data', help='Path to training data')
    parser.add_argument('dev_data', help='Path to development data')
    parser.add_argument('--embed_size', type=int, default=128, help='Size of embedding layer')
    parser.add_argument('--hidden_size', type=int, default=128, help='Size of LSTM hidden layer')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    args = parser.parse_args()

    train_data = read_data(args.train_data)
    dev_data = read_data(args.dev_data)

    vocab = build_vocab(train_data)

    train_dataset = TextDataset(train_data)
    dev_dataset = TextDataset(dev_data)

    collate_fn = lambda batch: collate(batch, vocab)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = TextClassifier(len(vocab), args.embed_size, args.hidden_size, 5, args.dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        dev_loss, dev_accuracy = evaluate(model, dev_loader, criterion, device)
        print(f'Epoch {epoch + 1}: Train Loss = {train_loss:.4f}, Dev Loss = {dev_loss:.4f}, Dev Accuracy = {dev_accuracy:.4f}')

if __name__ == '__main__':
    main()
"""
