import torch
import torchtext
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import sys
from torch.utils.data import DataLoader, Dataset
import argparse
import torch.optim as optim


# train_data_path = sys.argv[1]
# dev_data_path = sys.argv[2]
# parfile_path = sys.argv[3]
# BATCH_SIZE = 2

def parse_args():
    parser = argparse.ArgumentParser(description="Train a model with hyperparameters")


    parser.add_argument('trainingdata', type=str, help="Path to the training data file")
    parser.add_argument('devdata', type=str, help="Path to the development data file")
    parser.add_argument('parfile', type=str, help="Path to the parfile")

    parser.add_argument('--embedding_dim', type=int, default=64, help="Embedding dim (default: 64)")
    parser.add_argument('--batch_size', type=int, default=4, help="Batch size for training (default: 4)")
    parser.add_argument('--learning_rate', type=float, default=0.05, help="Learning rate (default: 0.05)")
    parser.add_argument('--epochs', type=int, default=10, help="Number of epochs (default: 10)")
    parser.add_argument('--hidden_size', type=int, default=32, help="Number of hidden units (default: 32)")
    parser.add_argument('--rnn_size', type=int, default=256, help="Rnn size (default: 256)")
    parser.add_argument('--dropout', type=float, default=0.5, help="Dropout rate (default: 0.5)")

    # 解析命令行参数
    args = parser.parse_args()

    return args

args = parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_data(file_path):
    """
    Trainingsdaten (oder Development- Daten) aus einer Datei einliest und eine Liste vom Paaren zurueckgibt,
    wobei jedes Paar aus einem Klassen-Label (als Integer) und einer Liste von Tokens besteht.
    """
    df = pd.read_csv(file_path, sep='\t', header=None)
    df.columns = ['label', 'text']
    labels = df['label']
    texts = df['text']
    data = list(zip(labels, texts))
    return data


train_data = read_data(args.trainingdata)
dev_data = read_data(args.devdata)


def get_vocab(texts):
#vocab
    vocab = torchtext.vocab.build_vocab_from_iterator((text.split() for text in texts), specials=["<unk>", "<pad>"])
    vocab.set_default_index(vocab["<unk>"])
    return vocab


labels = set([label for label,_ in train_data])
LABEL_SIZE = len(labels)
train_texts = [text for _, text in train_data]
vocab = get_vocab(train_texts)
VOCAB_SIZE = len(vocab)

def collate(batch):
    """
    ein Batch von Datenpaaren als Argument erhaelt
    und die Wortfolgen mit Hilfe von vocab auf Zahlen abbildet
    und drei Tensoren mit den Labels, den Wort-IDs und den Textlaengen zurueckgibt.
    Die Wort-IDs werden mit der PyTorch-Methode pad sequence “gepaddet”.
    """
    labels, texts = zip(*batch)
    text_ids = [torch.tensor([vocab.get_stoi()[word] for word in text]) for text in texts]
    padded_text = pad_sequence(text_ids, batch_first=True, padding_value=vocab.get_stoi()['<pad>'])
    lengths = torch.tensor([len(text) for text in text_ids], dtype=torch.long) # original length
    labels = torch.tensor(labels, dtype=torch.long)
    return labels, padded_text, lengths


class TextDataset(Dataset):
    def __init__(self, data, vocab):
        self.data = data
        self.vocab = vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        label, text = self.data[idx]
        tokens = text.split()
        token_ids = [self.vocab.get_stoi()[word] for word in tokens]
        return torch.tensor(token_ids), label


train_set = TextDataset(train_data, vocab)
dev_set = TextDataset(dev_data, vocab)

# set train and dev DataLoader
train_loader = DataLoader(train_set, batch_size=args.batch_size, collate_fn=lambda batch: collate(batch, vocab))
dev_loader = DataLoader(dev_set, batch_size=args.batch_size, collate_fn=lambda batch: collate(batch, vocab))


class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, rnn_size, hidden_size, dropout, label_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=1)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=rnn_size, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(rnn_size * 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, label_size)
        self.relu = nn.ReLU()

    def forward(self, seq, lengths):
        x = self.dropout(self.embedding(seq))
        packed_x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False) # 注意params
        packed_out, _ = self.lstm(packed_x)
        rnn_out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True, padding_value=1)
        # mean pooling for rnn output
        lstm_out = torch.mean(rnn_out, dim=1)
        hidden = self.relu(self.fc1(lstm_out))
        hidden = self.dropout(hidden)
        out = self.fc2(hidden)
        return out



model = TextClassifier(
    VOCAB_SIZE,
    embedding_dim=args.embedding_dim,
    rnn_size=args.rnn_size,
    hidden_size=args.hidden_size,
    dropout=args.dropout,
    label_size=LABEL_SIZE).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)


def train(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch in train_loader:
        labels, sequences, lengths = batch
        sequences, labels = sequences.to(device), labels.to(device)
        optimizer.zero_grad()
        # train
        output = model(sequences, lengths)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        # calculate correct predictions
        _, predicted = torch.max(output, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    accuracy = correct / total
    avg_loss = total_loss / len(train_loader)

    return avg_loss, accuracy


def evaluate(model, dev_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dev_loader:
            labels, sequences, lengths = batch
            sequences, labels = sequences.to(device), labels.to(device)
            # evaluation
            output = model(sequences, lengths)
            loss = criterion(output, labels)
            total_loss += loss.item()
            # calculate correct predictions
            _, predicted = torch.max(output, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    avg_loss = total_loss / len(dev_loader)

    return avg_loss, accuracy

num_epochs = args.epochs

for epoch in range(1, num_epochs + 1):
    # train
    train_loss, train_accuracy = train(model, train_loader, optimizer, criterion, device)
    print(f"Epoch {epoch} - Training Loss: {train_loss}, Training Accuracy: {train_accuracy}")

    # evaluation
    dev_loss, dev_accuracy = evaluate(model, dev_loader, criterion, device)
    print(f"Epoch {epoch} - Dev Loss: {dev_loss}, Dev Accuracy: {dev_accuracy}")

# save model params
parfile = args.parfile
torch.save(model, parfile)

