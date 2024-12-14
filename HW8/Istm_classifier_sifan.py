import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import argparse


# 1. Read Data Function
def read_data(file_path):
    """
    Reads data from a file and returns a list of (label, tokens) pairs.
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            label, text = line.strip().split('\t')
            tokens = text.split()
            data.append((int(label), tokens))
    return data

# 2. Build Vocabulary
def build_vocab(data):
    """
    Builds a vocabulary from the tokenized data using a generator expression.
    """
    texts = (tokens for _, tokens in data)
    vocab = {"<unk>": 0, "<pad>": 1}
    idx = 2
    for tokens in texts:
        for token in tokens:
            if token not in vocab:
                vocab[token] = idx
                idx += 1
    return vocab

# 3. Collate Function
def collate(batch, vocab):
    """
    Collates a batch of data by converting tokens to IDs and padding sequences.
    Returns labels, padded sequences, and lengths.
    """
    labels, texts = zip(*batch)
    text_ids = [torch.tensor([vocab.get(token, vocab["<unk>"]) for token in tokens], dtype=torch.long) for tokens in texts]
    lengths = torch.tensor([len(seq) for seq in text_ids], dtype=torch.long)
    
    # Sort by length in descending order
    lengths, sorted_indices = lengths.sort(descending=True)
    text_ids = [text_ids[i] for i in sorted_indices]
    labels = torch.tensor([labels[i] for i in sorted_indices], dtype=torch.long)

    padded_texts = pad_sequence(text_ids, padding_value=vocab["<pad>"], batch_first=True)
    return labels.clone().detach(), padded_texts, lengths

# 4. LSTM Classifier Model
class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, pad_idx):
        super(TextClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.dropout = nn.Dropout(0.5)  # Added dropout layer
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, text, lengths):
        embedded = self.embedding(text)
        embedded = self.dropout(embedded)  # Apply dropout
        
        # Pack the padded sequence
        packed_embedded = pack_padded_sequence(embedded, lengths.cpu(), enforce_sorted=True, batch_first=True)
        packed_output, (hidden, _) = self.lstm(packed_embedded)
        
        # Unpack the packed sequence
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        
        # Apply max pooling over the sequence length dimension
        pooled_output, _ = torch.min(output, dim=1)
        
        output = self.fc(pooled_output)
        return output  # Return raw logits

# 5. Train Function
def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for labels, texts, lengths in dataloader:
        labels, texts, lengths = labels.to(device), texts.to(device), lengths.to(device)
        optimizer.zero_grad()
        predictions = model(texts, lengths)
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (predictions.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    train_accuracy = correct / total
    return total_loss / len(dataloader), train_accuracy

# 6. Evaluate Function
def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for labels, texts, lengths in dataloader:
            labels, texts, lengths = labels.to(device), texts.to(device), lengths.to(device)
            predictions = model(texts, lengths)
            predicted_labels = predictions.argmax(dim=1)
            correct += (predicted_labels == labels).sum().item()
            total += labels.size(0)
    return correct / total

# 7. Main Function
def main(train_file, dev_file, parfile, embed_dim=128, hidden_dim=128, batch_size=32, epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load data
    train_data = read_data(train_file)
    dev_data = read_data(dev_file)
    
    # Get number of unique labels
    unique_labels = set(label for label, _ in train_data)
    output_dim = len(unique_labels)
    
    # Build vocabulary
    vocab = build_vocab(train_data)
    pad_idx = vocab["<pad>"]
    
    # Create DataLoaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, 
                              collate_fn=lambda batch: collate(batch, vocab))
    dev_loader = DataLoader(dev_data, batch_size=batch_size, shuffle=False, 
                            collate_fn=lambda batch: collate(batch, vocab))
    
    # Initialize model
    model = TextClassifier(len(vocab), embed_dim, hidden_dim, output_dim=output_dim, pad_idx=pad_idx).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Train and evaluate
    for epoch in range(epochs):
        train_loss, train_accuracy = train(model, train_loader, optimizer, criterion, device)
        dev_accuracy = evaluate(model, dev_loader, device)
        print(f"Epoch {epoch+1}/{epochs}: Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Dev Accuracy: {dev_accuracy:.4f}")
    
    # Save model
    torch.save(model.state_dict(), parfile)
    print(f"Model saved to {parfile}")

def parse_args():
    parser = argparse.ArgumentParser(description="Train an LSTM sentiment classifier.")
    parser.add_argument("train_file", type=str, help="Path to the training data file.")
    parser.add_argument("dev_file", type=str, help="Path to the development data file.")
    parser.add_argument("parfile", type=str, help="Path to save the trained model.")
    parser.add_argument("--embed_dim", type=int, default=128, help="Embedding dimension size (default: 128).")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden dimension size (default: 128).")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (default: 32).")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs (default: 5).")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    main(
        train_file=args.train_file,
        dev_file=args.dev_file,
        parfile=args.parfile,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,
        epochs=args.epochs
    )


# if __name__ == "__main__":
#     import sys
#     if len(sys.argv) != 4:
#         print("Usage: python lstm_classifier.py <train_file> <dev_file> <parfile>")
#         sys.exit(1)
    
#     train_file = sys.argv[1]
#     dev_file = sys.argv[2]
#     parfile = sys.argv[3]
    
#     main(train_file, dev_file, parfile)


### Example Usage:
# --embed_dim 512 --hidden_dim 512 --batch_size 16 --epochs 10
# Epoch 1/10: Train Loss: 1.5358, Train Accuracy: 0.3038, Dev Accuracy: 0.3606
# Epoch 2/10: Train Loss: 1.3854, Train Accuracy: 0.4032, Dev Accuracy: 0.3388
# Epoch 3/10: Train Loss: 1.2208, Train Accuracy: 0.4857, Dev Accuracy: 0.3624
# Epoch 4/10: Train Loss: 1.0546, Train Accuracy: 0.5616, Dev Accuracy: 0.3878
# Epoch 5/10: Train Loss: 0.8983, Train Accuracy: 0.6387, Dev Accuracy: 0.3733
# Epoch 6/10: Train Loss: 0.7408, Train Accuracy: 0.7074, Dev Accuracy: 0.3797
# Epoch 7/10: Train Loss: 0.6210, Train Accuracy: 0.7601, Dev Accuracy: 0.3815
# Epoch 8/10: Train Loss: 0.5053, Train Accuracy: 0.8089, Dev Accuracy: 0.3669
# Epoch 9/10: Train Loss: 0.4276, Train Accuracy: 0.8438, Dev Accuracy: 0.3615
# Epoch 10/10: Train Loss: 0.3670, Train Accuracy: 0.8665, Dev Accuracy: 0.3787
# Model saved to parfile

### Final train accuracy: 0.8665, Final dev accuracy: 0.3787
