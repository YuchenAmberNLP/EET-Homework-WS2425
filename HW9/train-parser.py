import sys
import torch
import random
import torch.nn as nn
import torch.optim as optim
from Data import Data, NoConstLabelID
from Parser import Parser



def train(path_train, path_dev, parfile, n_epochs=10, lr=0.001, clip=1.0):
    data = Data(path_train, path_dev)
    print("data loading successful")
    model = Parser(
        num_symbols=data.num_char_types(),
        embedding_dim=64,
        word_hidden_size=128,
        span_hidden_size=128,
        ff_hidden_dim=64,
        span_lstm_depth=2,
        dropout_rate=0.5,
        num_labels=data.num_label_types()
    )
    # model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    print("model loading successful")

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_dev_errors = float('inf')
    num_errors = []

    for epoch in range(n_epochs):
        print(f"Epoch {epoch + 1}/{n_epochs}")
        random.shuffle(data.train_parses)
        model.train()
        total_loss = 0

        for words, constituents in data.train_parses:
            fwd_charID_seqs, bwd_charID_seqs = data.words2charIDvec(words)
            fwd_charID_seqs = torch.tensor(fwd_charID_seqs, dtype=torch.long)
            bwd_charID_seqs = torch.tensor(bwd_charID_seqs, dtype=torch.long)

            num_words = len(words)
            num_spans = (num_words * (num_words + 1)) // 2
            labels = torch.full((num_spans,), NoConstLabelID, dtype=torch.long)

            span_positions = torch.combinations(torch.arange(num_words + 1), r=2)
            for label, start, end in constituents:
                span_index = torch.where((span_positions[:, 0] == start) & (span_positions[:, 1] == end))[0]
                labels[span_index] = data.labelID(label)

            # 转移到设备
            fwd_charID_seqs = fwd_charID_seqs.to(model.device)
            bwd_charID_seqs = bwd_charID_seqs.to(model.device)
            labels = labels.to(model.device)

            optimizer.zero_grad()
            span_scores = model(fwd_charID_seqs, bwd_charID_seqs)

            # 计算损失
            loss = criterion(span_scores, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

            optimizer.step()

            total_loss += loss.item()

        print(f"Training loss: {total_loss:.4f}")

        model.eval()
        dev_errors = 0
        with torch.no_grad():
            for words, constituents in data.dev_parses:
                fwd_charID_seqs, bwd_charID_seqs = data.words2charIDvec(words)
                fwd_charID_seqs = torch.tensor(fwd_charID_seqs, dtype=torch.long).to(model.device)
                bwd_charID_seqs = torch.tensor(bwd_charID_seqs, dtype=torch.long).to(model.device)
                span_scores = model(fwd_charID_seqs, bwd_charID_seqs)
                predicted_labels = span_scores.argmax(dim=-1)

                span_positions = torch.combinations(torch.arange(len(words) + 1), r=2)
                for label, start, end in constituents:
                    span_index = torch.where((span_positions[:, 0] == start) & (span_positions[:, 1] == end))[0]
                    if predicted_labels[span_index] != data.labelID(label):
                        dev_errors += 1

        print(f"Development errors: {dev_errors}")
        num_errors.append(dev_errors)

        # save best model
        if dev_errors < best_dev_errors:
            best_dev_errors = dev_errors
            torch.save(model.state_dict(), f"{parfile}.model")
            data.store_parameters(f"{parfile}.params")

    with open("num-errors.txt", "w") as file:
        file.write("\n".join(map(str, num_errors)))

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python train-parser.py train-parses dev-parses parfile")
        sys.exit(1)

    path_train = sys.argv[1]
    path_dev = sys.argv[2]
    parfile = sys.argv[3]
    train(path_train, path_dev, parfile)
