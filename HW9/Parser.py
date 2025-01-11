import torch
import torch.nn as nn


class WordRepresentation(nn.Module):
    def __init__(self, num_symbols, embedding_dim, hidden_size, dropout_rate=0.5):
        super(WordRepresentation, self).__init__()
        self.embedding = nn.Embedding(num_symbols, embedding_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.forward_lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
        self.backward_lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
        self.hidden_dropout = nn.Dropout(dropout_rate)

    def forward(self, suffix_tensor, prefix_tensor):
        suffix_embedded = self.dropout(self.embedding(suffix_tensor))  # (batch_size, seq_len, hidden_size)
        prefix_embedded = self.dropout(self.embedding(prefix_tensor))
        _, (forward_hidden, _) = self.forward_lstm(suffix_embedded)  # forward_hidden: (1, L, hidden_dim)
        _, (backward_hidden, _) = self.backward_lstm(prefix_embedded)  # backward_hidden: (1, L, hidden_dim)

        forward_hidden = forward_hidden.squeeze(0)  # (N, hidden_dim)
        backward_hidden = backward_hidden.squeeze(0)  # (N, hidden_dim)

        # Dropout after LSTM hidden states
        forward_hidden = self.hidden_dropout(forward_hidden)
        backward_hidden = self.hidden_dropout(backward_hidden)

        word_representation = torch.cat((forward_hidden, backward_hidden), dim=1)  # (N, 2 * hidden_dim)

        return word_representation


class SpanRepresentation(nn.Module):
    def __init__(self, input_dim, hidden_size, lstm_depth=1, dropout_rate=0.5):
        super(SpanRepresentation, self).__init__()
        self.bilstm = nn.LSTM(
            input_dim,
            hidden_size,
            num_layers=lstm_depth,
            bidirectional=True,
            dropout=dropout_rate if lstm_depth > 1 else 0,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.hidden_size = hidden_size

    def forward(self, word_representations):
        seq_len, word_rep_size = word_representations.shape
        word_representations = word_representations.unsqueeze(0)
        dummy = torch.zeros((1, 1, word_rep_size), device=word_representations.device)
        padded_input = torch.cat([dummy, word_representations, dummy], dim=1)

        # BiLSTM
        bilstm_out, _ = self.bilstm(self.dropout(padded_input))  # (batch_size, seq_len + 2, 2 * hidden_size)

        forward = bilstm_out[:, :, :self.hidden_size]  #  (batch_size, seq_len + 2, hidden_size)
        backward = bilstm_out[:, :, self.hidden_size:]  #  (batch_size, seq_len + 2, hidden_size)

        # remove last timestep of forward and first timestep of backward
        forward = forward[:, :-1, :]  #  (batch_size, seq_len + 1, hidden_size)
        backward = backward[:, 1:, :]  #  (batch_size, seq_len + 1, hidden_size)
        forward = forward.squeeze(0)  # (N+1, hidden_dim)
        backward = backward.squeeze(0)
        positions = torch.arange(seq_len + 1, device=word_representations.device)  # (N+1)
        span_positions = torch.combinations(positions, r=2)
        start_positions = span_positions[:, 0]  # (M,)
        end_positions = span_positions[:, 1]
        forward_diff = forward[end_positions] - forward[start_positions]  # (M, hidden_dim)
        backward_diff = backward[start_positions] - backward[end_positions]
        span_representations = torch.cat([forward_diff, backward_diff], dim=-1)  # (M, 2 * hidden_dim)

        return span_representations



class SpanClassification(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_labels):
        super(SpanClassification, self).__init__()
        self.feedforward = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_labels)
        )
        self.num_categories = num_labels

    def forward(self, span_representations):
        """
        span_representations: (num_spans, 2 * hidden_size)
        """
        scores = self.feedforward(span_representations)
        # score ("keine Konstituente") = 0
        scores[:, 0] = 0
        return scores



class Parser(nn.Module):
    def __init__(self, num_symbols, embedding_dim, word_hidden_size, span_hidden_size, ff_hidden_dim, num_labels, span_lstm_depth=1, dropout_rate=0.5):
        super(Parser, self).__init__()
        self.word_representation = WordRepresentation(num_symbols, embedding_dim, word_hidden_size, dropout_rate)
        # 使用 SpanRepresentation 獲得 span 表示
        self.span_representation = SpanRepresentation(2 * word_hidden_size, span_hidden_size, span_lstm_depth, dropout_rate)
        # SpanClassification
        self.span_classification = SpanClassification(2 * span_hidden_size, ff_hidden_dim, num_labels)

    def forward(self, suffix_tensor, prefix_tensor):
        """
        suffix_emb: (seq_len, embedding_dim)
        prefix_emb: (seq_len, embedding_dim)
        """
        word_representations = self.word_representation(suffix_tensor, prefix_tensor)  # (batch_size, seq_len, 2 * word_hidden_size)

        # get span representations
        span_representations = self.span_representation(word_representations) #num_spans, 2 * span_hidden_size)

        # calculate scores
        scores = self.span_classification(span_representations)  # (num_spans, num_categories)
        return scores


if __name__ == "__main__":
    parser = Parser(num_symbols=26, embedding_dim=128, word_hidden_size=64, span_hidden_size=64, ff_hidden_dim=32, num_labels=20)
    prefix_tensor = torch.randint(0, 26, (8, 10))
    suffix_tensor = torch.randint(0, 26, (8, 10))
    span_scores = parser(suffix_tensor, prefix_tensor)
    print("Span Scores Shape:", span_scores.shape)


