import torch.nn as nn
import numpy as np
import torch

class RNNModel(nn.Module):
    def __init__(self, hidden_dim=64, hidden_layer=1, embedding_layer=None, sequence_length=200, vocab_size=100000, embedding_dim=50, dropout=0.2, bidirectional=True):
        super().__init__()
        if embedding_layer is not None:
            if isinstance(embedding_layer, np.ndarray):
                self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(embedding_layer))
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        self.rnn = nn.LSTM(
            input_size=sequence_length,
            hidden_size=hidden_dim,
            num_layers=hidden_layer,
            dropout=dropout,
            bidirectional=bidirectional,
            batch_first=True
        )
        h_out = hidden_layer * hidden_dim
        if bidirectional:
            h_out *= 2
        self.classifier = nn.Linear(h_out, 5)
        self.loss = nn.CrossEntropyLoss()
        
    def forward(self, input_ids, label):
        self.embeds = self.embedding(input_ids)
        output, hidden = self.rnn(self.embeds)
        h_n, c_n = hidden
        h = torch.transpose(h_n, 0, 1).contiguous().view(input_ids.shape[0], -1)
        logits = self.classifier(h)
        loss = self.loss(logits, label)
        return logits, loss
    