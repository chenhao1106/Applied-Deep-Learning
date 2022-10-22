import torch
import torch.nn as nn

class ExtractiveModel(nn.Module):
    def __init__(self, embeddings, padding_idx=0, embedding_size=300, hidden_size=128, num_layers=2, batch_first=True, dropout=0.1):
        super(ExtractiveModel, self).__init__()

        self.embed = nn.Embedding.from_pretrained(torch.tensor(embeddings.embeddings), padding_idx=padding_idx)
        self.gru = nn.GRU(embedding_size, hidden_size, num_layers, batch_first=batch_first, dropout=dropout, bidirectional=True)
        self.linear = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):
        x = self.embed(x)
        x, _ = self.gru(x)
        x = self.linear(x)
        x = x.squeeze(2)
        return x

