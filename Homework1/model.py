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


class Seq2SeqEncoder(nn.Module):
    def __init__(self, embeddings, padding_idx=0, embedding_size=300, hidden_size=128, batch_first=True):
        super(Seq2SeqEncoder, self).__init__()

        self.embed = nn.Embedding.from_pretrained(torch.tensor(embeddings.embeddings), padding_idx=padding_idx)
        self.gru = nn.GRU(embedding_size, hidden_size, batch_first=batch_first, bidirectional=True)
        self.linear = nn.Linear(hidden_size * 2, hidden_size)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.embed(x)
        x, h = self.gru(x)
        h = torch.reshape(torch.permute(h[-2:, :, :], (1, 0, 2)), (x.size(0), -1))
        h = self.linear(h)
        h = self.tanh(x)
        return x, h


class Seq2SeqDecoder(nn.Module):
    def __init__(self, embeddings, padding_idx=0, embedding_size=300, hidden_size=128, batch_first=True):
        super(Seq2SeqDecoder, self).__init__()

        self.embed = nn.Embedding.from_pretrained(torch.tensor(embeddings.embeddings), padding_idx=padding_idx)
        self.gru = nn.GRU(embedding_size, hidden_size, batch_first=batch_first)
        self.linear = nn.Linear(hidden_size, len(embeddings.vocab))

    def forward(self, x, hidden):
        x = self.embed(x)
        x, hidden = self.gru(x, hidden)
        x = self.linear(x)
        return x, hidden


class Seq2SeqModel(nn.Module):
    def __init__(self, embeddings, padding_idx=0, embedding_size=300, hidden_size=128, batch_first=True):
        super(Seq2SeqModel, self).__init__()

        self.encoder = Seq2SeqEncoder(embeddings, padding_idx, embedding_size, hidden_size, batch_first)
        self.decoder = Seq2SeqDecoder(embeddings, padding_idx, embedding_size, hidden_size, batch_first)

    def forward(self, x, y):
        _, context = self.encoder(x)
        context = context.unsqueeze(0)
        x, _ = self.decoder(y, context)
        return x

