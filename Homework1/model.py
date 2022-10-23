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
        h = self.tanh(h)
        return x, h


class Seq2SeqDecoder(nn.Module):
    def __init__(self, embeddings, padding_idx=0, embedding_size=300, hidden_size=128, batch_first=True, use_attention=False):
        super(Seq2SeqDecoder, self).__init__()
        self.use_attention = use_attention
        self.output_size = len(embeddings.vocab)

        self.embed = nn.Embedding.from_pretrained(torch.tensor(embeddings.embeddings), padding_idx=padding_idx)
        self.gru = nn.GRU(embedding_size, hidden_size, batch_first=batch_first)

        if use_attention:
            self.attention = Attention(hidden_size)
            self.linear = nn.Linear(hidden_size * 3, self.output_size)
        else:
            self.linear = nn.Linear(hidden_size, self.output_size)

    def forward(self, x, hidden, encoder_outputs=None):
        if not self.use_attention:
            x = self.embed(x)
            x, hidden = self.gru(x, hidden)
            x = self.linear(x)
            return x, hidden

        else:
            predict = torch.empty(x.size(0), x.size(1), self.output_size, device=x.device)
            for i in range(x.size(1)):
                atten = self.attention(encoder_outputs, hidden.squeeze(0)).unsqueeze(1)
                w = torch.bmm(atten, encoder_outputs)
                o = self.embed(x[:, i:i + 1])
                o, hidden = self.gru(o, hidden)
                o = o.squeeze(1)
                w = w.squeeze(1)
                o = self.linear(torch.cat((o, w), 1))
                predict[:, i, :] = o

            return predict, hidden


class Seq2SeqModel(nn.Module):
    def __init__(self, embeddings, padding_idx=0, embedding_size=300, hidden_size=128, batch_first=True, use_attention=False):
        super(Seq2SeqModel, self).__init__()
        self.encoder = Seq2SeqEncoder(embeddings, padding_idx, embedding_size, hidden_size, batch_first)
        self.decoder = Seq2SeqDecoder(embeddings, padding_idx, embedding_size, hidden_size, batch_first, use_attention)

    def forward(self, x, y):
        encoder_outputs, context = self.encoder(x)
        context = context.unsqueeze(0)
        x, _ = self.decoder(y, context, encoder_outputs)
        return x


class Attention(nn.Module):
    def __init__(self, hidden_size=128):
        super(Attention, self).__init__()

        self.linear = nn.Linear(hidden_size * 2, hidden_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, key, query):
        key = self.linear(key)
        atten = torch.bmm(key, query.unsqueeze(2)).squeeze(2)
        atten = self.softmax(atten)
        return atten

