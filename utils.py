import random
import spacy


class Tokenizer:
    def __init__(self, vocab, lower=True):
        self.nlp = spacy.load('en_core_web_sm', disable=['tagger', 'parser', 'ner', 'lemmatizer', 'textcat'])  # Only using tokenizer.
        self.lower = lower  # turn the token into lowercase
        self.set_vocab(vocab)

    # Split a sentence string into a list of words.
    def tokenize(self, sentence):
        return [token.text for token in self.nlp(sentence)]
        
    # Transform the token to index.
    def token_to_idx(self, token):
        return self.dict.get(token.lower() if self.lower else token, self.unk_token_idx)

    # Convert a sentence into word indices.
    def encode(self, sentence):
        return [self.token_to_idx(token) for token in self.tokenize(sentence)]

    # Translate word indices into a sentence.
    def decode(self, indices):
        return ''.join([self.vocab[idx] for idx in indices])

    # Turn a documents into a list of words.
    def collect_words(self, documents):
        return [token.text.lower() if self.lower else token.text for token in self.nlp.pipe(documents)]

    # Set up vocabulary and embedding table.
    def set_vocab(self, vocab, pad_token='<pad>', bos_token='<s>',
                  eos_token='</s>', unk_token='<unk>'):
        self.vocab = vocab
        self.dict = {word: idx for idx, word in enumerate(vocab)}
        self.pad_token = pad_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.unk_token = unk_token
        self.pad_token_idx = self.dict[pad_token]
        self.bos_token_idx = self.dict[bos_token]
        self.eos_token_idx = self.dict[eos_token]
        self.unk_token_idx = self.dict[unk_token]
    

class Embeddings:
    def __init__(self, embeddings_file, seed=524,
                 special_tokens=['<pad>', '<s>', '</s>', '<unk>']):
        self.vocab = special_tokens  # vocabulary list
        self.embeddings = [[] for _ in special_tokens]  # embedding of each word
        
        with open(embeddings_file, 'r') as f:
            data = f.readlines()
        # Set up vocabulary and their embeddings
        for d in data:
            d = d.rstrip().split(' ')
            word = d[0]
            embed = [float(v) for v in d[1:]]
            if word in special_tokens:
                self.embeddings[special_tokens.index(word)] = embed
            else:
                self.vocab.append(word)
                self.embeddings.append(embed)

        # Random generate embeddings if the special token is not in the embedding file
        random.seed(seed)
        for i in range(len(special_tokens)):
            if len(self.embeddings[i]) == 0:
                dim = len(self.embeddings[-1])
                self.embeddings[i] = [random.random() * 2 - 1 for _ in range(dim)]


# Padding setences to the certain length.
def pad_to_len(seqs, to_len, padding=0):
    paddeds = []
    for seq in seqs:
        paddeds.append(
                seq[:to_len] + [padding] * max(0, to_len - len(seq))
        )
    return paddeds
