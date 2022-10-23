import argparse
import pickle
import json
from pathlib import Path
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from model import Seq2SeqModel
from utils import Embeddings

def main(args):
    # Set up test data loader.
    loader = load_dataloader('./data/seq2seq_test_dataset.pkl', args.batch_size, False,
                                   args.num_workers)

    # Load embeddings.
    with open('./embedding.pkl', 'rb') as f:
        embeddings = pickle.load(f)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Set up model.
    model = Seq2SeqModel(embeddings, use_attention=args.use_attention)
    model.load_state_dict(torch.load('./seq2seq{"-attention" if args.use_attention else ""}-model.pt', map_location=device))
    model.to(device)
    model.eval()

    with open(args.output_file, 'w') as f:
        for batch in tqdm(loader):
            x = batch['text'].to(device)
            identifier = batch['id']

            prediction = []
            encoder_outputs, hidden = model.encoder(x)
            hidden = hidden.unsqueeze(0)  # Inference context vector.
            output = torch.tensor([[1]], device=device).repeat(x.size(0), 1)  # Start predicting with <s> token.
            for i in range(min(80, x.size(1))):  # Summarize the document by at most 80 words.
                output, hidden = model.decoder(output, hidden, encoder_outputs)
                output = torch.argmax(output, dim=2)
                prediction.append(output)
            prediction = torch.cat(prediction, 1)
            for i in range(x.size(0)):
                summary = []
                for word in prediction[i]:
                    if word == 2 or word == 0: break  # Stop when encountering </s> or <pad>
                    summary.append(embeddings.vocab[word.item()])
                f.write(json.dumps({'id': identifier[i], 'predict': ' '.join(summary)}) + '\n')


def load_dataloader(dataset_path, batch_size=1, shuffle=False,
                    num_workers=0, drop_last=False):
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    return DataLoader(dataset, batch_size, shuffle, collate_fn=dataset.collate_fn,
                      drop_last=drop_last)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-file', type=Path, default='./seq2seq-output.jsonl')

    parser.add_argument('--use-attention', action='store_true')

    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=4)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
