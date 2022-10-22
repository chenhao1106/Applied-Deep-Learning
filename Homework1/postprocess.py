import argparse
import pickle
import json
from pathlib import Path
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from model import ExtractiveModel
from utils import Embeddings

def main(args):
    # Set up test data loader.
    loader = load_dataloader('./data/seq_tag_test_dataset.pkl', args.batch_size, False,
                                   args.num_workers)

    # Load embeddings.
    with open('./embedding.pkl', 'rb') as f:
        embeddings = pickle.load(f)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Set up model.
    model = ExtractiveModel(embeddings)
    model.load_state_dict(torch.load('./extractive-model.pt', map_location=device))
    model.to(device)

    with open(args.output_file, 'w') as f:
        for batch in tqdm(loader):
            x = batch['text'].to(device)
            out = torch.sigmoid(model(x))
            identifier = batch['id']

            # 
            prediction = []
            for i in range(len(x)):
                sent_range = batch['sent_range'][i]

                predict = []
                tmp = []
                for j, (start, end) in enumerate(sent_range):
                    prob = torch.mean(out[i][start:end])
                    tmp.append((prob, j))
                tmp.sort(reverse=True)

                # Summarize the document by less than 5 sentences.
                for j in range(min(len(tmp), 5)):
                    # Choose the sentences with the probabilitis more higher than 0.5
                    if tmp[j][0] > 0.5:
                        predict.append(tmp[j][1])
                    else:
                        break
                # Choose at least on sentence to summarize the document.
                if len(predict) == 0:
                    predict.append(tmp[0][1])

                prediction.append({'id': identifier[i], 'predict_sentence_index': predict})

            for predict in prediction:
                f.write(json.dumps(predict) + '\n')
    

def load_dataloader(dataset_path, batch_size=1, shuffle=False,
                    num_workers=0, drop_last=False):
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    return DataLoader(dataset, batch_size, shuffle, collate_fn=dataset.collate_fn,
                      drop_last=drop_last)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-file', type=Path, default='./output.jsonl')

    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=4)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
