import argparse
import pickle
import json
from pathlib import Path
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from dataset import QADataset
from model import QAModel


def main(args):
    # Set up test data loader.
    loader = load_dataloader('./data/test-dataset.pkl', args.batch_size, False,
                             args.num_workers)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Set up model.
    model = QAModel()
    model.load_state_dict(torch.load('./qa.pt', map_location='cpu'))
    model.to(device)
    model.eval()

    # Load pretrained tokenizer.
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    with open(args.output_file, "w") as f:
        answer = {}
        for batch in tqdm(loader):
            qid = batch["qid"]
            x, attention_mask, token_type_ids = batch["input_ids"].to(device), batch["attention_mask"].to(device), batch["token_type_ids"].to(device)
            cls, start, end = model(x, attention_mask, token_type_ids)
            cls = torch.sigmoid(cls)

            context_len = batch["context_len"]
            for i in range(len(qid)):
                # Summarize if the probability of [cls] >= 0.5.
                if cls[i] >= 0.5:
                    predict = []
                    s = start[i][1:1 + context_len[i]]
                    s = torch.argmax(s, dim=0)
                    if s == context_len[i] - 1:
                        predict = [x[i][context_len[i]]]
                    else:
                        # Choose the token with the max probability after start to be the end token.
                        e = end[i][1:1 + context_len[i]]
                        e = torch.argmax(e[s + 1:], dim=0)
                        e += s + 1
                        predict = x[i][1 + s:1 + e + 1]
                    
                    # Summarize the document by as most 30 words.
                    if len(predict) > 30:
                        predict = predict[:30]

                    # Truncate [pad] tokens.
                    for j in range(len(predict)):
                        if predict[j] == 0:
                            predict = predict[0:j]

                    answer[qid[i]] = tokenizer.decode(predict).replace(" ","")
                else:
                    answer[qid[i]] = ""
            break
        f.write(json.dumps(answer))


def load_dataloader(dataset_path, batch_size=1, shuffle=False,
                    num_workers=0, drop_last=False):
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    return DataLoader(dataset, batch_size, shuffle, collate_fn=dataset.collate_fn,
                      drop_last=drop_last)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-file', type=Path, default='./output.json')

    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=4)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
