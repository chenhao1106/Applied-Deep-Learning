import logging
import argparse
import pickle
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import QAModel


def main(args):
    # Prepare data loader.
    logging.info('Load dataset...')
    train_loader = load_dataloader('./data/train-dataset.pkl', args.batch_size, True,
                                   args.num_workers, True)
    valid_loader = load_dataloader('./data/dev-dataset.pkl', args.batch_size, False,
                                   args.num_workers)

    # Choose training device.
    device = 'cuda' if torch.cuda.is_available() else 'cpu' 

    # Set up model, optimizer and training criterion.
    model = QAModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), args.lr)    
    criterion = {
        'bce': nn.BCEWithLogitsLoss(pos_weight=torch.tensor([args.pos_weight])).to(device),
        'ce': nn.CrossEntropyLoss(ignore_index=-1).to(device)
    }

    # Tensorboard writer
    writer = SummaryWriter(log_dir='./run/qa')

    # Train and validate.
    best_valid_loss = 1000.0
    for epoch in range(args.epochs):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        valid_loss = valid(model, valid_loader, criterion, device)

        logging.info(f'Epoch {epoch}  |  train loss = {train_loss:.4f}  |  valid loss = {valid_loss:.4f}')
        writer.add_scalars('Loss', {'train': train_loss, 'valid': valid_loss})


        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), './qa.pt')


def load_dataloader(dataset_path, batch_size=1, shuffle=False,
                    num_workers=0, drop_last=False):
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    return DataLoader(dataset, batch_size, shuffle, collate_fn=dataset.collate_fn,
                      drop_last=drop_last)


def train(model, loader, optimizer, criterion, device):
    model.train()
    train_loss = 0.
    for batch in tqdm(loader):
        x, attention_mask, token_type_ids = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['token_type_ids'].to(device)
        y_cls, y = batch['answerable'].to(device), batch['answer'].to(device)
        y_start = y[:, 0].reshape(-1)
        y_end = y[:, 1].reshape(-1)

        cls, start, end = model(x, attention_mask, token_type_ids)

        loss = criterion['bce'](cls, y_cls)
        context_len = batch['context_len']
        for i in range(x.size(0)):
            if y_start[i].item() != -1: loss += criterion['ce'](start[i][1:1 + context_len[i]].unsqueeze(0), y_start[i].unsqueeze(0))
            if y_end[i].item() != -1: loss += criterion['ce'](end[i][1:1 + context_len[i]].unsqueeze(0), y_end[i].unsqueeze(0))
        loss /= x.size(0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
    train_loss /= len(loader)

    return train_loss
        

def valid(model, loader, criterion, device):
    model.eval()
    valid_loss = 0.
    with torch.no_grad():
        for batch in tqdm(loader):
            x, attention_mask, token_type_ids = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['token_type_ids'].to(device)
            y_cls, y = batch['answerable'].to(device), batch['answer'].to(device)
            y_start = y[:, 0].reshape(-1)
            y_end = y[:, 1].reshape(-1)

            cls, start, end = model(x, attention_mask, token_type_ids)

            loss = criterion['bce'](cls, y_cls)
            context_len = batch['context_len']
            for i in range(x.size(0)):
                if y_start[i].item() != -1: loss += criterion['ce'](start[i][1:1 + context_len[i]].unsqueeze(0), y_start[i].unsqueeze(0))
                if y_end[i].item() != -1: loss += criterion['ce'](end[i][1:1 + context_len[i]].unsqueeze(0), y_end[i].unsqueeze(0))
            loss /= x.size(0)

            valid_loss += loss.item()
    valid_loss /= len(loader)
    
    return valid_loss


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--num-workers', type=int, default=4)

    parser.add_argument('--lr', type=float, default=5e-5, help='learning rate')
    parser.add_argument('--pos-weight', type=float, default=0.428, help='weight of postive samples in BCELoss')

    parser.add_argument('--epochs', type=int, default=8)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)

