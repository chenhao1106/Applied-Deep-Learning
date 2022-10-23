import argparse
import pickle
import logging
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import SeqTaggingDataset
from model import ExtractiveModel


def main(args):
    # Prepare data loader.
    logging.info('Load dataset...')
    train_loader = load_dataloader('./data/seq_tag_train_dataset.pkl', args.batch_size, True,
                                   args.num_workers, True)
    valid_loader = load_dataloader('./data/seq_tag_valid_dataset.pkl', args.batch_size, False,
                                   args.num_workers)

    # Choose training device.
    device = 'cuda' if torch.cuda.is_available() else 'cpu' 

    # Load word embeddings.
    with open('./embedding.pkl', 'rb') as f:
        embeddings = pickle.load(f)

    # Set up model, optimizer and training criterion.
    model = ExtractiveModel(embeddings).to(device)
    optimizer = torch.optim.Adam(model.parameters(), args.lr)    
    criterion = torch.nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor([args.pos_weight])).to(device)

    # Tensorboard writer
    writer = SummaryWriter(log_dir='./run/seq_tag')

    # Train and validate.
    best_valid_loss = 1000.0
    for epoch in range(args.epochs):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        valid_loss = valid(model, valid_loader, criterion, device)

        logging.info(f'Epoch {epoch}  |  train loss = {train_loss:.4f}  |  valid loss = {valid_loss:.4f}')
        writer.add_scalars('Loss', {'train': train_loss, 'valid': valid_loss})


        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), './extractive-model.pt')


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
        x, y = batch['text'].to(device), batch['label'].type(torch.float).to(device)
        out = model(x)
        
        loss = criterion(out, y)
        loss[y == SeqTaggingDataset.ignore_idx] = 0.
        loss = torch.mean(loss)

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
            x, y = batch['text'].to(device), batch['label'].type(torch.float).to(device)
            out = model(x)
            
            loss = criterion(out, y)
            loss[y == SeqTaggingDataset.ignore_idx] = 0.
            loss = torch.mean(loss)
            valid_loss += loss.item()
    valid_loss /= len(loader)
    
    return valid_loss


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=4)
    
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--pos-weight', type=float, default=4.995, help='weight of postive samples in BCELoss')

    parser.add_argument('--epochs', type=int, default=8)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s',
                        level='INFO', datefmt='%Y-%m-%d %H:%M:%S')
    main(args)
