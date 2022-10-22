import torch
from torch.utils.data import Dataset

from utils import pad_to_len


class SeqTaggingDataset(Dataset):
    ignore_idx = -100  # label padding

    def __init__(self, data, padding=0, max_text_len=300, max_summary_len=80):
        self.data = data
        self.padding = padding
        self.max_text_len = max_text_len
        self.max_summary_len = max_summary_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        instance = {
            'id': sample['id'],
            'text': sample['text'][:self.max_text_len],
            'sent_range': sample['sent_range'],
        }
        if 'label' in sample:
            instance['label'] = sample['label'][:self.max_text_len]
        return instance

    # Pad text and label
    def collate_fn(self, samples):
        batch = {}
        for key in ['id', 'sent_range']:
            batch[key] = [sample[key] for sample in samples]

        for key in ['text', 'label']:
            if any(key not in sample for sample in samples): continue

            to_len = max([len(sample[key]) for sample in samples])
            padded = pad_to_len(
                [sample[key] for sample in samples], to_len,
                self.padding if key != 'label' else SeqTaggingDataset.ignore_idx
            )
            batch[key] = torch.tensor(padded)
        return batch


