import torch
from torch.utils.data import Dataset


class QADataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        instance = {
            'context': sample['context'],
            'context_len': sample['context_len'],
            'qid': sample['qid'],
            'question': sample['question']
        }

        if 'answer' in sample:
            instance['answer'] = sample['answer']
            instance['answerable'] = sample['answerable']

        return instance

    def collate_fn(self, samples):
        batch = {}

        text = [(sample['context'], sample['question']) for sample in samples]
        batch = self.tokenizer.batch_encode_plus(text, add_special_tokens=True,
            padding='max_length', truncation='only_first', is_split_into_words=True,
            return_tensors='pt', return_token_type_ids=True, return_attention_mask=True)
        batch['input_ids'] = batch['input_ids'].type(torch.long)
        batch['qid'] = [sample['qid'] for sample in samples]
        batch['token_type_ids'] = batch['token_type_ids'].type(torch.long)
        batch['attention_mask'] = batch['attention_mask'].type(torch.long)
        batch['context_len'] = [sample['context_len'] for sample in samples]

        if 'answer' in samples[0]:
            batch['answerable'] = torch.tensor([[1 if sample['answerable'] else 0] for sample in samples], dtype=torch.float)

            answers = []
            for i, sample in enumerate(samples):
                done = False
                for answer in sample['answer']:
                    if answer[0] == -1:
                        answers.append((-1, -1))
                        done = True
                        break

                    elif answer[0] < batch['context_len'][i] and answer[1] <= batch['context_len'][i]:
                        answers.append((answer[0], answer[1] - 1))
                        done = True
                        break

                if not done:
                    answers.append((-1, -1))
                    batch['answerable'][i][0] = 0.0

            batch['answer'] = torch.tensor(answers, dtype=torch.long)

        return batch

