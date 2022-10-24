import logging
import json
import pickle

import torch
from transformers import BertTokenizer

from dataset import QADataset


def main():
    # Load pretrained tokenizer.
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')


    logging.info('Create dataset...')

    with open('./data/train.json', 'r') as f:
        data = json.load(f)['data']
    create_dataset(preprocess_sample(tokenizer, data),
            './data/train-dataset.pkl', tokenizer)
    
    with open('./data/dev.json', 'r') as f:
        data = json.load(f)['data']
    create_dataset(preprocess_sample(tokenizer, data),
            './data/dev-dataset.pkl', tokenizer)

    with open('./data/test.json', 'r') as f:
        data = json.load(f)['data']
    create_dataset(preprocess_sample(tokenizer, data),
            './data/test-dataset.pkl', tokenizer)

def create_dataset(samples, save_path, tokenizer):
    dataset = QADataset(samples, tokenizer)
    with open(save_path, 'wb') as f:
        pickle.dump(dataset, f)


def preprocess_sample(tokenizer, samples):
    processeds = []
    for sample in samples:
        for paragraph in sample['paragraphs']:
            context = tokenizer.tokenize(paragraph['context'])
            context = tokenizer.convert_tokens_to_ids(context)

            for qa in paragraph['qas']:
                question = tokenizer.tokenize(qa['question'])
                question = tokenizer.convert_tokens_to_ids(question)

                # Bert model's input should shorter than 512 tokens, including [cls] and [sep].
                # Only truncate the context.
                processed = {
                    'context': context,
                    'context_len': min(len(context), 509 - len(question)),
                    'qid': qa['id'],
                    'question': question,
                }

                # Training and validating data have 'answer' and 'answerable' fields.
                if 'answers' in qa:
                    answer = []
                    for a in qa['answers']:
                        start = a['answer_start']
                        if start == -1:
                            answer.append((-1, -1))
                        else:
                            start = 0 if start == 0 else len(tokenizer.tokenize(paragraph['context'][:start]))
                            length = len(tokenizer.tokenize(a['text']))
                            answer.append((start, start + length))

                    processed['answer'] = answer
                    processed['answerable'] = qa['answerable']

                processeds.append(processed)

    return processeds



if __name__ == '__main__':
    main()
