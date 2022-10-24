## Description
In this homework, we are asked to summarize the documents in three different ways.
1. Extractive - Choose some sentences in the document as the summary.
2. Summarize - Generate the summaries by using a sequence to sequence model.
3. Summarize + Attention - Further extends the second approach with attention mechanism.


## Usage
#### Install required packages and download spacy language
```shell
$ pip3 install -r requirements.txt
$ python3 -m spacy download en_core_web_sm
```

#### Preprocess
```python
$ python3 preprocess.py /path/to/embedding-file(e.g. GloVe)
```

#### Train models
```python
$ python3 train_seq_tag.py
$ python3 train_seq2seq.py
$ python3 train_seq2seq.py --use-attention
```

#### Inference
```python
$ python3 extractive.py
$ python3 seq2seq.py
$ python3 seq2seq.py --use-attention
```

