import torch
import torch.nn as nn
from transformers import BertModel


class QAModel(nn.Module):
    def __init__(self):
        super(QAModel, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-chinese')
        self.cls = nn.Linear(768, 1)

        self.start = nn.Linear(768, 1)
        self.end = nn.Linear(768, 1)

    def forward(self, x, attention_mask=None, token_type_ids=None):
        x, _ = self.bert(x, attention_mask=attention_mask, token_type_ids=token_type_ids, return_dict=False)
        cls = self.cls(x[:, 0, :])
        start = self.start(x).squeeze(2)
        end = self.end(x).squeeze(2)

        return cls, start, end

