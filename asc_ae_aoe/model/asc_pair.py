from transformers import BertModel
from transformers import BertPreTrainedModel
import torch
from torch import nn, einsum
import torch.nn.functional as F


class ASCModel(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(config.hidden_size, config.num_sentiment_labels)
        )

    def forward(self, input_ids, attention_mask, token_type_ids, labels):

        outputs = self.bert(
            input_ids, 
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        pooled_output = outputs[1]
        logits = self.classifier(pooled_output)

        loss_func = nn.CrossEntropyLoss()
        loss  = loss_func(logits, labels)
        preds = logits.argmax(dim=-1).cpu().numpy()

        return {
            'loss': loss,
            'preds': preds
        }



