from transformers import BertModel, BertPreTrainedModel
import torch
from torch import nn, einsum
import torch.nn.functional as F
from utils.ate_bio_datamodule import aspect_bio_tagset




class ATEModel(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.n_labels = len(aspect_bio_tagset)
        self.bert = BertModel(config)
        self.cls  = nn.Linear(config.hidden_size, self.n_labels)

    def forward(self, input_ids, attention_mask, bio_labels):

        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]

        bio_logits = self.cls(sequence_output)
        loss_func = nn.CrossEntropyLoss()
        loss = loss_func(bio_logits.view(-1, self.n_labels), bio_labels.view(-1))

        preds = self.decode_batch(bio_logits)        

        return {
            'loss' : loss,
            'preds': preds
        }

    def decode_batch(self, bio_logits):
        # bio_logits: B L N
        bio_predictions = bio_logits.argmax(dim=-1).cpu().numpy() # [B, L]
        
        span_predictions = []
        for i in range(bio_predictions.shape[0]):
            spans = [(start-1, end-1) for type_, start, end in aspect_bio_tagset.parse(bio_predictions[i])]
            span_predictions.append(spans)

        return span_predictions

