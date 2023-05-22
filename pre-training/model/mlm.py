from transformers import BertPreTrainedModel, BertModel
from transformers.models.bert.modeling_bert import BertOnlyMLMHead, BertPredictionHeadTransform


import torch
from torch import nn
import torch.nn.functional as F



class PreTrainingHeads(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.mlm_head = BertOnlyMLMHead(config)

    def forward(self, sequence_output, mlm_labels):
        mlm_loss = self.mlm_forward(sequence_output, mlm_labels)
        return {'mlm_loss': mlm_loss}

    def mlm_forward(self, hidden_states, labels):
        prediction_scores = self.mlm_head(hidden_states)
        loss_fct = nn.CrossEntropyLoss()
        masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
        return masked_lm_loss



class BERTForPreTraining(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config, add_pooling_layer=True)
        self.pretrain_head = PreTrainingHeads(config)
        # for init_weight
        self.cls = self.pretrain_head.mlm_head
        
        self.post_init()

    def get_output_embeddings(self):
        return self.pretrain_head.mlm_head.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.pretrain_head.mlm_head.predictions.decoder = new_embeddings

    def forward(self, input_ids, attention_mask, mlm_labels):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        pretrain_outputs = self.pretrain_head(sequence_output, mlm_labels)

        return pretrain_outputs
