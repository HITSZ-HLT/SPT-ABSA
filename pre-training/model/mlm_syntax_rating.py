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
        self.rating_classifier = nn.Linear(config.hidden_size, 5)
        
        self.dir_tran1 = BertPredictionHeadTransform(config)
        self.dir_tran2 = BertPredictionHeadTransform(config)
        self.dir_classifier = nn.Linear(config.hidden_size*4, 1)

        self.dis_tran1 = BertPredictionHeadTransform(config)
        self.dis_tran2 = BertPredictionHeadTransform(config)
        self.dis_classifier = nn.Linear(config.hidden_size*4, 5)

        self.part_of_speech_ffn = BertPredictionHeadTransform(config)
        self.part_of_speech_cls = nn.Linear(config.hidden_size, 18)

    def forward(self, sequence_output, pooled_output,
                mlm_labels, rating_labels,
                dir_batch, dir_token, dir_head, pos_labels, 
                dis_batch, dis_token1, dis_token2, dis_labels):

        mlm_loss = self.mlm_forward(sequence_output, mlm_labels)
        rating_loss = self.rating_forward(pooled_output, rating_labels)

        dir_loss = self.dir_forward(
            sequence_output, dir_batch, dir_token, dir_head)

        dis_loss = self.dis_forward(
            sequence_output, dis_batch, dis_token1, dis_token2, dis_labels)

        pos_loss = self.pos_forward(sequence_output, pos_labels)

        # print(mlm_loss, dir_loss, dep_loss, dis_loss)
        
        return {
            'mlm_loss': mlm_loss,
            'rat_loss': rating_loss,
            'dir_loss': dir_loss,
            'dis_loss': dis_loss,
            'pos_loss': pos_loss
        }

    def mlm_forward(self, hidden_states, labels):
        prediction_scores = self.mlm_head(hidden_states)
        loss_fct = nn.CrossEntropyLoss()
        masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
        return masked_lm_loss

    def rating_forward(self, cls_hidden, labels):
        logits = self.rating_classifier(cls_hidden)
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits, labels)
        return loss

    def dir_forward(self, hidden_states, dir_batch, dir_token, dir_head):
        t1 = hidden_states[dir_batch, dir_token]
        t2 = hidden_states[dir_batch, dir_head]

        labels = torch.randint(0, 2, (t1.size(0), ), device=t1.device).bool()
        t1_ = torch.cat([t1[labels], t2[~labels]], dim=0)
        t2_ = torch.cat([t2[labels], t1[~labels]], dim=0)

        t1_ = self.dir_tran1(t1_)
        t2_ = self.dir_tran2(t2_)

        hidden = torch.cat([t1_, t2_, t1_-t2_, t1_*t2_], dim=1)
        logits = self.dir_classifier(hidden).squeeze(-1)

        labels_ = torch.zeros_like(labels)
        labels_[:labels.sum()] = 1

        loss = F.binary_cross_entropy_with_logits(logits, labels_.float())

        # print(logits.size(), labels_.size(), loss)

        return loss

    def dis_forward(self, hidden_states, dis_batch, dis_token1, dis_token2, dis_labels):

        t1 = hidden_states[dis_batch, dis_token1]
        t2 = hidden_states[dis_batch, dis_token2]

        t1 = self.dis_tran1(t1)
        t2 = self.dis_tran2(t2)
        
        hidden = torch.cat([t1, t2, t1-t2, t1*t2], dim=1)
        logits = self.dis_classifier(hidden)
        loss = F.cross_entropy(logits, dis_labels)

        # print(logits.size(), dis_labels.size(), loss)

        return loss

    def pos_forward(self, hidden_states, labels):
        hidden_states = self.part_of_speech_ffn(hidden_states)
        prediction_scores = self.part_of_speech_cls(hidden_states)
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(prediction_scores.view(-1, 18), labels.view(-1))
        return loss



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

    def forward(self, input_ids, attention_mask, mlm_labels, rating_labels,
                dir_batch, dir_token, dir_head, pos_labels, 
                dis_batch, dis_token1, dis_token2, dis_labels):

        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        pooled_output   = outputs[1]

        pretrain_outputs = self.pretrain_head(
            sequence_output=sequence_output, 
            pooled_output=pooled_output,
            mlm_labels=mlm_labels, 
            rating_labels=rating_labels,
            dir_batch=dir_batch, 
            dir_token=dir_token, 
            dir_head=dir_head, 
            pos_labels=pos_labels, 
            dis_batch=dis_batch, 
            dis_token1=dis_token1, 
            dis_token2=dis_token2, 
            dis_labels=dis_labels
        )

        return pretrain_outputs
