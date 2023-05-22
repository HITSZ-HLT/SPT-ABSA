import os 
import time
import argparse
import logging

import torch
import pytorch_lightning as pl 
pl.seed_everything(42)


from transformers import AutoConfig, AutoTokenizer
from torch.optim import AdamW
from pytorch_lightning.utilities import rank_zero_info
from transformers import get_linear_schedule_with_warmup

from model.mlm import BERTForPreTraining
from utils.mlm_datamodule import PretrainingDataModule
from utils import params_count


logger = logging.getLogger(__name__)


class LightningModule(pl.LightningModule):
    def __init__(self, hparams, data_module):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.data_module = data_module

        self.config = AutoConfig.from_pretrained(self.hparams.model_name_or_path)
        self.model  = BERTForPreTraining.from_pretrained(self.hparams.model_name_or_path, config=self.config)
        self.model.resize_token_embeddings(len(self.data_module.tokenizer))

        print(self.model.config)
        print('---------------------------------------------')
        print('total params_count:', params_count(self.model))
        print('---------------------------------------------')

    @pl.utilities.rank_zero_only
    def save_model(self, loss):
        dir_name = os.path.join(self.hparams.output_dir, 'model', f'step={self.global_step+1},loss={loss:.2f}')
        print(f'## save model to {dir_name}')
        self.model.config.time = time.strftime('%Y-%m-%d %H_%M_%S', time.localtime())
        self.model.config.pt_loss = loss
        self.model.save_pretrained(dir_name)
        self.data_module.tokenizer.save_pretrained(dir_name)

    def forward(self, **inputs):
        outputs = self.model(**inputs)
        return outputs

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs['mlm_loss']
        self.log('loss', loss, sync_dist=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs['mlm_loss']
        self.log('valid_loss', loss, sync_dist=True)

    def setup(self, stage):
        if stage == 'fit':
            self.total_steps = self.hparams.max_steps

    def configure_optimizers(self):

        no_decay = ['bias', 'LayerNorm.weight']
        def has_keywords(n, keywords):
            return any(nd in n for nd in keywords)

        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters() if not has_keywords(n, no_decay)],
                'lr': self.hparams.learning_rate,
                'weight_decay': 0
            },
            {
                'params': [p for n, p in self.model.named_parameters() if has_keywords(n, no_decay)],
                'lr': self.hparams.learning_rate,
                'weight_decay': self.hparams.weight_decay
            }
        ]

        optimizer = AdamW(optimizer_grouped_parameters, eps=self.hparams.adam_epsilon, betas=(self.hparams.adam_beta1, self.hparams.adam_beta2))
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=self.hparams.warmup_steps, 
            num_training_steps=self.total_steps
        )
        scheduler = {'scheduler': scheduler, 'interval': 'step', 'frequency': 1}

        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args(parser):
        parser.add_argument("--learning_rate", default=2e-4, type=float)
        parser.add_argument("--adam_epsilon",  default=1e-6, type=float)
        parser.add_argument("--adam_beta1",    default=0.9, type=float)
        parser.add_argument("--adam_beta2",    default=0.98, type=float)
        parser.add_argument("--warmup_steps",  default=0, type=int)
        parser.add_argument("--weight_decay",  default=0., type=float)

        parser.add_argument("--seed", default=42, type=int)
        parser.add_argument("--output_dir", type=str)

        return parser


class LoggingCallback(pl.Callback):
    def on_validation_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        metrics = {k:(v.detach() if type(v) is torch.Tensor else v) for k,v in metrics.items()}
        rank_zero_info(metrics)

        pl_module.save_model(loss=metrics['valid_loss'].item())


def main():
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = LightningModule.add_model_specific_args(parser)
    parser = PretrainingDataModule.add_argparse_args(parser)

    args = parser.parse_args()
    pl.seed_everything(args.seed)

    if args.learning_rate >= 1:
        args.learning_rate /= 1e5

    data_module = PretrainingDataModule.from_argparse_args(args)
    data_module.load_dataset()
    data_module.prepare_dataset()

    model = LightningModule(args, data_module)
    
    logging_callback = LoggingCallback()    
    kwargs = {
        'weights_summary': None,
        'callbacks': [logging_callback],
        'logger': True,
        'checkpoint_callback': False,
        'num_sanity_val_steps': 5,
    }

    args.val_check_interval = args.val_check_interval * args.accumulate_grad_batches

    trainer = pl.Trainer.from_argparse_args(args, **kwargs)
    trainer.fit(model, datamodule=data_module)


if __name__ == '__main__':
    main()