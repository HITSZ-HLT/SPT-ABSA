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


from model.aoe_pair import AOEModel as Model
from utils.aoe_pair_datamodule import DataModule
from utils.aoe_result import Result
from utils import params_count



logger = logging.getLogger(__name__)


class LightningModule(pl.LightningModule):
    def __init__(self, hparams, data_module):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.data_module = data_module

        self.config = AutoConfig.from_pretrained(self.hparams.model_name_or_path)
        self.model = Model.from_pretrained(self.hparams.model_name_or_path, config=self.config, ignore_mismatched_sizes=True)

        print(self.model.config)
        print('---------------------------------------------')
        print('total params_count:', params_count(self.model))
        print('---------------------------------------------')

    @pl.utilities.rank_zero_only
    def save_model(self):
        dir_name = os.path.join(self.hparams.output_dir, f'dataset={self.hparams.dataset},seed={self.hparams.seed},m={self.hparams.output_sub_dir}', 'model')
        print(f'## save model to {dir_name}')
        self.model.config.time = time.strftime('%Y-%m-%d %H_%M_%S', time.localtime())
        self.model.save_pretrained(dir_name)

    def load_model(self):
        dir_name = os.path.join(self.hparams.output_dir, f'dataset={self.hparams.dataset},seed={self.hparams.seed},m={self.hparams.output_sub_dir}', 'model')
        print(f'## load model to {dir_name}')
        self.model = Model.from_pretrained(dir_name)

    def forward(self, **inputs):
        id_start_end = inputs.pop('id_start_end')
        outputs = self.model(**inputs)

        outputs['id_start_end'] = id_start_end
        return outputs

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs['loss']

        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs['loss']

        self.log('valid_loss', loss)

        return {
            'id_start_end': outputs['id_start_end'],
            'preds': outputs['preds']
        }

    def validation_epoch_end(self, outputs):
        examples = self.data_module.raw_datasets['dev']

        self.current_val_result = Result.parse_from(outputs, examples)
        self.current_val_result.cal_metric()

        if not hasattr(self, 'best_val_result'):
            self.best_val_result = self.current_val_result
            self.save_model()

        elif self.best_val_result < self.current_val_result:
            self.best_val_result = self.current_val_result
            self.save_model()

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        examples = self.data_module.raw_datasets['test']

        self.test_result = Result.parse_from(outputs, examples)
        self.test_result.cal_metric()

    def test_epoch_end(self, outputs):
        examples = self.data_module.raw_datasets['test']

        self.test_result = Result.parse_from(outputs, examples)
        self.test_result.cal_metric()

    def setup(self, stage):
        if stage == 'fit':
            self.train_loader = self.train_dataloader()
            ngpus = (len(self.hparams.gpus.split(',')) if type(self.hparams.gpus) is str else self.hparams.gpus)
            effective_batch_size = self.hparams.train_batch_size * self.hparams.accumulate_grad_batches * ngpus
            dataset_size = len(self.train_loader.dataset)
            self.total_steps = (dataset_size / effective_batch_size) * self.hparams.max_epochs

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

        optimizer = AdamW(optimizer_grouped_parameters, eps=self.hparams.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=self.hparams.warmup_steps, 
            num_training_steps=self.total_steps
        )
        scheduler = {'scheduler': scheduler, 'interval': 'step', 'frequency': 1}

        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args(parser):
        parser.add_argument("--learning_rate", default=1e-5, type=float)
        parser.add_argument("--adam_epsilon", default=1e-8, type=float)
        parser.add_argument("--warmup_steps", default=0, type=int)
        parser.add_argument("--weight_decay", default=0., type=float)

        # parser.add_argument("--seed", default=42, type=int)
        parser.add_argument("--output_dir", type=str)
        parser.add_argument("--output_sub_dir", type=str)
        parser.add_argument("--do_train", action='store_true')

        return parser


class LoggingCallback(pl.Callback):
    def on_validation_end(self, trainer, pl_module):
        print('-------------------------------------------------------------------------------------------------------------------\n[current]\t', end='')
        pl_module.current_val_result.report()

        print('[best]\t\t', end='')
        pl_module.best_val_result.report()
        print('-------------------------------------------------------------------------------------------------------------------\n')

    def on_test_end(self, trainer, pl_module):
        pl_module.test_result.report()
        pl_module.test_result.save_metric(
            output_dir=pl_module.hparams.output_dir,
            model_name_or_path=pl_module.hparams.model_name_or_path, 
            subname=pl_module.hparams.output_sub_dir, 
            dataset=pl_module.hparams.dataset, 
            seed=pl_module.hparams.seed
        )
        



def main():
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = LightningModule.add_model_specific_args(parser)
    parser = DataModule.add_argparse_args(parser)

    args = parser.parse_args()
    pl.seed_everything(args.seed)

    if args.learning_rate >= 1:
        args.learning_rate /= 1e5

    data_module = DataModule.from_argparse_args(args)
    data_module.load_dataset()
    data_module.prepare_dataset()

    model = LightningModule(args, data_module)
    
    logging_callback = LoggingCallback()    
    kwargs = {
        'weights_summary': None,
        'callbacks': [logging_callback],
        'logger': True,
        'checkpoint_callback': False,
        'num_sanity_val_steps': 5 if args.do_train else 0,
    }

    trainer = pl.Trainer.from_argparse_args(args, **kwargs)

    if args.do_train:
        trainer.fit(model, datamodule=data_module)
        model.load_model()
        trainer.test(model, datamodule=data_module)

    else:
        model.load_model()
        trainer.test(model, datamodule=data_module)


if __name__ == '__main__':
    main()