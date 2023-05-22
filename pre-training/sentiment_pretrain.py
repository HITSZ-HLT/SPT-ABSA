import pytorch_lightning as pl 
from transformers import AutoConfig
from utils import params_count

from mlm_pretrain import LoggingCallback
from mlm_pretrain import LightningModule as _LightningModule

from utils.spt_datamodule import PretrainingDataModule
from model.mlm_syntax_rating import BERTForPreTraining



class LightningModule(_LightningModule):    
    def __init__(self, hparams, data_module):
        super().__init__(hparams, data_module)
        self.save_hyperparameters(hparams)
        self.data_module = data_module

        self.config = AutoConfig.from_pretrained(self.hparams.model_name_or_path)
        self.model  = BERTForPreTraining.from_pretrained(self.hparams.model_name_or_path, config=self.config)
        self.model.resize_token_embeddings(len(self.data_module.tokenizer))

        print(self.model.config)
        print('---------------------------------------------')
        print('total params_count:', params_count(self.model))
        print('---------------------------------------------')

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        mlm_loss = outputs['mlm_loss']
        rat_loss = outputs['rat_loss']
        dir_loss = outputs['dir_loss']
        dis_loss = outputs['dis_loss']
        pos_loss = outputs['pos_loss']

        self.log('mlm', mlm_loss, sync_dist=True, prog_bar=True)
        self.log('rat', rat_loss, sync_dist=True, prog_bar=True)
        self.log('dir', dir_loss, sync_dist=True, prog_bar=True)
        self.log('dis', dis_loss, sync_dist=True, prog_bar=True)
        self.log('pos', pos_loss, sync_dist=True, prog_bar=True)

        loss = mlm_loss + (dis_loss * 0.3 + dir_loss + pos_loss * 0.8) * 1.2 + rat_loss * 1.2

        return loss

    def validation_step(self, batch, batch_idx):
        outputs  = self(**batch)
        mlm_loss = outputs['mlm_loss']
        rat_loss = outputs['rat_loss']
        dir_loss = outputs['dir_loss']
        dis_loss = outputs['dis_loss']
        pos_loss = outputs['pos_loss']

        self.log('valid_mlm', mlm_loss, sync_dist=True)
        self.log('valid_rat', rat_loss, sync_dist=True)
        self.log('valid_dir', dir_loss, sync_dist=True)
        self.log('valid_dis', dis_loss, sync_dist=True)
        self.log('valid_pos', pos_loss, sync_dist=True)
        
        loss = mlm_loss + (dis_loss * 0.3 + dir_loss + pos_loss * 0.8) * 1.2 + rat_loss * 1.2

        self.log('valid_loss', loss, sync_dist=True)

        return loss



def main():
    import argparse

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