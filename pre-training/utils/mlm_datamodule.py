import os
import random
import ujson as json

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from datasets import load_dataset
from transformers import AutoTokenizer, BertTokenizer, BertTokenizerFast
from transformers import DataCollatorForLanguageModeling as _DataCollatorForLanguageModeling
from transformers.data.data_collator import tolist, _torch_collate_batch

from utils import yield_data_file, mkdir_if_not_exist




class DataCollatorForLanguageModeling(_DataCollatorForLanguageModeling):

    def torch_call(self, examples):    
        input_ids      = torch.tensor([example['input_ids'] for example in examples], dtype=torch.long)
        attention_mask = torch.tensor([example['attention_mask'] for example in examples], dtype=torch.long)
        special_tokens_mask = torch.tensor([example['special_tokens_mask'] for example in examples], dtype=torch.long)

        input_ids, labels = self.torch_mask_tokens(input_ids, special_tokens_mask=special_tokens_mask)

        return {
            'input_ids': input_ids,
            'mlm_labels': labels,
            'attention_mask': attention_mask
        }

    def torch_mask_tokens(self, inputs, special_tokens_mask=None):
        labels = inputs.clone()
        probability_matrix = torch.full(labels.shape, self.mlm_probability)

        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()
        
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100
        inputs[masked_indices] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        return inputs, labels




class PretrainingDataModule(pl.LightningDataModule): 
    def __init__(
        self,
        model_name_or_path: str='',
        max_seq_length: int = -1,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        base_data_dir: str = '',
        data_dir: str = '',
        num_workers: int = 1,
        cache_dir: str='',
        train_size: int = 4_096_000,
        test_size : int = 102_400
    ):  
        super().__init__()

        self.model_name_or_path = model_name_or_path
        self.max_seq_length   = max_seq_length
        self.train_batch_size = train_batch_size
        self.eval_batch_size  = eval_batch_size
        self.base_data_dir    = base_data_dir
        self.data_dirs   = data_dir.split('__')
        self.num_workers = num_workers
        self.cache_dir  = cache_dir
        self.train_size = train_size
        self.test_size  = test_size

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
        self.collator_fn = DataCollatorForLanguageModeling(tokenizer=self.tokenizer)

    def load_dataset(self):
        data_files = []
        for data_dir in self.data_dirs:
            data_dir = os.path.join(self.base_data_dir, data_dir)
            print(data_dir)
            for data_file_name in yield_data_file(data_dir):
                data_files.append(data_file_name)

        if self.cache_dir:
            cache_dir = os.path.join(self.cache_dir, 'load', 'processed.arrow')
            print(cache_dir)
            mkdir_if_not_exist(cache_dir)
            self.raw_datasets = load_dataset('json', data_files=data_files, cache_dir=cache_dir, ignore_verifications=True)
        else:
            self.raw_datasets = load_dataset('json', data_files=data_files, ignore_verifications=True)
        print(self.raw_datasets)

    def prepare_dataset(self):

        def tokenize_function(examples):

            batch_encodings = self.tokenizer(
                examples['Text'],
                padding='max_length',
                truncation=True,
                max_length=self.max_seq_length,
                return_special_tokens_mask=True
            )

            opinion_tokens = []
            aspect_tokens  = []

            for i in range(len(examples['Text'])):
                def char_to_token(index, n=5):
                    for k in range(n):
                        token_index = batch_encodings[i].char_to_token(index+k)
                        if token_index is not None:
                            return token_index

                    return -1

                aspect_tokens.append([char_to_token(char_index) for char_index in examples['aspects'][i]])
                opinion_tokens.append([char_to_token(char_index) for char_index in examples['opinions'][i]])

            return {
                'input_ids'     : batch_encodings['input_ids'],
                'attention_mask': batch_encodings['attention_mask'],
                'special_tokens_mask': batch_encodings['special_tokens_mask'],
                'opinion_tokens': opinion_tokens,
                'aspect_tokens' : aspect_tokens,
            }

        kwargs = {
            'batched': True,
            'remove_columns': ['ID', 'Text', 'aspects', 'opinions'],
            'num_proc': 64
        }

        if self.cache_dir:
            cache_dir = os.path.join(self.cache_dir, 'tokenize', 'processed.arrow')
            mkdir_if_not_exist(cache_dir)
            kwargs['load_from_cache_file'] = True
            kwargs['cache_file_names'] = {
                'train': cache_dir
            }

        processed_datasets = self.raw_datasets.map(tokenize_function, **kwargs)
        print(processed_datasets)

        processed_datasets = processed_datasets['train'].train_test_split(
            test_size=self.test_size, seed=42, train_size=self.train_size
        )
        print(processed_datasets)

        self.train_dataset = processed_datasets['train']
        self.eval_dataset  = processed_datasets['test']

    def get_dataloader(self, mode, batch_size, shuffle):
        dataset = self.train_dataset if mode == 'train' else self.eval_dataset
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=self.collator_fn,
            pin_memory=True,
            prefetch_factor=16
        )

        print(mode, len(dataloader))
        return dataloader

    def train_dataloader(self):
        return self.get_dataloader('train', self.train_batch_size, shuffle=True)

    def val_dataloader(self):
        return self.get_dataloader("dev", self.eval_batch_size, shuffle=False)
