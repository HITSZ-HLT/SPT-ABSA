import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl 
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split

import os
from . import load_json


polarity_map = {
    'NEG': 0,
    'NEU': 1,
    'POS': 2,
    # 'CON': 3,
    0: 'NEG',
    1: 'NEU',
    2: 'POS',
    # 3: 'CON',
}


class Example:
    def __init__(self, mode, data, max_length=-1):
        self.data = data
        self.data['tokens'] = eval(self.data['tokens'])
        self.data['ID'] = f"{mode}_{self.data['ID']}"
        self.max_length = max_length

    def __getitem__(self, key):
        if key == 'aspects':
            return [aspect for aspect in self.data['aspects'] if len(aspect) > 0 and aspect[2] not in ('None', 'CON')]
        else:
            return self.data[key]

    def aspect_polarity(self):
        for aspect in self['aspects']:
            start, end, polarity, aspect_token = aspect[:4]
            yield self['ID'], self['sentence'], start, end, eval(aspect_token), polarity_map[polarity]




class DataCollator:
    def __init__(self, tokenizer, max_seq_length):
        self.tokenizer = tokenizer
        self.max_length = max_seq_length

    def __call__(self, examples):

        ID, text, start, end, aspect_string, polarity = [], [], [], [], [], []
        for _ID, _text, _start, _end, _aspect_token, _polarity in examples:
            ID.append(_ID)
            text.append(_text)
            start.append(_start)
            end.append(_end)
            aspect_string.append(self.tokenizer.convert_tokens_to_string(_aspect_token))
            polarity.append(_polarity)

        kwargs = {
            'text': aspect_string,
            'text_pair': text,
            'return_tensors': 'pt'
        }

        if self.max_length in (-1, 'longest'):
            kwargs['padding'] = True

        else:
            kwargs['padding'] = self.max_length
            kwargs['max_length'] = 'max_length'
            kwargs['truncation'] = True

        batch_encodings = self.tokenizer(**kwargs)
        labels = torch.tensor(polarity, dtype=torch.long)

        return {
            'id_start_end'  : (ID, start, end),
            'input_ids'     : batch_encodings['input_ids'],
            'attention_mask': batch_encodings['attention_mask'],
            'token_type_ids': batch_encodings['token_type_ids'],
            'labels'        : labels,
        }


class ASCDataset:
    def __init__(self):
        self.ID = []
        self.text = []
        self.start = []
        self.end = []
        self.aspect_token = []
        self.polarity = []

    def add(self, ID, text, start, end, aspect_token, polarity):
        self.ID.append(ID)
        self.text.append(text)
        self.start.append(start)
        self.end.append(end)
        self.aspect_token.append(aspect_token)
        self.polarity.append(polarity)

    def __len__(self):
        return len(self.ID)

    def __getitem__(self, i):
        return self.ID[i], self.text[i], self.start[i], self.end[i], self.aspect_token[i], self.polarity[i]


class DataModule(pl.LightningDataModule):
    def __init__(self,
                 model_name_or_path: str='',
                 max_seq_length: int = -1,
                 train_batch_size: int = 32,
                 eval_batch_size: int = 32,
                 data_dir: str = '',
                 dataset: str = '',
                 seed: int = 42,
                 merge_train_and_dev: bool = False,
                 training_data_prop: float = 1.,
                ):

        super().__init__()

        self.model_name_or_path = model_name_or_path
        self.max_seq_length     = max_seq_length
        self.train_batch_size   = train_batch_size
        self.eval_batch_size    = eval_batch_size
        self.data_dir           = os.path.join(data_dir, dataset)
        self.seed               = seed
        self.training_data_prop = training_data_prop

        # true for the comparison with scapt
        self.merge_train_and_dev= merge_train_and_dev 

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
        except:
            self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', use_fast=True)

    def load_dataset(self):
        train_file_name = os.path.join(self.data_dir, 'train.json')
        dev_file_name   = os.path.join(self.data_dir, 'dev.json')
        test_file_name  = os.path.join(self.data_dir, 'test.json')

        # train_file_name = os.path.join(self.data_dir, 'train.txt')
        # dev_file_name   = os.path.join(self.data_dir, 'dev.txt')
        # test_file_name  = os.path.join(self.data_dir, 'test.txt')

        train_examples = [Example('train', data, self.max_seq_length) for data in load_json(train_file_name)]
        test_examples  = [Example('test', data, self.max_seq_length) for data in load_json(test_file_name)]

        if not os.path.exists(dev_file_name):
            train_examples, dev_examples = train_test_split(train_examples, test_size=0.2, random_state=self.seed)
        else:
            dev_examples = [Example('dev', data, self.max_seq_length) for data in load_json(dev_file_name)]        

        # scapt
        if self.merge_train_and_dev:
            train_examples = train_examples + dev_examples
            dev_examples   = test_examples

        if self.training_data_prop < 1:
            import random
            random.seed(42)
            k = int(len(train_examples) * self.training_data_prop)
            train_examples = random.sample(train_examples, k=k)

        self.raw_datasets = {
            'train': train_examples, 
            'dev'  : dev_examples,
            'test' : test_examples
        }

    def prepare_dataset(self):
        self.processed_dataset = {}
        for split in self.raw_datasets:
            self.processed_dataset[split] = ASCDataset()
            for example in self.raw_datasets[split]:
                for ID, text, start, end, aspect_token, polarity_label in example.aspect_polarity():
                    self.processed_dataset[split].add(ID, text, start, end, aspect_token, polarity_label)

    def get_dataloader(self, mode, batch_size, shuffle):
        dataloader = DataLoader(
            dataset=self.processed_dataset[mode],
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=DataCollator(tokenizer=self.tokenizer, max_seq_length=self.max_seq_length),
            pin_memory=True,
            prefetch_factor=16,
            num_workers=1
        )

        print(mode, len(dataloader))
        return dataloader

    def train_dataloader(self):
        return self.get_dataloader('train', self.train_batch_size, shuffle=True)

    def val_dataloader(self):
        return self.get_dataloader("dev", self.eval_batch_size, shuffle=False)

    def test_dataloader(self):
        return self.get_dataloader("test", self.eval_batch_size, shuffle=False)

