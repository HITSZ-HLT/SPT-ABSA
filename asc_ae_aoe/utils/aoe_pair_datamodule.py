import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl 
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split

import os
from . import load_json
from .bio import Tagset


opinion_bio_tagset = Tagset(['SOS', 'B-opinion', 'I-opinion', 'O', 'PAD'])



class Example:
    def __init__(self, mode, data, max_length=-1):
        self.data = data
        self.data['tokens'] = eval(self.data['tokens'])
        self.data['ID'] = f"{mode}_{self.data['ID']}"
        self.max_length = max_length

    def __getitem__(self, key):
        return self.data[key]

    def ID_sentence_aspect_opinions(self):
        for aspect, opinion_start_ends in self.aspect_opinions():
            yield self['ID'], self['sentencce'], aspect, opinion_start_ends

    def aspect_opinions(self):
        for entity in self['entities']:
            type_, aspect_start, aspect_end, tokens, aspect_string = entity[:5]
            if type_ == 'target':
                opinion_start_ends = []                
                for pair in self['pairs']:
                    start1, end1, start2, end2 = pair
                    if start1 == aspect_start and end1 == aspect_end:
                        opinion_start_ends.append((start2, end2, 'opinion'))
                yield (aspect_start, aspect_end, aspect_string), opinion_start_ends



class AOEDataset:
    def __init__(self):
        self.ID = []
        self.text = []
        self.aspect = []
        self.opinions = []

    def add(self, ID, text, aspect, opinions):
        self.ID.append(ID)
        self.text.append(text)
        self.aspect.append(aspect)
        self.opinions.append(opinions)

    def __len__(self):
        return len(self.ID)

    def __getitem__(self, i):
        return self.ID[i], self.text[i], self.aspect[i], self.opinions[i]



class DataCollator:
    def __init__(self, tokenizer, max_seq_length):
        self.tokenizer = tokenizer
        self.max_length = max_seq_length

    def __call__(self, examples):
        ID, text, start, end, aspect_string, opinions = [], [], [], [], [], []
        for _ID, _text, _aspect, _opinions in examples:
            ID.append(_ID)
            text.append(_text)
            opinions.append(_opinions)

            _start, _end, _aspect_string = _aspect
            start.append(_start)
            end.append(_end)
            aspect_string.append(_aspect_string)

        kwargs = {
            'text': text,
            'text_pair': aspect_string,
            'return_tensors': 'pt'
        }

        if self.max_length in (-1, 'longest'):
            kwargs['padding'] = True

        else:
            kwargs['padding'] = self.max_length
            kwargs['max_length'] = 'max_length'
            kwargs['truncation'] = True

        batch_encodings = self.tokenizer(**kwargs)
        max_seq_length = batch_encodings['input_ids'].size(1)
        text_length    = batch_encodings['attention_mask'].sum(dim=1)-2

        bio_labels = []
        for i, _opinions in enumerate(opinions):
            bio_label = opinion_bio_tagset.make(_opinions, text_length[i], max_seq_length)
            bio_labels.append(bio_label)

        bio_labels = torch.tensor(bio_labels, dtype=torch.long)

        return {
            'id_start_end'  : (ID, start, end),
            'input_ids'     : batch_encodings['input_ids'],
            'attention_mask': batch_encodings['attention_mask'],
            'token_type_ids': batch_encodings['token_type_ids'],
            'bio_labels'    : bio_labels,
        }


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
        self.merge_train_and_dev= merge_train_and_dev
        self.training_data_prop = training_data_prop

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
        except:
            self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', use_fast=True)

    def load_dataset(self):
        train_file_name = os.path.join(self.data_dir, 'train.json')
        dev_file_name   = os.path.join(self.data_dir, 'dev.json')
        test_file_name  = os.path.join(self.data_dir, 'test.json')

        train_examples = [Example('train', data, self.max_seq_length) for data in load_json(train_file_name)]
        test_examples  = [Example('test', data, self.max_seq_length) for data in load_json(test_file_name)]

        if not os.path.exists(dev_file_name):
            # 可以考虑固定self.seed
            # train_examples, dev_examples = train_test_split(train_examples, test_size=0.2, random_state=self.seed)
            train_examples, dev_examples = train_test_split(train_examples, test_size=0.2, random_state=42)
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

        for split in ('train', 'dev', 'test'):
            print(split, self.stat(self.raw_datasets[split]))

    def stat(self, examples):
        example_n = len(examples)
        aspect_n  = sum([sum([item[0]=='target' for item in example['entities']]) for example in examples])
        opinion_n = sum([sum([item[0]=='opinion' for item in example['entities']]) for example in examples])
        pair_n = sum([len(example['pairs']) for example in examples])
        return example_n, aspect_n, opinion_n, pair_n

    def prepare_dataset(self):
        self.processed_dataset = {}
        for split in self.raw_datasets:
            self.processed_dataset[split] = AOEDataset()
            for example in self.raw_datasets[split]:
                for ID, text, aspect, opinions in example.ID_sentence_aspect_opinions():
                    self.processed_dataset[split].add(ID, text, aspect, opinions)

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

