import os
import random
import ujson as json

import torch
from torch.distributions.geometric import Geometric
from torch.utils.data import DataLoader

import pytorch_lightning as pl
import datasets
from datasets import load_dataset
from transformers import AutoTokenizer, BertTokenizer, BertTokenizerFast
from transformers import DataCollatorForLanguageModeling as _DataCollatorForLanguageModeling
from transformers.data.data_collator import tolist, _torch_collate_batch


from utils import yield_data_file, mkdir_if_not_exist
from utils.lite_spacy_token import Doc, yield_dep_distance2



def random_sample(population, k):
    if k != int(k):
        k = int(k) + (random.random()<(k-int(k)))

    if k > len(population):
        k = len(population)

    return random.sample(population, k=int(k))




def merge_adjacent_aspect(aspects):
    if len(aspects) == 0:
        return aspects
    aspects = sorted(aspects)
    new_aspects = []
    last_aspect = aspects[0]
    for i in range(1, len(aspects)):
        start1, end1 = last_aspect
        start2, end2 = aspects[i]
        if end1 == start2:
            last_aspect = start1, end2
        else:
            new_aspects.append(last_aspect)
            last_aspect = start2, end2
    new_aspects.append(last_aspect)
    return new_aspects




class DataCollatorForLanguageModeling(_DataCollatorForLanguageModeling):

    sentiment_dict = {1.: 0, 2.: 1, 3.: 2, 4.: 3, 5.: 4, 0.: -100}
    def __init__(self, geo_p, min_prob, z_for_aspect, tokenizer, mlm_probability=0.15, pad_to_multiple_of=None):

        self.geo_p = geo_p
        self.min_prob = min_prob
        self.z_for_aspect = z_for_aspect

        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability
        self.pad_to_multiple_of = pad_to_multiple_of

        print()
        print('aspect_geo_p', self.geo_p)
        print('min_prob', self.min_prob)
        print('z_for_aspect', self.z_for_aspect)
        print()

    def torch_call(self, examples):
        input_ids      = torch.tensor([example['inp'] for example in examples], dtype=torch.long)
        lengths        = torch.tensor([example['len'] for example in examples], dtype=torch.long)
        max_length     = input_ids.size(1)
        attention_mask = torch.tensor([[1]*length + [0]*(max_length-length) for length in lengths.tolist()], dtype=torch.bool)


        if 'special_tokens_mask' not in examples[0]:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in input_ids.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = torch.tensor([example['special_tokens_mask'] for example in examples], dtype=torch.bool)

        input_ids, labels = self.torch_mask_tokens(input_ids, examples, special_tokens_mask=special_tokens_mask, lengths=lengths)
        rating_labels = torch.tensor([self.sentiment_dict[example['Overall']] for example in examples], dtype=torch.long)

        docs = [Doc(example['par'], 'char') for example in examples]

        dir_batch, dir_token, dir_head, pos_labels = self.build_direction(docs, input_ids)
        dis_batch, dis_token1, dis_token2, dis_labels = self.build_distance(docs)

        return {
            'input_ids': input_ids,
            'mlm_labels': labels,
            'attention_mask': attention_mask,
            'dir_batch': dir_batch, 
            'dir_token': dir_token, 
            'dir_head' : dir_head, 
            'pos_labels': pos_labels,
            'dis_batch': dis_batch, 
            'dis_token1': dis_token1, 
            'dis_token2': dis_token2, 
            'dis_labels': dis_labels,
            'rating_labels': rating_labels
        }

    def loop(self, seq, width=4):
        for i in range(len(seq)//width):
            yield seq[i*width: i*width+width]

    def build_direction(self, docs, inputs):
        pos_labels = torch.full(inputs.shape, -100)
        dir_batch, dir_token, dir_head = [], [], []

        for i, doc in enumerate(docs):
            for token in doc:
                dir_batch.append(i)
                dir_token.append(token.start)
                dir_head.append(token.head.start)
                pos_labels[i, token.start:token.end] = token.pos_

        dir_batch = torch.tensor(dir_batch, dtype=torch.long)
        dir_token = torch.tensor(dir_token, dtype=torch.long)
        dir_head  = torch.tensor(dir_head,  dtype=torch.long)

        return dir_batch, dir_token, dir_head, pos_labels

    def build_distance(self, docs, max_distance=5):
        dis_batch, dis_token1, dis_token2, dis_labels = [], [], [], []
        for i, doc in enumerate(docs):
            for s1, s2, d in yield_dep_distance2(doc, max_distance=10):
                dis_batch.append(i)
                dis_token1.append(s1)
                dis_token2.append(s2)
                dis_labels.append(d)

        dis_batch  = torch.tensor(dis_batch,  dtype=torch.long)
        dis_token1 = torch.tensor(dis_token1, dtype=torch.long)
        dis_token2 = torch.tensor(dis_token2, dtype=torch.long)
        dis_labels = torch.tensor(dis_labels, dtype=torch.long)
        dis_labels[dis_labels>5] = 5
        dis_labels = dis_labels - 1

        return dis_batch, dis_token1, dis_token2, dis_labels

    def torch_mask_tokens(self, inputs, examples, special_tokens_mask, lengths):

        labels = inputs.clone()
        probability_matrix = torch.full(labels.shape, 1e-6)
        # special_tokens_mask = special_tokens_mask.bool()

        probability_matrix = self.build_probability_matrix(
            probability_matrix, examples, lengths, special_tokens_mask
        )
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100
        inputs[masked_indices] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        return inputs, labels

    def dual_geo(self, max_length, start, end):
        n = Geometric(self.geo_p)
        x = torch.arange(max_length)
        probs = torch.zeros(max_length)

        # ------[start  end)---------------
        # -----(start-1  end)---------------
        probs[x>=end] = n.log_prob((x-end)[x>=end]).exp()
        probs[x<=(start-1)] = n.log_prob((start-1-x)[x<=(start-1)]).exp()

        return probs

    def build_probability_matrix(self, probability_matrix, examples, lengths, special_tokens_mask):

        max_length = probability_matrix.size(1)
        # -2 for [CLS] and [SEP]
        lengths = lengths-2

        sampled_aspect_mask = torch.zeros_like(probability_matrix).bool()
        for i, example in enumerate(examples):
            # 收集aspect
            all_aspect = []
            aspect_tokens = example['asp']
            for j in range(len(aspect_tokens)//2):
                start, end = aspect_tokens[j*2: j*2+2]
                # if -1 not in (start, end):
                all_aspect.append((start, end))

            # all_aspect = list(self.loop(example['asp'], width=2))
            all_aspect = merge_adjacent_aspect(all_aspect)

            sum_aspect_prob = self.z_for_aspect * lengths[i]

            for start, end in random_sample(all_aspect, k=sum_aspect_prob):
                probs = self.dual_geo(max_length, start, end)
                _flag = probability_matrix[i]<probs
                probability_matrix[i][_flag] = probs[_flag]
                sampled_aspect_mask[i, start:end] = True

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        probability_matrix.masked_fill_(sampled_aspect_mask, value=1e-6)

        norm = probability_matrix.sum(dim=1, keepdim=True)
        probability_matrix = probability_matrix * lengths[:, None] * (self.mlm_probability - self.min_prob) / norm
        probability_matrix += self.min_prob
        probability_matrix[probability_matrix>1] = 1
        return probability_matrix




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
        test_size : int = 102_400,
        z_for_aspect: float = 0.1,
        geo_p: float = 0.4,
        min_prob: float = 0.04,
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
        self.collator_fn = DataCollatorForLanguageModeling(geo_p, min_prob, z_for_aspect, tokenizer=self.tokenizer)

    def load_dataset(self):
        datasets.config.IN_MEMORY_MAX_SIZE = 100_000_000
        print('datasets.config.IN_MEMORY_MAX_SIZE', datasets.config.IN_MEMORY_MAX_SIZE)

        data_files = []
        for data_dir in self.data_dirs:
            data_dir = os.path.join(self.base_data_dir, data_dir)
            # max files for each domain
            file_names = list(yield_data_file(data_dir))
            file_names = sorted(file_names, key=lambda s: int(s.split('_')[-2]))[:12]

            print(data_dir, len(file_names))

            data_files.extend(file_names)

        print('sum file:', len(data_files))

        if self.cache_dir:
            cache_dir = os.path.join(self.cache_dir, 'load', 'processed.arrow')
            print(cache_dir)
            mkdir_if_not_exist(cache_dir)
            self.raw_datasets = load_dataset('json', data_files=data_files, cache_dir=cache_dir, ignore_verifications=True)
        else:
            self.raw_datasets = load_dataset('json', data_files=data_files, ignore_verifications=True)

        print(self.raw_datasets)

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

    def prepare_dataset(self):

        def tokenize_function(examples):

            batch_encodings = self.tokenizer(
                examples['Text'],
                padding='max_length',
                truncation=True,
                max_length=self.max_seq_length
            )

            aspect_tokens = []
            parsed_char   = []

            for i in range(len(examples['Text'])):
                def char_to_token(index, n=5):
                    for k in range(n):
                        token_index = batch_encodings[i].char_to_token(index+k)
                        if token_index is not None:
                            return token_index

                    return sum(batch_encodings['attention_mask'][i])-1

                aspect_tokens.append([char_to_token(char_index) for char_index in examples['aspects'][i]])

                doc = Doc(examples['lite_parsed'][i])
                doc.char_to_token2(func=char_to_token)
                parsed_char.append(doc.to_string())

            return {
                'inp': batch_encodings['input_ids'],
                'len': [sum(val) for val in batch_encodings['attention_mask']],
                'asp': aspect_tokens,
                'par': parsed_char
            }

        kwargs = {
            'batched': True,
            'remove_columns': ['ID', 'Text', 'aspects', 'lite_parsed', 'Summary'],
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

        # https://discuss.huggingface.co/t/solved-image-dataset-seems-slow-for-larger-image-size/10960/6
        processed_datasets = processed_datasets.with_format("numpy")
        # dataset.set_format(type='torch', columns=['input_ids'])

        self.train_dataset = processed_datasets['train']
        self.eval_dataset  = processed_datasets['test']


