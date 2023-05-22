import os
import yajl as json
from functools import partial
from collections import defaultdict

import spacy
from tqdm import tqdm
from datasets import load_dataset

from utils import _mkdir_if_not_exist, yield_data_file



def main(args):
    data_files = defaultdict(list)
    for name, file_name in yield_data_file(args.input_dir):
        i = (int(name.split('_')[1].split('.')[0])+4) // 5 # 100k_14.json => 3
        split = f'100k_{i*5-4}_{i*5}.json'
        data_files[split].append(file_name)

    raw_datasets = load_dataset('json', data_files=data_files, cache_dir=args.cache_dir)
    print(raw_datasets)

    nlp = spacy.load(args.spacy_model)
    spacy_tokenizer = partial(nlp.pipe, disable=['ner',], n_process=1)

    def tokenize_function(examples):
        parsed = []
        for doc in spacy_tokenizer(examples['Text']):
            token_strings = []
            for token in doc:
                children_i = [child.i for child in token.children]
                sent_id = hash(token.sent) % 100

                token_string = json.dumps([token.i, token.idx, token.text, token.lemma_, token.pos_, token.dep_, token.head.i, children_i, sent_id])
                token_strings.append(token_string)

            parsed.append(token_strings)

        return {'parsed': parsed}

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        batch_size=args.batch_size,
        load_from_cache_file=True,
        num_proc=args.num_proc,
    )
    print(tokenized_datasets)

    print('save to', args.output_dir)
    mkdir_if_not_exist(args.output_dir)
    for split in tokenized_datasets:
        file_name = os.path.join(args.output_dir, split)
        tokenized_datasets[split].to_json(file_name)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--spacy_model', default='en_core_web_sm', type=str)
    parser.add_argument('--input_dir',   type=str)
    parser.add_argument('--output_dir',  type=str)
    parser.add_argument('--cache_dir',   type=str)
    parser.add_argument('--batch_size',  default=32, type=int)
    parser.add_argument('--num_proc',    default=32, type=int)
    
    args = parser.parse_args()

    args.cache_dir = os.path.join(args.cache_dir, 'load/processed.arrow')

    main(args)