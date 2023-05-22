import os
import random

from utils.vocab import Vocab
from utils.spacy_token import Doc
from utils import yield_data_file

from datasets import load_dataset



def simplify_doc(doc, available_postags={'ADJ', 'NOUN'}):
    """
    只保留 特定词性集 中的词
    """
    available_indices = [token.i for token in doc if token.pos_ in available_postags]
    simp_doc = [' '.join((token.lemma_, token.pos_)) for token in doc if token.i in available_indices]
    return simp_doc



def main(args):
    # TODO: 检查文件是否存在
    data_files = {name: file_name for name, file_name in yield_data_file(args.input_dir)}
    raw_datasets = load_dataset('json', data_files=data_files, cache_dir=args.cache_dir)
    print(raw_datasets)


    def process_function1(examples):
        simp_docs = []
        for parsed in examples['parsed']:
            doc = Doc(parsed)
            simp_doc = simplify_doc(doc)
            simp_docs.append(simp_doc)

        return {'simp_doc': simp_docs}

    
    processed_datasets = raw_datasets.map(
        process_function1,
        batched=True,
        batch_size=args.batch_size,
        num_proc=args.num_proc
    )
    print(processed_datasets)

    print('build vocab')
    vocab = Vocab(min_count=5)
    vocab.build2(processed_datasets, key='simp_doc')
    print(f'save {len(vocab)} to {os.path.join(args.output_dir, 'vocab.json')}')
    vocab.save(os.path.join(args.output_dir, 'vocab.json'))


    def process_function2(examples):
        simp_docs = [vocab.list_word2id(doc) for doc in examples['simp_doc']]
        return {'simp_doc': simp_docs}


    processed_datasets = processed_datasets.map(
        process_function2,
        batched=True,
        batch_size=args.batch_size,
        num_proc=args.num_proc//4,
        remove_columns=['Text', 'parsed'],
        writer_batch_size=64
    )
    print(processed_datasets)
    for split in processed_datasets:
        file_name = os.path.join(args.output_dir, split)
        print('save to', file_name)
        processed_datasets[split].to_json(file_name)



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--input_dir',    type=str)
    parser.add_argument('--file_names',   type=str)
    parser.add_argument('--output_dir',   type=str)
    parser.add_argument('--cache_dir',    type=str)
    parser.add_argument('--batch_size',   default=32, type=int)
    parser.add_argument('--num_proc',     default=48, type=int)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    args.cache_dir = os.path.join(args.cache_dir, 'load/processed.arrow')

    print(args)
    random.seed(args.seed)
    main(args)
