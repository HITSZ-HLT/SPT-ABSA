import os
import random
from datasets import load_dataset

from utils.vocab import Vocab 
from utils.path_manager import PathManager, create_path_mat, simplify_doc_and_path_mat
from utils.spacy_token import Doc



def main(args):
    raw_dataset = load_dataset('json', data_files=[args.input_file_name], cache_dir=args.cache_dir)
    raw_dataset = (raw_dataset['train'].train_test_split(train_size=args.max_examples, shuffle=False)['train'] 
                   if args.max_examples < len(raw_dataset['train']) else raw_dataset['train'])
    print(raw_dataset)

    def process_function1(examples):
        simp_docs = []
        simp_path_mats = []
        for parsed in examples['parsed']:
            doc = Doc(parsed)
            path_mat = create_path_mat(doc)
            simp_doc, simp_path_mat = simplify_doc_and_path_mat(doc, path_mat)
            simp_docs.append(simp_doc)
            simp_path_mats.append(simp_path_mat)

        return {'simp_doc': simp_docs, 'simp_path_mat': simp_path_mats}

    
    processed_dataset = raw_dataset.map(
        process_function1,
        batched=True,
        batch_size=args.batch_size,
        num_proc=args.num_proc
    )
    print(processed_dataset)

    print('build vocab')
    vocab = Vocab(min_count=5)
    vocab.build(processed_dataset['simp_doc'])
    vocab.save(os.path.join(args.output_dir, 'vocab.json'))
    print(len(vocab))

    print('build path-manager')
    path_manager = PathManager(min_count=10)
    # 数量很多，所以需要采样
    path_manager.build(processed_dataset[:20_000]['simp_path_mat'])
    path_manager.save(os.path.join(args.output_dir, 'path_manager.json'))
    print(len(path_manager))


    def process_function2(examples):
        simp_docs = [vocab.list_word2id(doc) for doc in examples['simp_doc']]
        simp_path_mats = [path_manager.mat_path2id(path_mat) for path_mat in examples['simp_path_mat']]
        return {'simp_doc': simp_docs, 'simp_path_mat': simp_path_mats}

    processed_dataset = processed_dataset.map(
        process_function2,
        batched=True,
        batch_size=args.batch_size,
        num_proc=args.num_proc//4,
        # remove_columns=['Overall', 'Text', 'parsed'],
        remove_columns=['Text', 'parsed'],
        writer_batch_size=64
    )
    print(processed_dataset)

    processed_dataset.to_json(os.path.join(args.output_dir, 'data.json'))


    

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--input_file_name', type=str)
    parser.add_argument('--output_dir',   type=str)
    parser.add_argument('--cache_dir',    type=str)
    parser.add_argument('--max_examples', default=500_000, type=int)
    parser.add_argument('--batch_size',   default=32, type=int)
    parser.add_argument('--num_proc',     default=48, type=int)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    args.cache_dir = os.path.join(args.cache_dir, 'load/processed.arrow')

    print(args)
    random.seed(args.seed)
    main(args)
