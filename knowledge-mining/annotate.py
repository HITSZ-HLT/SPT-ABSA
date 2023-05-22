import os
import numpy as np
from datasets import load_dataset

from utils.vocab import Vocab 
from utils.path_manager import PathManager, create_path_mat
from utils.spacy_token import Doc
from utils import _mkdir_if_not_exist, yield_data_file, load_json
from mine import Annotation as _Annotation



def simplify_doc_and_path_mat(doc, path_mat, available_postags={'ADJ', 'NOUN'}):
    available_indices = [token.i for token in doc if token.pos_ in available_postags]
    words = [' '.join((token.lemma_, token.pos_)) for token in doc if token.i in available_indices]
    simp_doc = [token for token in doc if token.i in available_indices]
    simp_pat_mat = [[path_mat[i][j] for j in available_indices] for i in available_indices]
    return words, simp_doc, simp_pat_mat



class Annotation(_Annotation):
    def load_polarity(self, file_name):
        data = load_json(file_name)
        positive = data['positive']
        negative = data['negative']

        print('pos', len(positive), 'neg', len(negative))
        self.pmi_so = np.zeros(len(self.vocab)) - 100
        for word in positive:
            word = f'{word} ADJ'
            index = self.vocab[word]
            self.pmi_so[index] = 1

        for word in negative:
            word = f'{word} ADJ'
            index = self.vocab[word]
            self.pmi_so[index] = 0

        self.pmi_so[0] = -100

        p = (self.ot_vocab * (1 - self.ot_del) * (self.pmi_so == 1))
        n = (self.ot_vocab * (1 - self.ot_del) * (self.pmi_so == 0))

        print('1:', p.sum(), '| 0:', n.sum())

    def annotate(self, doc):
        path_mat = create_path_mat(doc)
        words, simp_doc, simp_pat_mat = simplify_doc_and_path_mat(doc, path_mat)

        if len(words) == 0:
            return [], [], []

        x = self.vocab.list_word2id(words)
        mat = self.path_manager.mat_path2id(simp_pat_mat)

        Po, Pa, po, do, da = self(x, mat)[:5]

        Po = Po * (1-do)
        Pa = Pa * (1-da)

        opinions = [simp_doc[i] for i in range(len(x)) if Po[i]]
        aspects  = [simp_doc[i] for i in range(len(x)) if Pa[i]]
        polarity = [int(po[i])  for i in range(len(x)) if Po[i]]

        def convert_to_start_end(tokens):
            start_ends = []
            for token in tokens:
                start = token.idx
                end   = start + len(token.text)
                start_ends.extend((start, end))
            return start_ends

        opinions = convert_to_start_end(opinions)
        aspects  = convert_to_start_end(aspects)

        return opinions, aspects, polarity



def build_mapping(lst):
    _mapping = {}
    for i, item in enumerate(lst):
        _mapping[item] = i 

    def mapping(key):
        if key in _mapping:
            return _mapping[key]
        else:
            print(f'\t\t\t\t\t\t\t\t\t\t\t\t|{key}|')
            return -100

    return mapping



def lite(token_string, pos_mapping, dep_mapping):
    i, idx, text, lemma_, pos_, dep_, head_i, children_i, sent_id = json.loads(token_string)
    pos_id = pos_mapping(pos_)
    dep_id = dep_mapping(dep_)
    width  = len(text)
    # return json.dumps([i, idx, width, pos_id, dep_id, head_i])
    return json.dumps([idx, width, pos_id, head_i])



def main(args):
    vocab_file = os.path.join(args.vocab_path_dir, 'vocab.json')
    path_manager_file = os.path.join(args.vocab_path_dir, 'path_manager.json')

    vocab = Vocab()
    vocab.load(vocab_file)
    print('vocab', len(vocab))

    path_manager = PathManager()
    path_manager.load(path_manager_file)
    print('path', len(path_manager))

    annotator = Annotation(path_manager, vocab)
    annotator.load(args.annotator_file)
    annotator.load_polarity(args.polarity_file)
    annotator.stat()

    # data_files = {'100k_1_5.json': os.path.join(args.data_dir, '100k_1_5.json')}
    data_files = {name: file_name for name, file_name in yield_data_file(args.data_dir)}
    datasets = load_dataset('json', data_files=data_files, cache_dir=args.cache_dir)
    print(datasets)

    deps = ['ROOT', 'acl', 'acomp', 'advcl', 'advmod', 
            'agent', 'amod', 'appos', 'attr', 'aux', 
            'auxpass', 'case', 'cc', 'ccomp', 'compound', 
            'conj', 'csubj', 'csubjpass', 'dative', 'dep', 
            'det', 'dobj', 'expl', 'intj', 'mark', 
            'meta', 'neg', 'nmod', 'npadvmod', 'nsubj', 
            'nsubjpass', 'nummod', 'oprd', 'parataxis', 'pcomp', 
            'pobj', 'poss', 'preconj', 'predet', 'prep', 
            'prt', 'punct', 'quantmod', 'relcl', 'xcomp'] # 45

    poses = ['NOUN', 'ADJ', 'VERB', 'ADV', 'PART', 
             'PUNCT', 'CCONJ', 'PRON', 'DET', 'ADP', 
             'SCONJ', 'AUX', 'NUM', 'PROPN', 'SYM', 
             'INTJ', 'X', 'SPACE']

    pos_mapping = build_mapping(poses)
    dep_mapping = build_mapping(deps)

    def annotate_function(examples):
        opinions = []
        aspects  = []
        polarity = []

        for parsed in examples['parsed']:
            doc = Doc(parsed)
            o, a, p = annotator.annotate(doc)
            opinions.append(o)
            aspects.append(a)
            polarity.append(p)

        lite_parsed = [' '.join([lite(it, pos_mapping, dep_mapping) for it in parsed])
                       for parsed in examples['parsed']]

        return {'aspects': aspects, 'lite_parsed': lite_parsed}
        # 由于在最终的SPT方法中用不到opinion，polarity，因此不保存这些属性以减少文件体积
        # return {'aspects': aspects, 'lite_parsed': lite_parsed, 'opinions': opinions, 'polarity': polarity}

    # # test ##########################################################
    # from tqdm import tqdm
    # for parsed in tqdm(datasets['100k_1_5.json']['parsed']):
    #     doc = Doc(parsed)
    #     o, a, p, pos  = annotator.annotate(doc)
    # #################################################################

    annotated_datasets = datasets.map(
        annotate_function,
        batched=True,
        num_proc=args.num_proc,
        batch_size=32,
        remove_columns=['parsed']
    )
    print(annotated_datasets)

    print('save to', args.output_dir)
    _mkdir_if_not_exist(args.output_dir)
    for split in annotated_datasets:
        file_name = os.path.join(args.output_dir, split)
        annotated_datasets[split].to_json(file_name)



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--vocab_path_dir', type=str)
    parser.add_argument('--annotator_file', type=str)
    parser.add_argument('--polarity_file', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--cache_dir', type=str)
    parser.add_argument('--num_proc', type=int, default=64)

    args = parser.parse_args()
    args.cache_dir = os.path.join(args.cache_dir, 'load/processed.arrow')

    main(args)


"""
f'python annotation_12m25d.py --data_dir "/data10T/zhangyice/2022/sentiment pre-training/spacy_tokenized-6m28d/{domain}/" --cache_dir "/data10T/zhangyice/.cache/huggingface/datasets/tokenized-6m28d/{domain}/" --output_dir "/data10T/zhangyice/2022/sentiment pre-training/8m28/annotated/propagation_12m25d/{domain}/" --vocab_path_dir "/data10T/zhangyice/2022/sentiment pre-training/8m28d/lexicon_and_rule/{domain}" --annotator_file "/data10T/zhangyice/2022/sentiment pre-training/12m25d/lexicon_and_rule/main/{domain}/9/annotator.json" --polarity_file "/data10T/zhangyice/2022/sentiment pre-training/12m25d/polarity/{domain}/polarity.json"'
"""
