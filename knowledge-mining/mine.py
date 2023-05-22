import random
import os
import torch
from tqdm import tqdm
import numpy as np
# import scipy.stats as st

from transformers import AutoTokenizer

from utils import load_mpqa2, load_line_json, load_json, save_json
from utils.vocab import Vocab
from utils.path_manager import PathManager, create_path_mat



def load_doc_path_mat(input_file_name, max_examples):
    docs = []
    path_mats = []
    ratings = []
    for example in tqdm(load_line_json(input_file_name, max_examples), desc='Load-data', total=max_examples):
        doc = np.array(example['simp_doc'])
        path_mat = np.array(example['simp_path_mat'])
        if len(doc) > 0:
            docs.append(doc)
            # UNK的单词对应的路径也为UNK
            path_mat[doc==0] = 0
            path_mat[:, doc==0] = 0
            path_mats.append(path_mat)
            try:
                ratings.append(example['Overall'])
            except Exception as e:
                print('----', e)
                ratings.append(0)
    return docs, path_mats, ratings



def prepare_test_examples(test_file_name, nlp, path_manager, vocab):
    examples = load_json(test_file_name)

    all_words = set()
    all_opinions = set()
    all_aspects  = set() 

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    for example in examples:
        example['text'] = example['sentence'].replace('n \' t', 'n\'t').lower()
        
        example['opinions'] = [tokenizer.convert_tokens_to_string(eval(opinion[-1])).replace('n \' t', 'n\'t').lower() for opinion in example['opinions'] if len(opinion) == 3]
        example['aspects']  = [tokenizer.convert_tokens_to_string(eval(aspect[-1])).replace('n \' t', 'n\'t').lower() for aspect in example['aspects'] if len(aspect) != 0]
        example['doc'] = nlp(example['sentence'])


        example['x'] = vocab.list_word2id((' '.join((token.lemma_, token.pos_)) for token in example['doc']))
        example['path_mat'] = path_manager.mat_path2id(create_path_mat(example['doc']))

        example['path_mat'][example['x']==0] = 0
        example['path_mat'][:, example['x']==0] = 0

    return examples



class Annotation:
    def __init__(self, path_manager, vocab, at_入阈值p=0.5):
        """
        at_入阈值p=0.4 for lap
        at_入阈值p=0.5 for rest
        """
        self.path_manager = path_manager
        self.vocab = vocab
        self.is_adj  = np.array(([('ADJ'  in self.vocab[i]) for i in range(len(vocab))]))
        self.is_noun = np.array(([('NOUN' in self.vocab[i]) for i in range(len(vocab))]))

        self.min_word_count = 5
        self.min_path_count = 50

        # 入阈值越小，出阈值越大，opinion/aspect越少
        self.ot_出阈值  = 0.50
        self.at_出阈值  = 0.30

        self.ot_入阈值p = 0.7
        self.at_入阈值p = at_入阈值p

        self.oo_阈值 = 0.9
        self.oa_阈值 = 0.9
        self.aa_阈值 = 0.9
        self.ao_阈值 = 0.9

        self.gold_oo_paths = [
            ('False', 'ADJ', '>conj', 'ADJ'), 
            ('False', 'ADJ', '<conj', 'ADJ'),
            ('True',  'ADJ', '>conj', 'ADJ'),
            ('True',  'ADJ', '<conj', 'ADJ'),
            ('ADJ', 'self-loop', 'ADJ'),
        ]

        self.gold_oa_paths = [
            ('True', 'ADJ', '<acomp', 'AUX', '>nsubj', 'NOUN'),
            ('False','ADJ', '<acomp', 'AUX', '>nsubj', 'NOUN'),
        ]

        self.gold_aa_paths = [
            ('False', 'NOUN', '>conj', 'NOUN'), 
            ('False', 'NOUN', '<conj', 'NOUN'),
            ('True',  'NOUN', '>conj', 'NOUN'),
            ('True',  'NOUN', '<conj', 'NOUN'),
            ('NOUN', 'self-loop', 'NOUN')
        ]

        self.gold_ao_paths = [
            ('True', 'NOUN', '<nsubj', 'AUX', '>acomp', 'ADJ'),
            ('False','NOUN', '<nsubj', 'AUX', '>acomp', 'ADJ'),
        ]
        self.init_path_score()
        self.init_word_score()

    def load(self, file_name):
        data = load_json(file_name)

        for word in data['ot']:
            index = self.vocab[word]
            self.ot_vocab[index] = 1
        self.ot_vocab[0] = 0

        for word in data['at']:
            index = self.vocab[word]
            self.at_vocab[index] = 1
        self.at_vocab[0] = 0

        for word in data['ot_del']:
            index = self.vocab[word]
            self.ot_del[index] = 1

        for word in data['at_del']:
            index = self.vocab[word]
            self.at_del[index] = 1

        self.pmi_so = np.zeros(len(self.vocab))
        for word, polarity_score in data['pmi_so']:
            index = self.vocab[word]
            self.pmi_so[index] = polarity_score
        self.pmi_so[0] = 0

        for path_type in ('oo', 'oa', 'aa', 'ao'):
            for path in data[path_type]:
                path  = tuple(path) if type(path) is list else path 
                index = self.path_manager[path]
                getattr(self, f'{path_type}_path_score')[index] = 1

    def load_opinion(self, file_name):
        data = load_json(file_name)
        for word in data['ot']:
            index = self.vocab[word]
            self.ot_vocab[index] = 1
            self.ot_prob[index] = 1
        
    def save(self, file_name):
        ot_word = [self.vocab[i] for i in range(len(self.vocab)) if self.ot_vocab[i]*(1-self.ot_del[i])]
        at_word = [self.vocab[i] for i in range(len(self.vocab)) if self.at_vocab[i]*(1-self.at_del[i])]

        ot_del  = [self.vocab[i] for i in range(len(self.vocab)) if self.ot_del[i]]
        at_del  = [self.vocab[i] for i in range(len(self.vocab)) if self.at_del[i]]

        pmi_so = [(self.vocab[i], float(self.pmi_so[i])) for i in range(len(self.vocab)) if (1-self.ot_del[i])]

        oo_path = [self.path_manager[i] for i in range(len(self.path_manager)) if self.oo_path_score[i]]
        oa_path = [self.path_manager[i] for i in range(len(self.path_manager)) if self.oa_path_score[i]]
        aa_path = [self.path_manager[i] for i in range(len(self.path_manager)) if self.aa_path_score[i]]
        ao_path = [self.path_manager[i] for i in range(len(self.path_manager)) if self.ao_path_score[i]]

        data = {
            'ot': ot_word,
            'at': at_word,
            'ot_del': ot_del,
            'at_del': at_del,
            'pmi_so': pmi_so,
            'oo': oo_path,
            'oa': oa_path,
            'aa': aa_path,
            'ao': ao_path,
        }
        save_json(data, file_name)

    def _set_path_score(self, paths, path_score):
        for path in paths:
            if not self.path_manager.is_UNK(path):
                path_id = self.path_manager[path]
                path_score[path_id] = 1
        path_score[0] = 0
    
    def init_word_score(self):
        self.ot_vocab = np.zeros(len(self.vocab), dtype=int)
        self.ot_del   = np.zeros(len(self.vocab), dtype=int)
        self.ot_strongsubj = np.zeros(len(self.vocab), dtype=int)
        self.ot_weaksubj   = np.zeros(len(self.vocab), dtype=int)

        polarity_map = {'positive': 1, 'negative': -1, 'neutral': 0}
        for word, pos, intensity, priorpolarity in load_mpqa2('subjclueslen1-HLTEMNLP05.tff'):
            pos = pos.upper() if pos != 'adverb' else 'ADV'
            if pos in ('ADJ', ):
                word = ' '.join((word, pos))
                if not self.vocab.is_UNK(word):
                    word_id = self.vocab[word]
                    self.ot_vocab[word_id] = 1

        self.at_vocab  = np.zeros(len(self.vocab), dtype=int)
        self.at_del    = np.zeros(len(self.vocab), dtype=int)

    def init_path_score(self):
        self.oo_path_score = np.zeros(len(self.path_manager), dtype=int)
        self._set_path_score(self.gold_oo_paths, self.oo_path_score)

        self.oa_path_score = np.zeros(len(self.path_manager), dtype=int)
        self._set_path_score(self.gold_oa_paths, self.oa_path_score)

        self.aa_path_score = np.zeros(len(self.path_manager), dtype=int)
        self._set_path_score(self.gold_aa_paths, self.aa_path_score)

        self.ao_path_score = np.zeros(len(self.path_manager), dtype=int)
        self._set_path_score(self.gold_ao_paths, self.ao_path_score)

    def __call__(self, x, path_mat):
        """
        Po: 是否为情感词
        Pa: 是否为方面词
        po: 情感得分
        do: 不是情感词的词
        da: 不是方面词的词
        yo, ya, Too, Toa, Taa, Tao: 辅助信息
        """
        yo = self.ot_vocab[x]
        do = self.ot_del[x]
        yo = yo * (1 - do)
        po = self.pmi_so[x]

        ya = self.at_vocab[x]
        da = self.at_del[x]
        ya = ya * (1 - da)

        Too = np.take(self.oo_path_score, path_mat, axis=0)
        Toa = np.take(self.oa_path_score, path_mat, axis=0)
        Taa = np.take(self.aa_path_score, path_mat, axis=0)
        Tao = np.take(self.ao_path_score, path_mat, axis=0)

        Pa = yo @ Toa + ya @ Taa
        Po = yo @ Too + ya @ Tao

        return Po, Pa, po, do, da, yo, ya, Too, Toa, Taa, Tao

    def annotate(self, example):
        x = example['x']
        path_mat = example['path_mat']
        doc = example['doc']
        n = len(x)

        Po, Pa, po, do, da, yo, ya, Too, Toa, Taa, Tao = self(x, path_mat)

        Po = Po * (1-do)
        Pa = Pa * (1-da)
        po[po> 0.2] =  1
        po[po<-0.2] = -1
        po[(-0.2<=po) * (po<=0.2)] = 0

        opinions = [doc[i].text for i in range(n) if Po[i]]
        aspects  = [doc[i].text for i in range(n) if Pa[i]]
        polarity = [po[i] for i in range(n) if Po[i]]

        return opinions, aspects, polarity

    def get_word_count(self, docs, eps=1e-6):
        word_count  = np.zeros(len(self.vocab), dtype=float)
        for x in tqdm(docs, total=len(docs), desc='Stat-Word-Count'):
            word_count[x] += 1

        word_count[word_count==0] = eps
        return word_count

    def update(self, docs, path_mats, iter_num, word_count):
        ot_evidence = np.zeros(len(self.vocab), dtype=int)
        at_evidence = np.zeros(len(self.vocab), dtype=int)

        oo_出边 = np.zeros(len(self.vocab), dtype=int)
        aa_出边 = np.zeros(len(self.vocab), dtype=int)

        oops_分子 = np.zeros(len(self.path_manager), dtype=int)
        oaps_分子 = np.zeros(len(self.path_manager), dtype=int)
        oaps_oops_分母 = np.zeros(len(self.path_manager), dtype=int)

        aaps_分子 = np.zeros(len(self.path_manager), dtype=int)
        aops_分子 = np.zeros(len(self.path_manager), dtype=int)
        aops_aaps_分母 = np.zeros(len(self.path_manager), dtype=int)

        for x, path_mat in tqdm(zip(docs, path_mats), total=len(docs), desc='Update'):
            n = len(x)
            Po, Pa, po, do, da, yo, ya, Too, Toa, Taa, Tao = self(x, path_mat)

            oo_出边[x] += (Too.sum(axis=1)-1) > 0 # -1 for self-loop
            ot_evidence[x] += (Po - yo) > 0
            
            aa_出边[x] += (Taa.sum(axis=1)-1) > 0 # -1 for self-loop
            at_evidence[x] += (Pa - ya) > 0

            Po = Po > 0
            Pa = Pa > 0
            for i in range(n):
                pline = path_mat[i]

                oops_分子[pline] += Po[i] * Po
                oaps_分子[pline] += Po[i] * Pa
                oaps_oops_分母[pline] += Po[i]

                aaps_分子[pline] += Pa[i] * Pa
                aops_分子[pline] += Pa[i] * Po
                aops_aaps_分母[pline] += Pa[i]

        ################# add-opinion #################
        oo_出边[oo_出边==0] = 1
        ot_score = ot_evidence / word_count

        self.ot_del = ((word_count < self.min_word_count) + (ot_evidence / oo_出边 < self.ot_出阈值) + (ot_score==0)).astype(int)
        self.ot_del[~self.is_adj] = 1

        exist_ot = ot_score[(self.ot_vocab==1) * (self.ot_del==0)]
        thre = exist_ot.mean() * self.ot_入阈值p

        add_words = [i for i in range(len(self.vocab)) if ot_score[i] >= thre and self.ot_del[i] == 0 and self.ot_vocab[i] == 0]
        self.ot_vocab[add_words] = 1
        print('ot-thre', thre, exist_ot.mean(), len(add_words))

        # case
        strings = "great ok white golden red black yellow able right real sure traditional bloody unable least true hidden similar microphone japanese greek korean southern homemade accesible viewable"
        for word in strings.split():
            i = self.vocab[f'{word} ADJ']
            print(f'{i:5d}', '_' * (12-len(word)) + word, self.ot_vocab[i], ot_score[i], self.ot_del[i], self.pmi_so[i], word_count[i])

        ################# add-aspect #################
        aa_出边[aa_出边==0] = 1
        at_score = at_evidence / word_count

        self.at_del = ((word_count < self.min_word_count) + (at_evidence / aa_出边 < self.at_出阈值) + (at_score==0)).astype(int)
        self.at_del[~self.is_noun] = 1

        if not ((self.at_vocab==1) * (self.at_del==0)).any():
            N = (self.ot_vocab * (1-self.ot_del)).sum()
            k = int(N * 0.6)
            at_no_del = at_score[self.at_del==0]
            at_入阈值 = sorted(at_no_del, reverse=True)[k-1]
            add_words = [i for i in range(len(self.vocab)) if at_score[i] >= at_入阈值 and self.at_del[i] == 0]
            print('at-thre', at_入阈值, len(add_words))

        else:
            exist_at = at_score[(self.at_vocab==1) * (self.at_del==0)]
            thre = exist_at.mean() * self.at_入阈值p
            add_words = [i for i in range(len(self.vocab)) if at_score[i] >= thre and self.at_del[i] == 0 and self.at_vocab[i] == 0]
            print('at-thre', thre, exist_at.mean(), len(add_words))
        
        self.at_vocab[add_words] = 1

        ################# add-path ##################
        oaps_oops_分母[oaps_oops_分母==0] = 1
        aops_aaps_分母[aops_aaps_分母==0] = 1

        def 根据阈值和topk来选择路径(分子, 分母, 阈值, k=500):
            path_score = 分子/分母
            path_score[分母<self.min_path_count] = 0

            topk = sorted(path_score, reverse=True)[k-1]
            print(阈值, topk)
            # 阈值 = max(topk, 阈值)
            path_score[path_score>=阈值] = 1
            path_score[path_score< 阈值] = 0
            return path_score.astype(int)
        
        self.oo_path_score = 根据阈值和topk来选择路径(oops_分子, oaps_oops_分母, self.oo_阈值)
        self.oa_path_score = 根据阈值和topk来选择路径(oaps_分子, oaps_oops_分母, self.oa_阈值)
        self.aa_path_score = 根据阈值和topk来选择路径(aaps_分子, aops_aaps_分母, self.aa_阈值)
        self.ao_path_score = 根据阈值和topk来选择路径(aops_分子, aops_aaps_分母, self.ao_阈值)

        self._set_path_score(self.gold_oo_paths, self.oo_path_score)
        self._set_path_score(self.gold_oa_paths, self.oa_path_score)
        self._set_path_score(self.gold_aa_paths, self.aa_path_score)
        self._set_path_score(self.gold_ao_paths, self.ao_path_score)

    def stat(self):
        val_opinion = (self.ot_vocab * (1-self.ot_del)).sum()
        val_aspect  = (self.at_vocab * (1-self.at_del)).sum()

        print()
        print('[VOCAB] | ot:', val_opinion, self.ot_vocab.sum(), self.ot_del.sum(), 
                      '| at:', val_aspect, self.at_vocab.sum(), self.at_del.sum())
        print('[PATH]  | oo:', self.oo_path_score.sum(), '| oa:', self.oa_path_score.sum())
        print('        | aa:', self.aa_path_score.sum(), '| ao:', self.ao_path_score.sum())
        print()



def evaluate(annotator, examples, **kwargs):
    all_o_s4 = []
    all_a_s4 = []
    result = {}
    for example in examples:
        opinions = example['opinions']
        aspects  = example['aspects']
        annotated_opinions, annotated_aspects = annotator.annotate(example, **kwargs)[:2]
        o_s4 = _cal_recall_precision(opinions, annotated_opinions)
        a_s4 = _cal_recall_precision(aspects,  annotated_aspects)

        ph,ps,rh,rs = o_s4

        all_o_s4.append(o_s4)
        all_a_s4.append(a_s4)

    def _prec_recall_f1(all_s4):
        prec_hit = sum([s4[0] for s4 in all_s4])
        prec_sum = sum([s4[1] for s4 in all_s4])
        recall_hit = sum([s4[2] for s4 in all_s4])
        recall_sum = sum([s4[3] for s4 in all_s4])

        prec = prec_hit / prec_sum if prec_sum > 0 else 0.
        recall = recall_hit / recall_sum if recall_sum > 0 else 0.
        f1 = 2 * prec * recall / (prec + recall) if (prec + recall) > 0 else 0.
        return prec, recall, f1

    o_prec, o_recall, o_f1 = _prec_recall_f1(all_o_s4)
    a_prec, a_recall, a_f1 = _prec_recall_f1(all_a_s4)

    print('----------------------[ Overlap-f1 ]-----------------------')
    print(f'opinion | precision: {o_prec*100:.2f} | recall: {o_recall*100:.2f} | f1-score: {o_f1*100:.2f}')
    print(f'aspect  | precision: {a_prec*100:.2f} | recall: {a_recall*100:.2f} | f1-score: {a_f1*100:.2f}')

    return result



def _cal_recall_precision(gold, pred):
    prec_hit = 0
    prec_sum = 0
    for p in pred:
        prec_sum += 1
        for g in gold:
            if p in g:
                prec_hit += 1
                break

    recall_hit = 0
    recall_sum = 0
    for g in gold:
        recall_sum += 1
        for p in pred:
            if p in g:
                recall_hit += 1
                break

    return prec_hit, prec_sum, recall_hit, recall_sum




if __name__ == '__main__':
    import argparse
    import spacy
    import random

    parser = argparse.ArgumentParser()

    parser.add_argument('--train_dir',      type=str)
    parser.add_argument('--test_file_name', type=str)
    parser.add_argument('--output_dir',     type=str)
    parser.add_argument('--max_examples',   type=int, default=500_000)
    parser.add_argument('--n_iter',  type=int, default=10)
    parser.add_argument('--seed',    type=int, default=42)
    parser.add_argument('--at_thre', type=float, default=0.5, help="0.5 for rest, 0.4 for other domain")

    args = parser.parse_args()
    random.seed(args.seed)
    np.set_printoptions(linewidth=300, precision=2)
    nlp = spacy.load('en_core_web_sm')

    vocab_dir = os.path.join(args.train_dir, 'vocab.json')
    path_manager_dir = os.path.join(args.train_dir, 'path_manager.json')
    data_dir = os.path.join(args.train_dir, 'data.json')

    vocab = Vocab()
    vocab.load(vocab_dir)
    print('vocab', len(vocab))

    path_manager = PathManager()
    path_manager.load(path_manager_dir)
    print('path', len(path_manager))
    
    annotator = Annotation(path_manager=path_manager, vocab=vocab, at_入阈值p=args.at_thre)
    annotator.stat()

    test_examples = prepare_test_examples(args.test_file_name, nlp, path_manager, vocab)

    train_docs, train_path_mats, ratings = load_doc_path_mat(data_dir, max_examples=args.max_examples)
    word_count = annotator.get_word_count(train_docs)

    evaluate(annotator, test_examples)

    for i in range(args.n_iter):
        print(f'--------------{i}------------------')
        annotator.update(train_docs, train_path_mats, i, word_count)
        annotator.stat()
        print('save to', os.path.join(args.output_dir, str(i), 'annotator.json'))
        annotator.save(os.path.join(args.output_dir, str(i), 'annotator.json'))
        evaluate(annotator, test_examples)