import random
import os
import torch
from tqdm import tqdm
import numpy as np
# import scipy.stats as st

from utils import load_mpqa2, load_line_json, save_json, yield_data_file
from utils.vocab import Vocab



def load_doc(input_file_names, max_examples=500_000):
    docs = []
    ratings = []

    for input_file_name in input_file_names:
        for example in tqdm(load_line_json(input_file_name, max_examples), desc='Load-data', total=max_examples):
            doc = np.array(example['simp_doc'])
            if len(doc) > 0:
                docs.append(doc)
                ratings.append(example['Overall'])

    return docs, ratings



class Annotation:
    def __init__(self, vocab):
        self.vocab = vocab
        self.is_adj  = np.array(([('ADJ'  in self.vocab[i]) for i in range(len(vocab))]))
        self.is_noun = np.array(([('NOUN' in self.vocab[i]) for i in range(len(vocab))]))

    def save(self, file_name, thre=0.2):
        positive = [self.vocab.toword(i) for i in range(len(self.vocab)) if self.pmi_so[i]>= thre]
        negative = [self.vocab.toword(i) for i in range(len(self.vocab)) if self.pmi_so[i]<=-thre]

        data = {
            'positive': positive,
            'negative': negative
        }
        print(len(data['positive']), len(data['negative']))
        save_json(data, file_name)
    
    def init_word_score(self):
        self.train = np.zeros(len(self.vocab), dtype=int)

        polarity_map = {'positive': 1, 'negative': -1, 'neutral': 0}
        for word, pos, intensity, priorpolarity in load_mpqa2('subjclueslen1-HLTEMNLP05.tff'):
            pos = pos.upper() if pos != 'adverb' else 'ADV'
            word = ' '.join((word, pos))
            if pos in ('ADJ', ) and not self.vocab.is_UNK(word):
                word_id = self.vocab[word]
                self.train[word_id] = polarity_map[priorpolarity]

        self.test = self.train

    def get_word_count(self, docs, eps=1e-6):
        word_count  = np.zeros(len(self.vocab), dtype=float)
        for x in tqdm(docs, total=len(docs), desc='Stat-Word-Count'):
            word_count[x] += 1

        word_count[word_count==0] = eps
        return word_count

    def eval_pmi_so(self, eps=0.2):
        accuracy = np.array([(self.pmi_so[i]>0 and self.test[i]>0) or (self.pmi_so[i]<0 and self.test[i]<0)
                             for i in range(len(self.vocab)) 
                             if self.test[i] != 0 and abs(self.pmi_so[i]) >= eps])
        print('eps=0.2', accuracy.mean(), accuracy.shape, accuracy[:100].mean(), accuracy[:200].mean(), accuracy[:500].mean())

        accuracy = np.array([(self.pmi_so[i]>0 and self.test[i]>0) or (self.pmi_so[i]<0 and self.test[i]<0)
                             for i in range(len(self.vocab))
                             if self.test[i] != 0])
        print('eps=0', accuracy.mean(), accuracy.shape, accuracy[:100].mean(), accuracy[:200].mean(), accuracy[:500].mean())

    def polarity_assign(self, docs, ratings, word_count):
        pos_word_count = np.zeros(len(self.vocab), dtype=int)
        neg_word_count = np.zeros(len(self.vocab), dtype=int)
        pos_num = neg_num = 0

        # sentiment_dict = {1.: 0, 2.: 1, 3.: 2, 4.: 3, 5.: 4, 0.: -100}
        for x, rating in tqdm(zip(docs, ratings), desc='assign polarity1', total=len(docs)):
            if rating >= 4:
                pos_word_count[x] += 1
                pos_num += 1
            elif 0 < rating <= 2:
                neg_word_count[x] += 1
                neg_num += 1
        
        """
        方案1：
        根据评论的情感得分来挖掘单词的情感倾向
            pmi(w1,p) = log p(w1|p) / p(w1) 
            pmi(w1,n) = log p(w1|n) / p(w1)
            pmi-so(w1) = pmi(w1,p) - pmi(w1,n) = log p(w1|p) / p(w1|n) = log count(w1,p)count(n) / count(w1,n)count(p)
            +1 平滑, 假设存在两篇文档包含所有的词，情感类别分别为正面和负面
        """
        
        alpha = neg_num / pos_num # +1 平滑
        self.pmi_so = ((pos_word_count+1)*(neg_num+alpha)) / ((neg_word_count+alpha)*(pos_num+1))
        self.pmi_so = np.log(self.pmi_so)
        self.pmi_so[~self.is_adj] = 0

        self.eval_pmi_so()

        """
        方案2：
        根据单词的共现 来计算单词的情感倾向
            将MPQA中的正面词和负面词作为种子词。
            一个单词只要与正面种子词共现，则视作出现在正面评论中。
            一个单词只要与负面种子词共现，则视作出现在负面评论中。
        """

        pos_anchors = np.array([i for i in range(len(self.vocab)) if self.train[i]== 1])
        neg_anchors = np.array([i for i in range(len(self.vocab)) if self.train[i]==-1])
        
        set_pos_anchors = set(pos_anchors)
        set_neg_anchors = set(neg_anchors)
        print(len(set_pos_anchors), len(set_neg_anchors))

        self.pmi_so_old = self.pmi_so.copy()

        pos_word_count = np.zeros(len(self.vocab), dtype=int)
        neg_word_count = np.zeros(len(self.vocab), dtype=int)
        pos_num = neg_num = 0
        
        for x, rating in tqdm(zip(docs, ratings), desc='assign polarity2', total=len(docs)):
            if set_pos_anchors.intersection(x):
                pos_word_count[x] += 1
                pos_num += 1
            if set_neg_anchors.intersection(x):
                neg_word_count[x] += 1
                neg_num += 1

        alpha = neg_num / pos_num
        self.pmi_so = ((pos_word_count+1)*(neg_num+alpha)) / ((neg_word_count+alpha)*(pos_num+1))
        self.pmi_so[self.is_adj]  = np.log(self.pmi_so[self.is_adj])
        self.pmi_so[~self.is_adj] = 0
        self.pmi_so[(pos_word_count+neg_word_count)<10] = 0

        self.eval_pmi_so()

        # case
        for word in "interesting boring good pretty mindful friendly approachable low cacual cool awful horrible disgusting authentic ok milky displeased large weak empty clean negative positive nice".split():
            i = self.vocab[f'{word} ADJ']
            print('_' * (12-len(word))+word, '|', word_count[i], self.pmi_so[i], self.pmi_so_old[i], pos_word_count[i], neg_word_count[i])



if __name__ == '__main__':
    import argparse
    import spacy
    import random

    parser = argparse.ArgumentParser()

    parser.add_argument('--train_dir', type=str)
    parser.add_argument('--output_dir',   type=str)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    random.seed(args.seed)
    np.set_printoptions(linewidth=300, precision=2)
    nlp = spacy.load('en_core_web_sm')

    vocab_dir = os.path.join(args.train_dir, 'vocab.json')
    vocab = Vocab()
    vocab.load(vocab_dir)
    print('vocab', len(vocab))
    
    annotator = Annotation(vocab=vocab)
    annotator.init_word_score()

    file_names = [file_name for name, file_name in yield_data_file(args.train_dir) if name != 'vocab.json']

    train_docs, ratings = load_doc(file_names)
    word_count = annotator.get_word_count(train_docs)

    annotator.polarity_assign(train_docs, ratings, word_count)
    file_name = os.path.join(args.output_dir, 'polarity.json')
    print('save to', file_name)
    annotator.save(file_name)
