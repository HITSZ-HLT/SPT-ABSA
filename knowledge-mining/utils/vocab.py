import numpy as np 
from collections import Counter



class Vocab:
    """
    mapping between word-pos and id
    """
    def __init__(self, min_count=1):
        self.min_count = min_count
        self._word2id  = {'UNK': 0}
        self._id2word  = ['UNK']
        self._counter  = [0]

    def save(self, file_name):
        from . import save_json
        
        data = {
            'id2word': self._id2word,
            'counter': self._counter
        }
        save_json(data, file_name)

    def load(self, file_name):
        from . import load_json

        data = load_json(file_name)
        self._id2word = data['id2word']
        self._word2id = {word: i for i, word in enumerate(self._id2word)}
        self._counter = data['counter']

    def word2id(self, word):
        if word not in self._word2id:
            word = 'UNK'
        return self._word2id[word]

    def id2word(self, id_):
        if id_ >= len(self._id2word):
            raise Exception('Unknown word id.')
        return self._id2word[id_]

    def toword(self, id_):
        return self.id2word(id_).split()[0]

    def count(self, word_or_id):
        if self.is_UNK(word_or_id):
            return 0

        id_ = self[word_or_id] if type(word_or_id) is str else word_or_id
        return self._counter[id_]

    def list_word2id(self, doc):
        return np.array([self.word2id(word) for word in doc])
        # return np.take(self.word2id, x)

    def list_id2word(self, x):
        return ' '.join([self.id2word(_x).split()[0] for _x in x])

    def __getitem__(self, index):
        if type(index) is int:
            return self.id2word(index)

        elif type(index) is str:
            return self.word2id(index)

        raise NotImplementedError(f'Unspported dtype: {type(index)}.')

    def __len__(self):
        return len(self._id2word)

    def is_UNK(self, word_or_id):
        if type(word_or_id) is int:
            return word_or_id == 0

        elif type(word_or_id) is str:
            return word_or_id not in self._word2id or word_or_id == 'UNK'

        raise NotImplementedError(f'Unspported dtype: {type(word_or_id)}.')

    def _add_new_word(self, word, count):
        if word not in self._word2id:
            self._word2id[word] = len(self._word2id)
            self._id2word.append(word)
            self._counter.append(count)

            assert self._id2word[self._word2id[word]] == word

    def build(self, docs):
        def yield_2d_list(list_2d):
            for list_1d in list_2d:
                for itm in list_1d:
                    yield itm

        counter = Counter(yield_2d_list(docs))
        for word, count in counter.items():
            if count >= self.min_count:
                self._add_new_word(word, count)

    def build2(self, datasets, key):
        def yield_2d_list(list_2d):
            for list_1d in list_2d:
                for itm in list_1d:
                    yield itm

        def yield_3d_list():
            for split in tqdm(datasets.keys()):
                list_2d = datasets[split][key]
                yield from yield_2d_list(list_2d)

        counter = Counter(yield_3d_list())
        for word, count in counter.items():
            if count >= self.min_count:
                self._add_new_word(word, count)



if __name__ == '__main__':
    import spacy
    from tqdm import tqdm

    nlp = spacy.load('en_core_web_sm')

    vocab = Vocab(min_count=5)

    N = 100
    texts = ['I trust my apple greatly.' for i in range(N)]
    docs = [[token.text for token in nlp(text)] 
             for text in tqdm(texts, desc='Spacy')]

    # print(docs)
    vocab.build(docs)

    print(vocab)
    print(vocab._word2id)

    
