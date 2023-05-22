from collections import Counter
import numpy as np



class PathManager:
    """
    manager for dependency paths
    """
    def __init__(self, max_length=25, min_count=1):
        self.max_length = max_length
        self.min_count  = min_count

        self._path2id = {'UNK': 0}
        self._id2path = ['UNK']
        self._counter = [0]

    def save(self, file_name):
        from . import save_json
        
        data = {
            'id2path': self._id2path,
            'counter': self._counter
        }
        save_json(data, file_name)

    def load(self, file_name):
        from . import load_json

        tuple_ = lambda path: (path if type(path) is str else tuple(path))

        data = load_json(file_name)
        self._id2path = [tuple_(path) for path in data['id2path']]
        self._path2id = {tuple_(path): i for i, path in enumerate(self._id2path)}
        self._counter = data['counter']

    def path2id(self, path):
        if path not in self._path2id:
            path = 'UNK'
        return self._path2id[path]

    def id2path(self, id_):
        if id_ >= len(self._id2path):
            raise Exception('Unknown path id.')

        return self._id2path[id_]

    def count(self, path_or_id):
        if self.is_UNK(path_or_id):
            return 0

        id_ = self[path_or_id] if type(path_or_id) is tuple else path_or_id
        return self._counter[id_]

    def mat_path2id(self, path_mat):
        return np.array([[
            self.path2id(tuple(path_mat[i][j]))
            for j in range(len(path_mat))]
            for i in range(len(path_mat))])
        # return np.take(self._path2id, path_mat)

    def __getitem__(self, index):
        if type(index) is int:
            return self.id2path(index)

        elif type(index) is tuple:
            return self.path2id(index)

        raise NotImplementedError(f'Unspported dtype: {type(index)}.')

    def __len__(self):
        return len(self._id2path)

    def is_UNK(self, path_or_id):
        if type(path_or_id) is int:
            return path_or_id == 0

        elif type(path_or_id) is tuple:
            return path_or_id not in self._path2id

        elif type(path_or_id) is str:
            return path_or_id == 'UNK'

        raise NotImplementedError(f'Unspported dtype: {type(index)}.')

    def _add_new_path(self, path, count):
        if path not in self._path2id:
            self._path2id[path] = len(self._path2id)
            self._id2path.append(path)
            self._counter.append(count)

            assert self._id2path[self._path2id[path]] == path

    def build(self, path_mats):
        def yield_3d_list(list_3d):
            for list_2d in list_3d:
                for list_1d in list_2d:
                    for itm in list_1d:
                        if 0 < len(itm) <= self.max_length:
                            yield tuple(itm)

        print('path-counting')
        counter = Counter(yield_3d_list(path_mats))
        print('counting-over')

        for path, count in counter.items():
            if count >= self.min_count:
                self._add_new_path(path, count)



def create_path_mat(doc):
    """
    为 文档对象 建立 路径矩阵
    """
    n = len(doc)
    path_mat = [[tuple() for i in range(n)] for j in range(n)]

    for token in doc:
        if token.i != token.head.i:
            path_mat[token.i][token.head.i] = ('<' + token.dep_, token.head.pos_)
            path_mat[token.head.i][token.i] = ('>' + token.dep_, token.pos_)

    for k in range(n):
        for i in range(n):
            for j in range(n):
                if (i != j and 
                    len(path_mat[i][j]) == 0 and
                    len(path_mat[i][k]) != 0 and
                    len(path_mat[k][j]) != 0):

                    path_mat[i][j] = path_mat[i][k] + path_mat[k][j]

    for i in range(n):
        for j in range(n):
            if i != j:
                path = (str(i<j), doc[i].pos_) + path_mat[i][j] if len(path_mat[i][j]) > 0 else tuple()
            else:
                path = (doc[i].pos_, 'self-loop', doc[i].pos_)

            path_mat[i][j] = path

    return path_mat



def simplify_doc_and_path_mat(doc, path_mat, available_postags={'ADJ', 'ADV', 'VERB', 'NOUN'}):
    """
    只保留 特定词性集 中的词
    """
    available_indices = [token.i for token in doc if token.pos_ in available_postags]
    simp_doc = [' '.join((token.lemma_, token.pos_)) for token in doc if token.i in available_indices]
    simp_pat_mat = [[path_mat[i][j] for j in available_indices] for i in available_indices]
    return simp_doc, simp_pat_mat




if __name__ == '__main__':
    import spacy
    from tqdm import tqdm

    nlp = spacy.load('en_core_web_sm')

    N = 100

    def test_normal_python():
        
        texts = ['certainly not the best sushi in New York , however , it is always fresh , and the place is very clean , sterile .'
             for i in range(N)]
        docs = [nlp(text) for text in tqdm(texts, desc='Spacy')]
        path_mats = [create_path_mat(doc) for doc in tqdm(docs, desc='Create-path-mat')]
        return docs, path_mats

    test_normal_python()
