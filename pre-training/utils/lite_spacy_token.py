import ujson as json



class Token:
    def __init__(self, i, pos_, head_i, idx=None, width=None, start=None, end=None):
        self.i = i
        self.idx = idx
        self.width = width
        self.pos_ = pos_
        self.head_i = head_i
        self.start = start
        self.end   = end

        self.head = None
        self.children = None

    def set_head(self, tokens):
        self.head = tokens[self.head_i]

    def char_start_end2(self, func):
        self.start = func(self.idx)
        self.end   = func(self.idx+self.width)

    def to_string(self):
        return json.dumps([self.start, self.end, self.pos_, self.head_i])





class Doc:
    def __init__(self, token_strings, mode='default'):
        self.tokens = []
        for i, token_string in enumerate(token_strings.split()):
            token = self.parse_token(i, token_string, mode)
            self.tokens.append(token)

        for token in self.tokens:
            token.set_head(self.tokens)

    def parse_token(self, i, token_string, mode):
        if mode == 'default':
            idx, width, pos_, head_i = json.loads(token_string)
            return Token(i, idx=int(idx), width=int(width), pos_=int(pos_), head_i=int(head_i))

        elif mode == 'char':
            start, end, pos_, head_i = json.loads(token_string)
            return Token(i, start=int(start), end=int(end), pos_=int(pos_), head_i=int(head_i)) 

    def char_to_token2(self, func):
        for token in self.tokens:
            token.char_start_end2(func)

    def __getitem__(self, i):
        return self.tokens[i]

    def __len__(self):
        return len(self.tokens)

    def __iter__(self):
        for token in self.tokens:
            yield token 

    def to_string(self):
        return ' '.join([token.to_string() for token in self.tokens])



def yield_dep_distance2(doc, max_distance=10):
    for token in doc:
        for ancestor, distance in yield_ancestor(token, max_distance):
            yield token.start, ancestor.start, distance




def yield_ancestor(token, max_distance=10):
    ancestor = token
    distance = 0
    while True:
        # if ancestor.dep_ == ROOT_id:  # 0
        if ancestor.i == ancestor.head.i:
            break
        ancestor = ancestor.head
        distance += 1

        if distance > max_distance:
            break

        yield ancestor, distance

