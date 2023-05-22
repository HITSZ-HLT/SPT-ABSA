import ujson as json



class Token:
    def __init__(self, i, idx, text, lemma_, pos_, dep_, head_i, children_i, sent_id):
        self.i = i
        self.idx = idx
        self.text = text
        self.lemma_ = lemma_
        self.pos_ = pos_
        self.dep_ = dep_
        self.head_i = head_i
        self.children_i = children_i

        self.head = None
        self.children = None

    def set_head(self, tokens):
        self.head = tokens[self.head_i]

    def set_children(self, tokens):
        self.children = [tokens[ci] for ci in self.children_i]

    def char_start_end(self, func):
        self.start = func(self.i)
        self.end   = func(self.i+len(self.text))

    def __repr__(self):
        return self.text



class Doc:
    def __init__(self, token_strings):
        self.tokens = []
        for token_string in token_strings:
            i, idx, text, lemma_, pos_, dep_, head_i, children_i, sent_id = json.loads(token_string)
            token = Token(int(i), int(idx), text, lemma_, pos_, dep_, int(head_i), children_i, sent_id)
            self.tokens.append(token)

        for token in self.tokens:
            token.set_head(self.tokens)
            token.set_children(self.tokens)

    def __getitem__(self, i):
        return self.tokens[i]

    def __len__(self):
        return len(self.tokens)

    def __iter__(self):
        for token in self.tokens:
            yield token 

    def __repr__(self):
        return ' '.join([token.text for token in self.tokens])

    def __str__(self):
        return ' '.join([token.text for token in self.tokens])     



if __name__ == '__main__':
    example = {"ID":210000,"Overall":4.0,"Text":"great service and the food was hot and fresh. onion rings rocked. shrimp lite breading and fresh tasting. the catfish was good size pieces. over all a great value for the price. a must do for a quick meal on the run.","parsed":["[0, 0, \"great\", \"great\", \"JJ\", \"amod\", 1, [], 29]","[1, 6, \"service\", \"service\", \"NN\", \"nsubj\", 5, [0, 2, 4], 29]","[2, 14, \"and\", \"and\", \"CC\", \"cc\", 1, [], 29]","[3, 18, \"the\", \"the\", \"DT\", \"det\", 4, [], 29]","[4, 22, \"food\", \"food\", \"NN\", \"conj\", 1, [3], 29]","[5, 27, \"was\", \"be\", \"VBD\", \"ROOT\", 5, [1, 6, 9], 29]","[6, 31, \"hot\", \"hot\", \"JJ\", \"acomp\", 5, [7, 8], 29]","[7, 35, \"and\", \"and\", \"CC\", \"cc\", 6, [], 29]","[8, 39, \"fresh\", \"fresh\", \"JJ\", \"conj\", 6, [], 29]","[9, 44, \".\", \".\", \".\", \"punct\", 5, [], 29]","[10, 46, \"onion\", \"onion\", \"NN\", \"compound\", 11, [], 53]","[11, 52, \"rings\", \"ring\", \"NNS\", \"ROOT\", 11, [10, 12, 13], 53]","[12, 58, \"rocked\", \"rock\", \"VBN\", \"acl\", 11, [], 53]","[13, 64, \".\", \".\", \".\", \"punct\", 11, [], 53]","[14, 66, \"shrimp\", \"shrimp\", \"NN\", \"compound\", 15, [], 96]","[15, 73, \"lite\", \"lite\", \"NN\", \"compound\", 16, [14], 96]","[16, 78, \"breading\", \"breading\", \"NN\", \"ROOT\", 16, [15, 17, 19, 20], 96]","[17, 87, \"and\", \"and\", \"CC\", \"cc\", 16, [], 96]","[18, 91, \"fresh\", \"fresh\", \"JJ\", \"amod\", 19, [], 96]","[19, 97, \"tasting\", \"tasting\", \"NN\", \"conj\", 16, [18], 96]","[20, 104, \".\", \".\", \".\", \"punct\", 16, [], 96]","[21, 106, \"the\", \"the\", \"DT\", \"det\", 22, [], 86]","[22, 110, \"catfish\", \"catfish\", \"NN\", \"nsubj\", 23, [21], 86]","[23, 118, \"was\", \"be\", \"VBD\", \"ROOT\", 23, [22, 26, 27], 86]","[24, 122, \"good\", \"good\", \"JJ\", \"amod\", 26, [], 86]","[25, 127, \"size\", \"size\", \"NN\", \"compound\", 26, [], 86]","[26, 132, \"pieces\", \"piece\", \"NNS\", \"attr\", 23, [24, 25], 86]","[27, 138, \".\", \".\", \".\", \"punct\", 23, [], 86]","[28, 140, \"over\", \"over\", \"IN\", \"ROOT\", 28, [32, 36], 74]","[29, 145, \"all\", \"all\", \"PDT\", \"predet\", 32, [], 74]","[30, 149, \"a\", \"a\", \"DT\", \"det\", 32, [], 74]","[31, 151, \"great\", \"great\", \"JJ\", \"amod\", 32, [], 74]","[32, 157, \"value\", \"value\", \"NN\", \"pobj\", 28, [29, 30, 31, 33], 74]","[33, 163, \"for\", \"for\", \"IN\", \"prep\", 32, [35], 74]","[34, 167, \"the\", \"the\", \"DT\", \"det\", 35, [], 74]","[35, 171, \"price\", \"price\", \"NN\", \"pobj\", 33, [34], 74]","[36, 176, \".\", \".\", \".\", \"punct\", 28, [], 74]","[37, 178, \"a\", \"a\", \"DT\", \"nsubj\", 39, [], 39]","[38, 180, \"must\", \"must\", \"MD\", \"aux\", 39, [], 39]","[39, 185, \"do\", \"do\", \"VB\", \"ROOT\", 39, [37, 38, 40, 47], 39]","[40, 188, \"for\", \"for\", \"IN\", \"prep\", 39, [43], 39]","[41, 192, \"a\", \"a\", \"DT\", \"det\", 43, [], 39]","[42, 194, \"quick\", \"quick\", \"JJ\", \"amod\", 43, [], 39]","[43, 200, \"meal\", \"meal\", \"NN\", \"pobj\", 40, [41, 42, 44], 39]","[44, 205, \"on\", \"on\", \"IN\", \"prep\", 43, [46], 39]","[45, 208, \"the\", \"the\", \"DT\", \"det\", 46, [], 39]","[46, 212, \"run\", \"run\", \"NN\", \"pobj\", 44, [45], 39]","[47, 215, \".\", \".\", \".\", \"punct\", 39, [], 39]"]}

    doc = Doc(example['parsed'])
    print(doc)

    for token in doc:
        print(token.i, token.idx, token, token.pos_, token.lemma_, token.dep_, token.head, token.children)
