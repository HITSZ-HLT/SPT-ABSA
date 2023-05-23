class Tagset:
    def __init__(self, tags):
        self.tag_to_idx = self._to_idx(tags)
        self.idx_to_tag = self._to_tag(tags)

    def _to_tag(self, tags):
        return dict((k, v) for k, v in enumerate(tags))

    def _to_idx(self, tags):
        return dict((k, v) for v, k in enumerate(tags))

    def __getitem__(self, tag):
        return self.tag_to_idx[tag]

    def __len__(self):
        return self.size()

    def size(self):
        return len(self.tag_to_idx)

    def make(self, entities, text_length, max_seq_length):
        bio_seq = from_entities_to_bio(entities, text_length, max_seq_length)
        return self.from_tag_seq_to_idx_seq(bio_seq)

    def parse(self, idx_seq):
        bio_seq = self.from_idx_seq_to_tag_seq(idx_seq)
        return from_bio_to_entites(bio_seq)

    def from_tag_seq_to_idx_seq(self, tag_seq):
        return [self.tag_to_idx[tag] for tag in tag_seq]

    def from_idx_seq_to_tag_seq(self, idx_seq):
        return [self.idx_to_tag[idx] for idx in idx_seq if self.idx_to_tag[idx] != 'PAD']



def from_entities_to_bio(entities, text_length, max_seq_length):
    """
    build bio_seq by entities
    """
    assert (text_length+1) <= max_seq_length
    bio_seq = ['SOS'] + ['O'] * text_length + ['PAD'] * (max_seq_length-text_length-1)
    
    for start, end, type_ in entities:
        start, end = start+1, end+1
        
        # assert bio_seq[start] == 'O'
        # if bio_seq[start] != 'O':
        #     print('warning', entities)
        bio_seq[start] = f'B-{type_}'
        
        for i in range(start+1, end):
            # assert bio_seq[i] == 'O'
            # if bio_seq[i] != 'O':
            #     print('warning', entities)
            bio_seq[i] = f'I-{type_}'

    return bio_seq



def from_bio_to_entites(bio_seq):
    """
    parse bio_seq to get entities

    bio: ['O', O', 'B-type', 'I-type', 'O', 'PAD']
    labels: [(type, 2, 4),]
    """
    entities = []
    state = 'O'
    entity = ()

    for i, bio in enumerate(bio_seq):
        if state == 'O':
            if bio[0] == 'B':
                entity = (bio[2:], i)

        elif state in ('B', 'I'):
            if bio[0] in ('O', 'B'):
                entity += (i,)
                entities.append(entity)

            if bio[0] == 'B':
                entity = (bio[2:], i)

        state = 'O' if bio in ('PAD', 'SOS') else bio[0]

    # O O B (I)
    if len(entity) == 2:
        entity += (i+1, )
        entities.append(entity)

    return [entity for entity in entities if len(entity) == 3 and type(entity[0]) is str]