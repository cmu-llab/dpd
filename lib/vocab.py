from specialtokens import *

import pickle

class Vocab:
    protolang: str
    daughter_langs: list[str]
    
    # creates v2i and i2v dictionaries, starting with special tokens, then adding the provided tokens.
    def __init__(self, tokens):
        tokens = set(tokens)
        
        self.v2i = {special_token: idx for idx, special_token in enumerate(SPECIAL_TOKENS)}

        for idx, token in enumerate(sorted(tokens)):
            self.v2i[token] = idx + len(SPECIAL_TOKENS)

        self.i2v = {v: k for k, v in self.v2i.items()}
        assert len(self.v2i) == len(self.i2v)

    def to_tokens(self, index_sequence, remove_special=True):
        '''
        * convert indices to tokens
        '''
        ret = []
        for idx in index_sequence:
            idx = idx.item()
            if remove_special:
                if idx in {UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX}:
                    continue
            ret.append(self.i2v[idx])

        return ret

    def to_indices(self, token_sequence):
        '''
        * convert tokens to indices
        '''
        return [self.v2i.get(token, UNK_IDX) for token in token_sequence]

    def get_idx(self, v):
        return self.v2i.get(v, UNK_IDX)

    def __len__(self):
        return len(self.v2i)

    def __getitem__(self, idx):
        return self.i2v.get(idx, self.i2v[UNK_IDX])

    def __iter__(self):
        for idx, tkn in sorted(self.i2v.items(), key=lambda x: x[0]):
            yield idx, tkn

    def add_token(self, token):
        index = len(self.v2i)
        self.v2i[token] = index
        self.i2v[index] = token

def build_vocab(train_filepath, include_lang_tkns_in_ipa_vocab, verbose):
    '''
    * build ipa vocabulary and language vocabulary from training data
    '''
    vocab = set()
    languages = set()

    with open(train_filepath, 'rb') as fin:
        langs, data = pickle.load(fin)
        for char, entry in data.items():
            target = entry['protoform']
            vocab.update(list(target.values())[0])
            for language, source in entry['daughters'].items():
                vocab.update(source)
                languages.add(language)

    ipa_vocab = Vocab(vocab)
    langs_vocab = Vocab(languages)
    langs_vocab.protolang = langs[0]
    langs_vocab.daughter_langs = langs[1:]

    if include_lang_tkns_in_ipa_vocab:
        # note - the separators are already in the vocabulary
        # add daughter languages to the token vocabulary
        for lang in langs_vocab.daughter_langs:
            ipa_vocab.add_token(lang)

        langs_vocab.add_token(langs_vocab.protolang)
        # special tokens will belong to this separate language
        langs_vocab.add_token(SEPARATOR_LANG)

    if verbose: print(f'ipa vocabulary: {len(ipa_vocab)}')
    if verbose: print(f'language vocabulary: {len(langs_vocab)}')

    return ipa_vocab, langs_vocab, langs