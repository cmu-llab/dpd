import pickle

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from .vocab import Vocab
import torch.nn.functional as F
from lib.tensor_utils import random_mask_by_proportion_subsets
from specialtokens import *
import copy
from einops import repeat, rearrange
from prelude import batch_t, LabelStatus
import hashlib

class DatasetBase(Dataset):
    def __init__(self, 
        filepath: str, 
        ipa_vocab: Vocab, 
        lang_vocab: Vocab, 
        skip_daughter_tone: bool, 
        skip_protoform_tone: bool, 
        daughter_subset: list[str] | None, # list of daughter languages to include. the vocab will still be built from the entire dataset
            # None means include all daughters
            # sth like ["Italian", "French", "Cantonese(?)"] means only include those daughters. Concatinated dataset will also be in that order.
        min_daughters: int, # e.ge 3 # minimum number of daughters in a cognate set to be included in the dataset
        verbose: bool,
        proportion_labelled: float, # proportion of the dataset that is labelled, useful for semi-supervised learning
    ):
        self.ipa_vocab = ipa_vocab
        self.lang_vocab = lang_vocab
        
        self.skip_daughter_tone = skip_daughter_tone
        self.skip_protoform_tone = skip_protoform_tone
        
        self.verbose = verbose
        
        # 1 > load dataset
        D: list[dict[str, list[str]]] = [] # list of daughters, which are dicts
        Pl: list[dict[str, list[str]]] = [] # list of protos, always labelled
        Ps: list[dict[str, list[str]]] = [] # surface label, the one visible to model
        Pf: list[LabelStatus] = [] # list of labelling status flags for protos
        with open(filepath, 'rb') as file:
            langs_list, data = pickle.load(file)
        
        # 2 > get langs
        self.protolang: str = langs_list[0]
        self.all_daughter_langs: list[str] = langs_list[1:]
        self.daughter_subset: list[str] | None = daughter_subset

        if self.daughter_subset is None:
            self.daughter_subset = self.all_daughter_langs

        if self.verbose: print(f'protolang: {self.protolang}')
        if self.verbose: print(f'all daughter langs: {self.all_daughter_langs}')
        if self.verbose: print(f'selected daughter langs: {self.daughter_subset}')

        assert all([L in self.all_daughter_langs for L in self.daughter_subset]), f'invalid self.daughter_subset: {self.daughter_subset}'
        self.num_daughters = len(self.daughter_subset)
        
        # 3 > filter and built dataset            
        for _cognate, entry in data.items():
            entry_subset = self.extract_subset_of_daughters(entry['daughters'])
            if len(entry_subset) >= min_daughters:
                D.append(entry_subset)
                Pl.append(entry['protoform'])

        self.length = len(D)
        
        # 4 > built partially labelled dataset
        p_labelled_mask = random_mask_by_proportion_subsets(self.length, proportion_labelled).bool()
        self.p_labelled_mask = p_labelled_mask
        
        if proportion_labelled != 1.0:
            # print(p_labelled_mask[:100])
            fingerprint = hashlib.sha256(f'{list(p_labelled_mask)}'.encode()).hexdigest()
            print(f"train set fingerprint {fingerprint}")
            self.p_labelled_mask_fingerprint = fingerprint
        else:
            self.p_labelled_mask_fingerprint = None
        
        for (pl, is_labelled) in zip(Pl, p_labelled_mask):
            if is_labelled:
                Ps.append(pl)
                Pf.append(LabelStatus.LABELLED)
            else:
                pu = copy.deepcopy(pl)
                pu[self.protolang] = []
                Ps.append(pu)
                Pf.append(LabelStatus.UNLABELLED)

        # 9 > store
        assert len(D) == len(Pl) == len(Ps) == len(Pf)
        self.D = D
        self.Pl = Pl
        self.Ps = Ps
        self.Pf = Pf

    # DESTRUCTIVE!
    # turns partially labelled train set into a test set that tests the model's performance on the unlabelled portion of training data
    def into_transductive_test_set(self):
        print("WARNING: entire train set turning into transductive test set. Not more training allowed")
        
        assert (self.length == len(self.D))
        length_before = self.length
        
        # 1 > reinitialize dataset entries
        D: list[dict[str, list[str]]] = []
        Pl: list[dict[str, list[str]]] = []
        Ps: list[dict[str, list[str]]] = []
        Pf: list[LabelStatus] = []
        
        # 2 > filter
        
        for i in range(self.length):
            if self.Pf[i] != LabelStatus.LABELLED:
                D.append(self.D[i])
                Pl.append(self.Pl[i])
                Ps.append(self.Ps[i])
                Pf.append(LabelStatus.REVEALED)
                
        length_after = len(D)
        
        # 9 > store 
        print(f"length went from {length_before} to {length_after}, that's {length_after/length_before:.2} of entries preserved")
        
        self.D = D
        self.Pl = Pl
        self.Ps = Pl # surface all labels
        self.Pf = Pf
        self.length = length_after
        assert len(self.D) == len(self.Pl) == len(self.Ps) == len(self.Pf) == self.length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        raise NotImplementedError()
    
    def extract_subset_of_daughters(self, entry_daughters: dict) -> dict:
        assert self.daughter_subset != None
        return {L: entry_daughters[L] for L in filter(lambda L: L in entry_daughters, self.daughter_subset)}



class DatasetConcat(DatasetBase):
    def __init__(self, 
        lang_separators: bool, 
        transformer_d2p_d_cat_style: bool, # if true the concat daughter seq follow Kim et al.'s style: <bos>d1<eos><bos>d2<eos>...
        **kwargs # passed to DatasetBase
    ):
        super().__init__(**kwargs)

        self.lang_separators = lang_separators
        self.transformer_d2p_d_cat_style = transformer_d2p_d_cat_style
        # sanity check
        if lang_separators:
            assert self.lang_vocab.get_idx(SEPARATOR_LANG) != UNK_IDX
            assert self.lang_vocab.get_idx(self.protolang) != UNK_IDX
            for lang in self.all_daughter_langs:
                assert self.lang_vocab.get_idx(lang) != UNK_IDX

    # get one training example
    def __getitem__(self, idx) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, list[dict], Tensor, Tensor, Tensor, LabelStatus]:
        
        # 1 > create concatenated daughter sequences
        
        if self.transformer_d2p_d_cat_style:
            x_sequence, x_dialects, x_lens = [], [], []
            for dialect, sequence in self.D[idx].items():
                sequence = [SPECIAL_TOKENS[BOS_IDX]] + sequence + [SPECIAL_TOKENS[EOS_IDX]]
                x_sequence.append(torch.tensor([self.ipa_vocab.get_idx(tkn) for tkn in sequence], dtype=torch.long))
                x_dialects.extend([self.lang_vocab.get_idx(dialect)] * len(sequence))
                x_lens.append(len(sequence))

            x_sequence = torch.cat(x_sequence)  # concatenation of sequence of token indices
            x_dialects = torch.tensor(x_dialects, dtype=torch.long)  # set of dialect indices
            x_lens = torch.tensor(x_lens, dtype=torch.long)  # list of lengths for each dialect
            
            d_cat_lens_tensor: Tensor = torch.LongTensor([len(x_sequence)]) # length of the concatenated sequence
            d_cat_tkns_tensor, d_cat_langs_tensor, d_indv_lens_tensor, d_cat_lens_tensor = x_sequence, x_dialects, x_lens, d_cat_lens_tensor
        else: 
            d_cat_tkns: list[int] = []
            d_cat_langs: list[int] = []
            d_indv_lens: list[int] = [] # individual daughter lengths
            
            # p_lang: str
            # p_tkn_seq: list[str]
            for p_langs, p_tkn_seq in self.D[idx].items():
                if self.skip_daughter_tone:
                    p_tkn_seq = p_tkn_seq[:-1]

                p_lang_seq = [p_langs] * len(p_tkn_seq)

                if self.lang_separators:
                    p_tkn_seq =  [SPECIAL_TOKENS[COGNATE_SEP_IDX],  p_langs,      SPECIAL_TOKENS[LANG_SEP_IDX]]  + p_tkn_seq
                    p_lang_seq = [SEPARATOR_LANG,                   SEPARATOR_LANG,  SEPARATOR_LANG]            + p_lang_seq

                d_cat_tkns.extend([self.ipa_vocab.get_idx(tkn) for tkn in p_tkn_seq])
                d_cat_langs.extend([self.lang_vocab.get_idx(lang) for lang in p_lang_seq])
                d_indv_lens.append(len(p_tkn_seq))

            assert len(d_cat_langs) == len(d_cat_tkns)
            assert len(d_indv_lens) == len(self.D[idx])
            
            # example: BOS * French : croître * Italian : crescere * Spanish : crecer * Portuguese : crecer * Romanian : crește * EOS
            #   starts and ends with BOS and EOS
            #   COGNATE_SEP_IDX separates daughter cognates from each other
            #   the language tokens are one token
            #   LANG_SEP_IDX separates the language token from the actual cognate sequence
            if self.lang_separators:
                d_cat_tkns =  [BOS_IDX]                                  + d_cat_tkns   + [COGNATE_SEP_IDX,                          EOS_IDX]
                d_cat_langs = [self.lang_vocab.get_idx(SEPARATOR_LANG)]  + d_cat_langs  + [self.lang_vocab.get_idx(SEPARATOR_LANG),  self.lang_vocab.get_idx(SEPARATOR_LANG)]
            else:
                d_cat_tkns =  [BOS_IDX]                                  + d_cat_tkns   + [EOS_IDX]
                d_cat_langs = [self.lang_vocab.get_idx(SEPARATOR_LANG)]  + d_cat_langs  + [self.lang_vocab.get_idx(SEPARATOR_LANG)]
    
            d_cat_tkns_tensor: Tensor = torch.tensor(d_cat_tkns, dtype=torch.long)
            d_cat_langs_tensor: Tensor = torch.tensor(d_cat_langs, dtype=torch.long)
            
            # NOTE d_indv_lens_tensor is the len of the entire |Mandarin|<:>|i|˧˥|<*>| thing. so the example has length 5.
            d_indv_lens_tensor: Tensor = torch.tensor(d_indv_lens, dtype=torch.long)  # lengths of each daughter sequence
            d_cat_lens_tensor: Tensor = torch.LongTensor([len(d_cat_tkns_tensor)]) # length of the concatenated sequence

        # 2 > create protoform sequence
        
        p_tkns: list[int]
        p_langs: list[int]
        p_len: int       

        p_tkns = list(self.Ps[idx].values())[0] # surface
        p_l_tkns = list(self.Pl[idx].values())[0] # labelled
        p_f = self.Pf[idx] # labelling status flags
        if p_f == LabelStatus.UNLABELLED: # unlabelled protoform
            p_tkns_tensor = torch.tensor([PAD_IDX, PAD_IDX], dtype=torch.long)
            p_langs_tensor = torch.tensor([PAD_IDX, PAD_IDX], dtype=torch.long)
            p_lens_tensor = torch.tensor(2, dtype=torch.long)
        else:
            if self.skip_protoform_tone:
                p_tkns = p_tkns[:-1]
            p_langs = [self.lang_vocab.get_idx(self.protolang)] * len(p_tkns)
            p_len = len(p_tkns)
            
            p_tkns = [BOS_IDX] + [self.ipa_vocab.get_idx(tkn) for tkn in p_tkns] + [EOS_IDX]
            p_langs = [self.lang_vocab.get_idx(SEPARATOR_LANG)] + p_langs + \
                    [self.lang_vocab.get_idx(SEPARATOR_LANG)]
            assert len(p_langs) == len(p_tkns)
        
            p_tkns_tensor = torch.tensor(p_tkns, dtype=torch.long)
            p_langs_tensor = torch.tensor(p_langs, dtype=torch.long)
            p_lens_tensor = torch.tensor(p_len, dtype=torch.long)
            
        # 3 > expose labels for train time evaluation
        
        if self.skip_protoform_tone:
            p_l_tkns = p_l_tkns[:-1]
        p_l_langs = [self.lang_vocab.get_idx(self.protolang)] * len(p_l_tkns)
        p_l_len = len(p_l_tkns)
        
        p_l_tkns = [BOS_IDX] + [self.ipa_vocab.get_idx(tkn) for tkn in p_l_tkns] + [EOS_IDX]
        p_l_langs = [self.lang_vocab.get_idx(SEPARATOR_LANG)] + p_l_langs + \
                [self.lang_vocab.get_idx(SEPARATOR_LANG)]
        assert len(p_l_langs) == len(p_l_tkns)
        p_l_tkns_tensor = torch.tensor(p_l_tkns, dtype=torch.long)
        p_l_langs_tensor = torch.tensor(p_l_langs, dtype=torch.long)
        p_l_lens_tensor = torch.tensor(p_l_len, dtype=torch.long)
        p_f = p_f


        # 4 > create individual daughter sequences
        
        daughters_entries_list: list[dict] = []
        
        d_lang_str: str
        d_str_seq: list[str]
        for d_lang_str, d_str_seq in self.D[idx].items():
            if (len(d_str_seq) == 1) and (d_str_seq[0] == '-'): 
                continue # kim et al had placeholder '-' token for unknown daughter, we don't want the reflex pred model to learn that
            
            if self.skip_daughter_tone:
                d_str_seq = d_str_seq[:-1]
                
            d_lang_str_seq: list[str] = [d_lang_str] * len(d_str_seq)
            
            d_seq = [BOS_IDX] + [self.ipa_vocab.get_idx(tkn) for tkn in d_str_seq] + [EOS_IDX]
            d_lang_seq = [self.lang_vocab.get_idx(SEPARATOR_LANG)] + [self.lang_vocab.get_idx(lang) for lang in d_lang_str_seq] + [self.lang_vocab.get_idx(SEPARATOR_LANG)]
            
            daughters_entries_list.append({
                "d": d_lang_str,
                "d_ipa_id": self.ipa_vocab.get_idx(d_lang_str),
                "d_lang_id": self.lang_vocab.get_idx(d_lang_str),
                "d_seq": torch.tensor(d_seq),
                "l_seq": torch.tensor(d_lang_seq),
            })

        return d_cat_tkns_tensor, d_cat_langs_tensor, d_indv_lens_tensor, d_cat_lens_tensor, p_tkns_tensor, p_langs_tensor, p_lens_tensor, daughters_entries_list, p_l_tkns_tensor, p_l_langs_tensor, p_l_lens_tensor, p_f

    # function used by pytorch dataloader to pad sequence when a batch is created
    # before padding, we also append the language token after <eos> to each proto sequence. This is easier than doing so after padding.
    def d2p_collate_fn(self, 
        batch: list[tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, list[dict]], Tensor, Tensor, Tensor, LabelStatus]
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
                
        # 1 > unpack lists of batches into list of tensors
        
        d_cat_tkns_list: list[Tensor] = []
        d_cat_langs_list: list[Tensor] = []
        d_indv_lens_list: list[Tensor] = []
        d_cat_lens_list: list[Tensor] = []
        p_tkns_list: list[Tensor] = []
        
        p_l_tkns_list: list[Tensor] = []
        p_f_list: list[LabelStatus] = []

        # x_seq, x_lang, x_len, x_concat_seq_len, y_seq
        for p_concat_seq_tensor, p_concat_lang_tensor, p_individual_lens_tensor, p_concat_seq_len_tensor, p_seq_tensor, p_lang_tensor, p_len_tensor, daughters_entries_list, p_l_tkns_tensor, p_l_langs_tensor, p_l_lens_tensor, p_f in batch:
            d_cat_tkns_list.append(p_concat_seq_tensor)
            d_cat_langs_list.append(p_concat_lang_tensor)
            d_indv_lens_list.append(p_individual_lens_tensor)
            d_cat_lens_list.append(p_concat_seq_len_tensor)
            p_tkns_list.append(p_seq_tensor)
            
            p_l_tkns_list.append(p_l_tkns_tensor)
            p_f_list.append(p_f.value)

        d_cat_tkns: Tensor = pad_sequence(d_cat_tkns_list, batch_first=True, padding_value=PAD_IDX)
        d_cat_langs: Tensor = pad_sequence(d_cat_langs_list, batch_first=True, padding_value=PAD_IDX)
        d_indv_lens: Tensor = pad_sequence(d_indv_lens_list, batch_first=True, padding_value=0)
        d_cat_lens = torch.stack(d_cat_lens_list, dim=0)
        p_tkns: Tensor = pad_sequence(p_tkns_list, batch_first=True, padding_value=PAD_IDX)
        p_l_tkns: Tensor = pad_sequence(p_l_tkns_list, batch_first=True, padding_value=PAD_IDX)
        p_fs: Tensor = torch.tensor(p_f_list).view(-1, 1)

        # 2 > making proto lang and ipa lang vectors (this used to be done within training loop, but doing it here makes things easier)

        N = d_cat_tkns.shape[0]
        p_ipa_lang_vec  = repeat((torch.LongTensor([self.ipa_vocab.get_idx(self.protolang)])), '1 -> N 1', N=N) 
        p_lang_lang_vec = repeat((torch.LongTensor([self.lang_vocab.get_idx(self.protolang)])), '1 -> N 1', N=N)

        # 9 > return

        # d_cat_tkns (N, Ld)
        # d_cat_langs (N, Ld)
        # d_cat_lens (N, 1)
        # d_indv_lens (N, min(Nd, daughters present))
        # p_ipa_lang_vec (N, 1)
        # p_lang_lang_vec (N, 1)
        # p_tkns (N, Lp)
        # where N is the batch size, Ld is the max length of the concatinated daughter sequences in the batch, and Lp is the max length of the protoform sequences in the batch
        return d_cat_tkns, d_cat_langs, d_cat_lens, d_indv_lens, p_lang_lang_vec, p_tkns, p_l_tkns, p_fs

    def p2d_collate_fn(self, 
            batch: list[tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, list[dict]]]
        ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        
        prompted_p_tkns_list, prompted_p_lens_list, d_ipa_langs_list, d_lang_langs_list, d_tkns = [], [], [], [], []
        
        for p_concat_seq_tensor, p_concat_lang_tensor, p_individual_lens_tensor, p_concat_seq_len_tensor, p_seq_tensor, p_lang_tensor, p_len_tensor, daughters_entries_list, p_l_tkns_tensor, p_l_langs_tensor, p_l_lens_tensor, p_f in batch:
            for daughter_entry in daughters_entries_list:
                prompted_propo_seq = torch.cat((  torch.tensor([daughter_entry['d_ipa_id']]), p_seq_tensor  ))
                prompted_p_tkns_list.append(prompted_propo_seq)
                prompted_p_lens_list.append(torch.LongTensor([len(prompted_propo_seq)]))
                d_tkns.append(daughter_entry['d_seq'])
                d_ipa_langs_list.append(daughter_entry['d_ipa_id']) # ipa id for the daughter languages
                d_lang_langs_list.append(daughter_entry['d_lang_id']) # lang id for the daughter languages
                        
        prompted_p_tkns = pad_sequence(prompted_p_tkns_list, batch_first=True, padding_value=PAD_IDX)
        d_ipa_langs = torch.tensor(d_ipa_langs_list).unsqueeze(1)
        d_lang_langs = torch.tensor(d_lang_langs_list).unsqueeze(1)
        d_tkns = pad_sequence(d_tkns, batch_first=True, padding_value=PAD_IDX)
        prompted_p_lens = torch.stack(prompted_p_lens_list, dim=0)
        
        assert len(prompted_p_tkns) == len(d_ipa_langs) == len(d_lang_langs) == len(d_tkns) == len(prompted_p_lens)
        assert torch.max(prompted_p_lens) == prompted_p_tkns.shape[1]
        
        # prompted_proto_seqs (N', L_p)
        # prompted_proto_seqs_lens (N', 1)
        # daughters_ipa_langs, daughters_lang_langs (N', 1)
        # daughters_seqs (N', L_d)
        # where N' is the number of daughter sequences in the batch and L_p and L_d are the max lengths of the protoform and daughter sequences in the batch, respectively
        return prompted_p_tkns, prompted_p_lens, d_ipa_langs, d_lang_langs, d_tkns
        # enforce a set size for each cognate set so that we can batch many process that depend on pairing cognate sets to proto
    
    def p2d_padded_per_set_collate_fn(self, 
            batch: list[tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, list[dict]]]
        ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        
        N: int = len(batch)
        Nd: int = self.num_daughters
        
        prompted_proto_seqs_list_list, prompted_proto_seqs_lens_list, daughters_ipa_langs_list_list, daughters_lang_langs_list_list, daughters_seqs_list_list = [], [], [], [], []
        
        for p_concat_seq_tensor, p_concat_lang_tensor, p_individual_lens_tensor, p_concat_seq_len_tensor, p_seq_tensor, p_lang_tensor, p_len_tensor, daughters_entries_list, p_l_tkns_tensor, p_l_langs_tensor, p_l_lens_tensor, p_f in batch:
            prompted_proto_seqs_list, prompted_proto_seqs_lens, daughters_ipa_langs_list, daughters_lang_langs_list, daughters_seqs_list = [], [], [], [], []
            for i, daughter_entry in enumerate(daughters_entries_list):
                prompted_propo_seq = torch.cat((  torch.tensor([daughter_entry['d_ipa_id']]), p_seq_tensor  ))
                prompted_proto_seqs_list.append(prompted_propo_seq)
                prompted_proto_seqs_lens.append(torch.LongTensor([len(prompted_propo_seq)]))
                daughters_seqs_list.append(daughter_entry['d_seq'])
                daughters_ipa_langs_list.append(daughter_entry['d_ipa_id']) # ipa id for the daughter languages
                daughters_lang_langs_list.append(daughter_entry['d_lang_id']) # lang id for the daughter languages
            while i < Nd - 1:
                prompted_proto_seqs_list.append(torch.LongTensor([PAD_IDX]))
                prompted_proto_seqs_lens.append(torch.LongTensor([PAD_IDX]))
                daughters_seqs_list.append(torch.LongTensor([PAD_IDX]))
                daughters_ipa_langs_list.append(torch.LongTensor([PAD_IDX]))
                daughters_lang_langs_list.append(torch.LongTensor([PAD_IDX]))
                i += 1
                
            prompted_proto_seqs = pad_sequence(prompted_proto_seqs_list, batch_first=True, padding_value=PAD_IDX)
            daughters_ipa_langs = torch.tensor(daughters_ipa_langs_list).unsqueeze(1)
            daughters_lang_langs = torch.tensor(daughters_lang_langs_list).unsqueeze(1)
            daughters_seqs = pad_sequence(daughters_seqs_list, batch_first=True, padding_value=PAD_IDX)
            prompted_proto_seqs_lens = torch.stack(prompted_proto_seqs_lens, dim=0)
        
            assert len(prompted_proto_seqs) == len(daughters_ipa_langs) == len(daughters_lang_langs) == len(daughters_seqs) == len(prompted_proto_seqs_lens)
            assert torch.max(prompted_proto_seqs_lens) == prompted_proto_seqs.shape[1]
            
            prompted_proto_seqs_list_list.append(prompted_proto_seqs)
            prompted_proto_seqs_lens_list.append(prompted_proto_seqs_lens)
            daughters_ipa_langs_list_list.append(daughters_ipa_langs)
            daughters_lang_langs_list_list.append(daughters_lang_langs)
            daughters_seqs_list_list.append(daughters_seqs)
            
        prompted_proto_seqs_max_len = max([T.shape[1] for T in prompted_proto_seqs_list_list])
        daughters_seqs_max_len = max([T.shape[1] for T in daughters_seqs_list_list])
        
        prompted_proto_seqs_s = torch.stack(tuple((F.pad(T, (0, prompted_proto_seqs_max_len - T.shape[1]),"constant", PAD_IDX)) for T in prompted_proto_seqs_list_list))
        daughters_seqs_s = torch.stack(tuple((F.pad(T, (0, daughters_seqs_max_len - T.shape[1]),"constant", PAD_IDX)) for T in daughters_seqs_list_list))
        
        prompted_proto_seqs_lens_s = torch.stack(tuple(T for T in prompted_proto_seqs_lens_list))
        daughters_ipa_langs_s = torch.stack(tuple(T for T in daughters_ipa_langs_list_list))
        daughters_lang_langs_s = torch.stack(tuple(T for T in daughters_lang_langs_list_list))
        
        # prompted_proto_seqs (N, Nd, L_p)
        # prompted_proto_seqs_lens (N, Nd, 1)
        # daughters_ipa_langs (N, Nd, 1)
        # daughters_lang_langs (N, Nd, 1)
        # daughters_seqs (N, Nd, L_d)
        return prompted_proto_seqs_s, prompted_proto_seqs_lens_s, daughters_ipa_langs_s, daughters_lang_langs_s, daughters_seqs_s
    
    def collate_fn(self, 
            batch: list[tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, list[dict]]]
        ) -> batch_t:
        return self.d2p_collate_fn(batch), self.p2d_collate_fn(batch), self.p2d_padded_per_set_collate_fn(batch)
    
