# class for handling datasets and dataloaders

from .vocab import build_vocab, Vocab
from .dataset import DatasetConcat
from torch.utils.data import DataLoader
import os.path as path
import specialtokens
import tabulate
import pytorch_lightning as pl

class DataloaderManager:
    def __init__(self,
        data_dir: str, # e.g. "data/chinese_wikihan2022", 
        batch_size: int, # e.g. 32,
        test_val_batch_size: int, # e.g. 32,
        lang_separators: bool, # e.g. False,
        transformer_d2p_d_cat_style: bool,
        include_lang_tkns_in_ipa_vocab: bool, # e.g. False,
        skip_daughter_tone: bool, # e.g. False,
        skip_protoform_tone: bool, # e.g. False,
        shuffle_train: bool, # e.g. True,
        daughter_subset: list[str] | None, # see dataset.py for details
        min_daughters: int, # see dataset.py for details
        verbose: bool, 
        proportion_labelled: float,
        
        datasetseed: int,
        exclude_unlabelled: bool = False, # whether to throw away unlabelled examples, effectively making it supervised
    ) -> None:
        self.verbose = verbose
        if self.verbose: print(f'Loading data from {data_dir}...')
        
        print(f'setting seed to {datasetseed} to build dataset')
        pl.seed_everything(datasetseed)
        
        self.data_dir = path.abspath(data_dir)
        self.batch_size = batch_size
        self.test_val_batch_size = test_val_batch_size
        self.include_lang_tkns_in_ipa_vocab = include_lang_tkns_in_ipa_vocab
        self.lang_separators = lang_separators
        self.skip_daughter_tone = skip_daughter_tone
        self.skip_protoform_tone = skip_protoform_tone
        self.shuffle_train = shuffle_train
        self.daughter_subset = daughter_subset
        self.min_daughters = min_daughters
        self.proportion_labelled = proportion_labelled
        self.transformer_d2p_d_cat_style = transformer_d2p_d_cat_style
        self.exclude_unlabelled = exclude_unlabelled
        
        if not self.include_lang_tkns_in_ipa_vocab:
            print('WARN: include_lang_tkns_in_ipa_vocab is False. This will make all lang tokens unknown. Beware when using encodeTokenSeqAppendTargetLangToken mode')
        if self.lang_separators and not self.include_lang_tkns_in_ipa_vocab:
            print('ERR: lang_separators is True, but include_lang_tkns_in_ipa_vocab is False. This will make all lang separators unknown.')

        self.ipa_vocab, self.lang_vocab, self.langs = build_vocab(
            train_filepath = f'{self.data_dir}/train.pickle', 
            include_lang_tkns_in_ipa_vocab = self.include_lang_tkns_in_ipa_vocab,
            verbose = self.verbose,
        )
        
        self.train_set: DatasetConcat = self.get_dataset('train')
        self.train_p_labelled_mask_fingerprint = self.train_set.p_labelled_mask_fingerprint
        self.test_set: DatasetConcat = self.get_dataset('test')
        self.val_set: DatasetConcat = self.get_dataset('dev')
        
    # returns a table showing the sequence
    def format_sequence(self, ipa_seq, lang_seq) -> str:
        ipas = self.ipa_vocab.to_tokens(ipa_seq, remove_special=False)
        langs = self.lang_vocab.to_tokens(lang_seq, remove_special=False)
        pairs = [list(item) for item in list(zip(lang_seq, langs, ipa_seq, ipas))]
            
        table = tabulate.tabulate(pairs, headers=['id', 'lang', 'id', 'ipa'])
            
        return table
        
    def get_dataset(self, partition) -> DatasetConcat:
        assert partition in {'train', 'dev', 'test'}, f'invalid partition: {partition}'
        if partition == 'train':
            proportion_labelled = self.proportion_labelled
        else:
            proportion_labelled = 1.0
        
        # load datasets
        dataset = DatasetConcat(
            lang_separators = self.lang_separators,
            filepath = f"{self.data_dir}/{partition}.pickle",
            ipa_vocab = self.ipa_vocab,
            lang_vocab = self.lang_vocab,
            skip_daughter_tone = self.skip_daughter_tone,
            skip_protoform_tone = self.skip_protoform_tone,
            daughter_subset = self.daughter_subset,
            min_daughters = self.min_daughters,
            verbose = self.verbose,
            proportion_labelled = proportion_labelled,
            transformer_d2p_d_cat_style=self.transformer_d2p_d_cat_style,
            exclude_unlabelled=self.exclude_unlabelled
        )
        
        if self.verbose: print(f'# samples ({partition}): {len(dataset)}')
        
        return dataset 

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_set, 
            collate_fn=self.train_set.collate_fn, 
            batch_size=self.batch_size,
            shuffle=self.shuffle_train,
        )
        
    def train_transductive_test_dataloader(self) -> DataLoader:
        self.train_set.into_transductive_test_set()
        return DataLoader(
            self.train_set, 
            collate_fn=self.train_set.collate_fn, 
            batch_size=self.test_val_batch_size,
            shuffle=False,
        )

        
    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_set, 
            collate_fn=self.test_set.collate_fn, 
            batch_size=self.test_val_batch_size,
            shuffle=False,
        )
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_set, 
            collate_fn=self.val_set.collate_fn, 
            batch_size=self.test_val_batch_size,
            shuffle=False,
        )

    def print_data_summary(self):
        train_example = self.train_set.__getitem__(1) 
        (x_seqs, x_langs, x_lens, x_concat_seq_len, y_seqs, y_langs, y_lens, xs) = train_example
        
        print(f'''
              
=== DATA SUMMARY ===
data dir: {self.data_dir}
batch size: {self.batch_size}
lang_separators: {self.lang_separators}
include_lang_tkns_in_ipa_vocab: {self.include_lang_tkns_in_ipa_vocab}
skip_daughter_tone: {self.skip_daughter_tone}
skip_protoform_tone: {self.skip_protoform_tone}
shuffle_train: {self.shuffle_train}

ipa vocab: 
{self.ipa_vocab.i2v}

lang vocab: 
{self.lang_vocab.i2v}

train example: 
{train_example}

concatinated daughters:
{self.format_sequence(x_seqs, x_langs)}

length of each daughter:
{x_lens}

proto sequence, note that the dialect is the protolang:
{self.format_sequence(y_seqs, y_langs)}

each daughter in individual sequences:
        ''')   
             
        for x in xs:
            print(x['d'])
            print(self.format_sequence(x['d_seq'], x['l_seq']))
            print()
            
        print('\n === END DATA SUMMARY === \n')