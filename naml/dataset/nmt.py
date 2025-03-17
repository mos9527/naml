# nmt: Neural Machine Translation
from naml.dataset import Datasets, DatasetRemote
from naml.text import replace_multiple, tokenize_line

def nmt_tokenizer(line : str):
    return list(tokenize_line(line, '!.:;?\'',' '))

def load_nmt(db : Datasets, lang1: str = "fra", lang2: str = "eng", max_examples: int = -1, one_to_one: bool = True):
    table = db.fetch(DatasetRemote(
        f"{lang1}-{lang2}", f"https://www.manythings.org/anki/{lang1}-{lang2}.zip"
    )).as_zip().read(f'{lang1}.txt').decode('utf-8').lower()

    lines = table.split('\n')[:max_examples]
    lines = [replace_multiple(line, ["\u202f","\xa0"]," ").split('\t')[:2] for line in lines]
    lines = [line for line in lines if len(line) == 2]
    src_words, target_words = [line[0] for line in lines], [line[-1] for line in lines]
    src_target = list(zip(src_words, target_words))
    if one_to_one:    
        src_target = dict(src_target)
        src_target = list(src_target.items())    
    src_words = [nmt_tokenizer(line[0]) for line in src_target]
    target_words = [nmt_tokenizer(line[1]) for line in src_target]    

    return src_words, target_words