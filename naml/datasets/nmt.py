# nmt: Neural Machine Translation
from naml.dataset import Datasets, DatasetRemote
from naml.text import replace_multiple, tokenize_line

def load_nmt(db : Datasets, lang1: str = "fra", lang2: str = "eng"):
    table = db.fetch(DatasetRemote(
        f"{lang1}-{lang2}", f"https://www.manythings.org/anki/{lang1}-{lang2}.zip"
    )).as_zip().read(f'{lang1}.txt').decode('utf-8').lower()

    num_examples = -1
    lines = table.split('\n')[:num_examples]
    lines = [replace_multiple(line, ["\u202f","\xa0"]," ").split('\t')[:2] for line in lines]
    lines = [line for line in lines if len(line) == 2]
    src_words, target_words = [line[0] for line in lines], [line[-1] for line in lines]
    src_words = [list(tokenize_line(line, '!.:;?',' ')) for line in src_words]
    target_words = [list(tokenize_line(line, '!.:;?',' ')) for line in target_words]    

    return src_words, target_words