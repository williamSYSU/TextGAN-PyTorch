from gensim.models import Word2Vec
from tqdm import tqdm
from pathlib import Path

import config as cfg
from utils.text_process import text_file_iterator


class MultipleFilesEmbeddingIterator:
    def __init__(self, files):
        self.files = files

    def __iter__(self):
        for file in tqdm(self.files, desc="iterating files"):
            for tokens in text_file_iterator(file):
                yield [cfg.padding_token] * 5 + tokens


class EmbeddingsTrainer:
    def __init__(self, sources, save_filename):
        self.sources = sources
        self.save_filename = save_filename

    def make_embeddings(self):
        w2v = Word2Vec(
            sentences=MultipleFilesEmbeddingIterator(self.sources),
            size=cfg.w2v_embedding_size,
            window=cfg.w2v_window,
            min_count=cfg.w2v_min_count,
            workers=cfg.w2v_workers,
        )
        Path(self.save_filename).parents[0].mkdir(parents=True, exist_ok=True)
        w2v.save(self.save_filename)


def load_embedding(path):
    return Word2Vec.load(path)
