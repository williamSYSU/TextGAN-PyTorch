from gensim.models import Word2Vec

import config as cfg
from utils.text_process import text_file_iterator


class MultipleFilesIterator:
    def __init__(self, files):
        self.files = files

    def __iter__(self):
        for file in self.files:
            yield from [cfg.padding_token] * 5 + text_file_iterator(file)


class EmbeddingsTrainer:
    def __init__(self, sources, save_filename):
        self.sources = sources
        self.size = size
        self.save_filename = save_filename

    def make_embeddings(self):
        w2v = Word2Vec(
            sentences=MultipleFilesIterator(self.sources),
            size=cfg.w2v_embedding_size,
            window=cfg.w2v_window,
            min_count=cfg.w2v_min_count,
            workers=cfg.w2v_workers,
        )
        w2v.save(self.save_filename)


def load_embedding(path):
    return Word2Vec.load(path)
