from gensim.models import Word2Vec

import config as cfg
from utils.text_process import get_tokenized_from_file


class EmbeddingsTrainer:
    def __init__(self, *files, size=512, save_filename=cfg.word2vec_model_name):
        self.files = files
        self.size = size
        self.save_filename = save_filename

    def make_embeddings(self, verbose=True):
        tokenized = get_tokenized_from_file(self.files)
        W2V = Word2Vec(
            sentences=tokenized,
            size=self.size,
            window=cfg.w2v_window,
            min_count=cfg.w2v_min_count,
            workers=cfg.w2v_workers,
        )
        W2V.save(self.save_filename)
