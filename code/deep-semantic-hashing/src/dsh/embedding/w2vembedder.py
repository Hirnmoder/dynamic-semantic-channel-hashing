import re
import numpy as np
from numpy import frombuffer, dtype
from symspellpy.symspellpy import SymSpell
import torch

from dsh.embedding.embedder import Embedder
from dsh.utils.logger import Logger
from dsh.utils.progress import tqdm
from dsh.utils.types import TextRawInput


class W2VEmbedder(Embedder):
    def __init__(
        self,
        words: dict[str, int],
        embedding: torch.nn.Embedding,
        output_sequence_length: int,
    ):
        self.output_sequence_length = output_sequence_length
        self.words = {word.lower(): idx for word, idx in words.items()}
        self.embedding = embedding
        self.number_regex = re.compile(r"\d", re.IGNORECASE)
        self.tokenizer = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
        self._load_dictionary()
        self.ignored_words: set[str] = set()

    def _load_dictionary(self):
        Logger().info("[EMB] Creating custom dictionary.")
        import pkg_resources

        base_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
        # term_index is the column of the term and count_index is the
        # column of the term frequency
        self.tokenizer.load_dictionary(base_path, term_index=0, count_index=1)
        Logger().info("[EMB] Updating pre-loaded dictionary with custom words.")
        for word in tqdm(self.words.keys()):
            self.tokenizer.create_dictionary_entry(word, 1)

    def __call__(self, text: str | list[str]) -> TextRawInput:
        if isinstance(text, list):
            return torch.stack([self.__call__(text_) for text_ in text], dim=0)
        words = text.lower().split()
        indices = []
        while len(words) > 0 and len(indices) <= self.output_sequence_length:
            word = words.pop(0)  # pop the first word
            word = self.number_regex.sub("#", word)  # replace numbers with '#'
            if word not in self.words:
                # Use SymSpell to correct the spelling if possible
                result = self.tokenizer.word_segmentation(word)
                if result.corrected_string and len(result.corrected_string) > 0:
                    for corrected_word in result.corrected_string.split():
                        if corrected_word not in self.words:
                            self.ignored_words.add(word)
                        else:
                            indices.append(self.words[corrected_word])
                else:
                    self.ignored_words.add(word)
            else:
                indices.append(self.words[word])

        return torch.nn.functional.pad(
            self.embedding(torch.LongTensor(indices)),
            (0, 0, 0, self.output_sequence_length - len(indices)),
            value=0.0,
        )

    @staticmethod
    def load(path: str, sequence_length: int) -> "W2VEmbedder":
        # Load the Word2Vec model from disk but without gensim, because it can't be installed due to NumPy dependency conflicts
        # We can't use torchtext, too, because it stops working with PyTorch >= 2.4 and is not maintained anymore
        Logger().info("[EMB] Loading Word2Vec model from disk.")
        words, embeddings = _load_word2vec(path)
        return W2VEmbedder(
            words,
            torch.nn.Embedding.from_pretrained(embeddings, freeze=True),
            sequence_length,
        )


def _any2unicode(text, encoding="utf8", errors="strict"):
    if isinstance(text, str):
        return text
    return str(text, encoding, errors=errors)


def _load_word2vec(path: str, encoding="utf8", unicode_errors="strict") -> tuple[dict[str, int], torch.Tensor]:
    """Loads word2vec format vectors from a file.
    Args:
        path (str): Path to the file containing word vectors in word2vec format.
        encoding (str, optional): The text encoding for reading the file. Defaults to "utf8".
        unicode_errors (str, optional): How to handle errors when decoding Unicode strings. See Python's str function for details. Defaults to "strict".
    Returns:
        tuple[dict[str, int], torch.Tensor]: A dictionary mapping words to their vector indices and a tensor containing the vectors themselves.
    """
    index: dict[str, int] = {}
    embedding: np.ndarray
    # Basic parsing code from https://github.com/pytorch/text/pull/421#issuecomment-425102647
    with open(path, "rb") as f:
        # read header
        header = _any2unicode(f.readline(), encoding=encoding)
        num_words, dim = list(map(int, header.split()))
        embedding = np.zeros((num_words, dim), dtype=np.float32)

        binary_len = dtype(np.float32).itemsize * dim
        for i in tqdm(range(num_words)):
            # read word
            chars = []
            while True:
                ch = f.read(1)
                if ch == b" ":
                    break
                if ch == b"":
                    raise EOFError("unexpected end of input; is count incorrect or file otherwise damaged?")
                if ch != b"\n":  # ignore newlines in front of words (some binary files have)
                    chars.append(ch)
            word = _any2unicode(b"".join(chars), encoding=encoding, errors=unicode_errors)
            weights = frombuffer(f.read(binary_len), dtype=np.float32).astype(np.float32)
            # do something to store word and word-vector
            index[word] = i
            embedding[i] = weights
    return index, torch.tensor(embedding)
