"""
Tokenizer implementation backed by a pretrained Wikipedia2Vec model.
"""

from typing import Optional
import numpy as np
from wikipedia2vec import Wikipedia2Vec

from tokenizer import Tokenizer


class WikiTokenizer(Tokenizer):
    """
    A Tokenizer backed by a pretrained Wikipedia2Vec model.

    Tokens may be either plain words or Wikipedia entity titles. On lookup,
    words are tried first and entities second, so a token that exists as both
    resolves to its word vector.
    """

    def __init__(self, model_path: str):
        """
        Load a pretrained Wikipedia2Vec model from disk.

        Args:
            model_path: Path to a Wikipedia2Vec model file (e.g. one of the
                pretrained files distributed by the Wikipedia2Vec project).
        """
        self._model = Wikipedia2Vec.load(model_path)
        dim = int(self._model.syn0.shape[1])
        super().__init__(dim)

    def get_vector(self, token: str) -> Optional[np.ndarray]:
        """
        Look up the embedding vector for a token.

        Tries `token` first as a plain word, then as a Wikipedia entity title.

        Args:
            token: The word or entity title to look up.

        Returns:
            The embedding vector, or `None` if `token` is in neither the word
            nor entity vocabulary.
        """
        try:
            return np.asarray(self._model.get_word_vector(token))
        except KeyError:
            pass
        try:
            return np.asarray(self._model.get_entity_vector(token))
        except KeyError:
            return None

    def get_neighbors(self, vec: np.ndarray, k: int) -> list[tuple[str, float]]:
        """
        Find the `k` tokens whose embeddings are closest to `vec` in cosine
        similarity.

        Args:
            vec: A query vector with shape `(self.dim,)`.
            k: The number of neighbors to return.

        Returns:
            A list of `(token, cosine_similarity)` tuples ordered from most to
            least similar. Words are returned as their text and entities as
            their Wikipedia title.
        """
        results = self._model.most_similar_by_vector(vec, count=k)
        return [(_item_name(item), float(score)) for item, score in results]


def _item_name(item) -> str:
    """Return the display string for a Wikipedia2Vec Word or Entity item."""
    # Word objects expose `.text`; Entity objects expose `.title`.
    return getattr(item, "text", None) or item.title
