"""
Tokenizer implementation backed by a pretrained Wikipedia2Vec model.
"""

from typing import Optional
import numpy as np
from wikipedia2vec import Wikipedia2Vec
from wikipedia2vec.dictionary import Entity, Word

from src.tokenizers.tokenizer import Token, Tokenizer


class WikiTokenizer(Tokenizer):
    """
    A Tokenizer backed by a pretrained Wikipedia2Vec model.

    Tokens may be either plain words or Wikipedia entity titles. When a bare
    `str` is passed, words are tried first and entities second. To force a
    specific vocabulary, pass a `Token` with `is_entity` set.
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

    def get_vector(self, token: str | Token) -> Optional[np.ndarray]:
        """
        Look up the embedding vector for a token.

        For a `Token`, the `is_entity` flag selects the vocabulary exactly. For
        a bare `str`, `token` is tried first as a plain word, then as a
        Wikipedia entity title.

        Args:
            token: The word or entity title to look up.

        Returns:
            The embedding vector, or `None` if `token` is not present in the
            selected vocabulary.
        """
        if isinstance(token, Token):
            try:
                if token.is_entity:
                    return np.asarray(self._model.get_entity_vector(token.name))
                return np.asarray(self._model.get_word_vector(token.name))
            except KeyError:
                return None

        try:
            return np.asarray(self._model.get_word_vector(token))
        except KeyError:
            pass
        try:
            return np.asarray(self._model.get_entity_vector(token))
        except KeyError:
            return None

    def get_neighbors(self, vec: np.ndarray, k: int) -> list[tuple[Token, float]]:
        """
        Find the `k` tokens whose embeddings are closest to `vec` in cosine
        similarity.

        Args:
            vec: A query vector with shape `(self.dim,)`.
            k: The number of neighbors to return.

        Returns:
            A list of `(Token, cosine_similarity)` tuples ordered from most to
            least similar. Each `Token` carries an `is_entity` flag so callers
            can tell word hits apart from Wikipedia entity hits.
        """
        results = self._model.most_similar_by_vector(vec, count=k)
        return [(_to_token(item), float(score)) for item, score in results]


def _to_token(item: Word | Entity) -> Token:
    """Wrap a Wikipedia2Vec Word or Entity item in a `Token`."""
    if isinstance(item, Entity):
        return Token(item.title, is_entity=True)
    return Token(item.text, is_entity=False)
