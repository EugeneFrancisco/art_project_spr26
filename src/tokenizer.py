"""
This file defines the ABC for all the tokenizer model
"""

from abc import ABC, abstractmethod
from typing import Optional
import numpy as np

class Tokenizer(ABC):
    """
    Abstract base class for tokenizer models that map tokens to fixed-dimensional
    embedding vectors and support nearest-neighbor lookups in the embedding space.

    Subclasses must implement `get_vector` and `get_neighbors`. The other methods
    (`__contains__`, `similarity`, `most_similar`) are provided as shared
    utilities built on top of those two primitives.
    """
    def __init__(self, dim: int):
        """
        Initialize the tokenizer.

        Args:
            dim: The dimensionality of the embedding vectors produced by this
                tokenizer.
        """
        self.dim = dim

    # ================== Abstract methods =========================

    @abstractmethod
    def get_vector(self, token) -> Optional[np.ndarray]:
        """
        Look up the embedding vector for a token.

        Args:
            token: The token to look up.

        Returns:
            A `self.dim`-dimensional numpy array containing the embedding for
            `token`, or `None` if the token is not in the vocabulary.
        """

    @abstractmethod
    def get_neighbors(self, vec, k) -> list[tuple[str, int]]:
        """
        Find the `k` tokens in the vocabulary whose embeddings are closest to
        `vec` under cosine similarity.

        Args:
            vec: A `self.dim`-dimensional query vector.
            k: The number of neighbors to return.

        Returns:
            A list of `(token, cosine_similarity)` tuples, ordered from most to
            least similar.
        """

    # Methods shared by all instances

    def __contains__(self, name: str) -> bool:
        """
        Return True iff `name` is in the tokenizer's vocabulary.

        Enables `name in tokenizer` syntax.
        """
        return self.get_vector(name) is not None

    def similarity(self, a: str, b: str) -> Optional[float]:
        """
        Compute the cosine similarity between the embeddings of two tokens.

        Args:
            a: The first token.
            b: The second token.

        Returns:
            The cosine similarity between the two embeddings as a float, or
            `None` if either token is missing from the vocabulary.
        """
        va, vb = self.get_vector(a), self.get_vector(b)
        if va is None or vb is None:
            return None
        return float(va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb)))

    def most_similar(self, name: str, k: int = 10) -> list[tuple[str, float]]:
        """
        Find the `k` tokens in the vocabulary most similar to `name`, excluding
        `name` itself.

        Args:
            name: The query token.
            k: The number of similar tokens to return. Defaults to 10.

        Returns:
            A list of `(token, cosine_similarity)` tuples of length at most `k`,
            ordered from most to least similar. Returns an empty list if `name`
            is not in the vocabulary.
        """
        vec = self.get_vector(name)
        if vec is None:
            return []
        return [(n, s) for n, s in self.get_neighbors(vec, k + 1) if n != name][:k]
