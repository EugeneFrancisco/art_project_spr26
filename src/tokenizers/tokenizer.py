"""
This file defines the ABC for all the tokenizer model
"""

from abc import ABC, abstractmethod
from typing import Optional
import numpy as np

class Token:
    """
    A very small class to keep token information
    and provide a helpful print line.

    Equality is asymmetric by design: two `Token`s are equal only when both
    `name` and `is_entity` match, but a `Token` compares equal to a plain `str`
    whenever just `name` matches. This lets callers query/filter with bare
    strings while still distinguishing the word "polke" from the entity
    "Sigmar Polke" when both Tokens are in hand.
    """
    def __init__(self, name: str, is_entity: bool = False):
        # The string associated with the token.
        self.name = name

        # Whether or not this token is just a word or an entity.
        self.is_entity = is_entity

    def __repr__(self) -> str:
        sign = "e" if self.is_entity else "w"
        return f"[{sign}] {self.name}"

    def __str__(self) -> str:
        return self.name

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Token):
            return self.name == other.name and self.is_entity == other.is_entity
        if isinstance(other, str):
            return self.name == other
        return NotImplemented

    def __hash__(self) -> int:
        # Hash on `name` alone so `hash(Token(x)) == hash(x)`, keeping the
        # Token/str equality usable in sets and dicts.
        return hash(self.name)

    @property
    def url(self) -> Optional[str]:
        """
        The Wikipedia URL for this token, or `None` if it isn't an entity.

        Assumes the English Wikipedia, matching the `enwiki_*` pretrained
        Wikipedia2Vec models.
        """
        if not self.is_entity:
            return None
        return f"https://en.wikipedia.org/wiki/{self.name.replace(' ', '_')}"

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
    def get_vector(self, token: str | Token) -> Optional[np.ndarray]:
        """
        Look up the embedding vector for a token.

        Args:
            token: The token to look up. A plain `str` is resolved with
                whatever fallback policy the subclass chooses; a `Token` with
                `is_entity` set selects the word- or entity-vocabulary
                explicitly.

        Returns:
            A `self.dim`-dimensional numpy array containing the embedding for
            `token`, or `None` if the token is not in the vocabulary.
        """

    @abstractmethod
    def get_neighbors(self, vec: np.ndarray, k: int) -> list[tuple[Token, float]]:
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

    def __contains__(self, token: str | Token) -> bool:
        """
        Return True iff `token` is in the tokenizer's vocabulary.

        Enables `token in tokenizer` syntax.
        """
        return self.get_vector(token) is not None

    def similarity(self, a: str | Token, b: str | Token) -> Optional[float]:
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

    def most_similar(self, name: str | Token, k: int = 10) -> list[tuple[Token, float]]:
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

    def analogy(
        self,
        positives: list[str | Token],
        negatives: Optional[list[str | Token]] = None,
        k: int = 10,
    ) -> list[tuple[Token, float]]:
        """
        Solve a vector-arithmetic analogy: sum the `positives` embeddings and
        subtract the `negatives` embeddings, then return the `k` vocabulary
        tokens nearest to the resulting vector (excluding any input tokens).

        For the classic "king - man + woman = queen" analogy, call
        `analogy(positives=["king", "woman"], negatives=["man"])`.

        Args:
            positives: Tokens whose vectors are added to the query.
            negatives: Tokens whose vectors are subtracted. Defaults to none.
            k: The number of neighbors to return. Defaults to 10.

        Returns:
            A list of `(token, cosine_similarity)` tuples of length at most `k`,
            ordered from most to least similar. Returns an empty list if any
            input token is not in the vocabulary.
        """
        negatives = negatives or []
        if not positives:
            return []

        query = np.zeros(self.dim)
        inputs: set[Token | str] = set()
        for token, sign in [(t, 1.0) for t in positives] + [(t, -1.0) for t in negatives]:
            vec = self.get_vector(token)
            if vec is None:
                return []
            query += sign * vec
            inputs.add(token)

        return [(n, s) for n, s in self.get_neighbors(query, k + len(inputs)) if n not in inputs][:k]
