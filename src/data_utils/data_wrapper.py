"""
This file defines an abstract base classes for downloading data.
"""

from abc import ABC, abstractmethod
from typing import Optional, Callable
import numpy as np
from src.tokenizers.wiki_tokenizer import WikiTokenizer

class AuctionScraper(ABC):
    """
    This is an abstract base class for getting data about artists and
    their median auction prices.
    """

    @abstractmethod
    def get_data(
        self,
        acceptance_function: Optional[Callable[[str], bool]] = None
        ) -> tuple[list, np.ndarray]:
        """
        Returns a tuple of X and Y. X here is a list of names of artists. Y is
        the corresponding array of their median auction prices.

        acceptance_function is an optional callable which, if provided, will only add
        names to the returned list if acceptance_function returns true on the name.
        """
        raise NotImplementedError

    def get_data_in_wiki(self, wiki: WikiTokenizer, only_entries: bool) -> tuple[list, np.ndarray]:
        """
        Returns a tuple of X and Y. X here is a list of names of artists. Y is
        the corresponding array of their median auction prices. Except the list of
        artists here are ones whose names appear in the WikiTokenizer.
        
        Args:
            wiki: The WikiTokenizer (Wikipedia2Vec wrapper) in which we want to query.
            only_entries: if set to True, we will only return names which appear as a
                wikipedia entry. If set to False, then we will return all names which
                appear as either words or entries in wikipedia.
        """
        if only_entries:
            accept = wiki.has_entity
        else:
            accept = lambda name: name in wiki
        return self.get_data(acceptance_function=accept)
