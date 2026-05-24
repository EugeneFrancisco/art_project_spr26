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
    the individual prices their works have fetched at auction.
    """

    @abstractmethod
    def get_data(
        self,
        acceptance_function: Optional[Callable[[str], bool]] = None
        ) -> tuple[list, np.ndarray]:
        """
        Returns a tuple of X and Y with len(X) == len(Y). Each (X[i], Y[i]) is one
        individual sale: X[i] is the artist's name and Y[i] is the price that a single
        one of their works fetched at auction. The same artist will typically appear in
        X many times — once per sale — so there will be many duplicate Xs paired with
        different Ys.

        acceptance_function is an optional callable which, if provided, will only add
        sales whose artist name satisfies acceptance_function.
        """
        raise NotImplementedError

    def get_data_aggregate(
        self,
        method: str = "median",
        acceptance_function: Optional[Callable[[str], bool]] = None,
    ) -> tuple[list, np.ndarray]:
        """
        Like `get_data`, but collapses repeated sales per artist into a single
        aggregated price. Returns (X, Y) where each artist appears exactly once
        in X and Y[i] is the aggregated price for X[i].

        Args:
            method: "median" or "mean" — the aggregation applied to each artist's
                list of sale prices.
            acceptance_function: forwarded to `get_data`.
        """
        if method == "median":
            agg = np.median
        elif method == "mean":
            agg = np.mean
        else:
            raise ValueError(f"method must be 'median' or 'mean', got {method!r}")

        names, prices = self.get_data(acceptance_function=acceptance_function)

        grouped: dict[str, list[float]] = {}
        for name, price in zip(names, prices):
            grouped.setdefault(name, []).append(float(price))

        unique_names = list(grouped.keys())
        aggregated = np.array([agg(grouped[n]) for n in unique_names])
        return unique_names, aggregated

    def get_data_in_wiki(
            self,
            wiki: WikiTokenizer,
            only_entries: bool,
            aggregate: bool,
            method: str = "median"
        ) -> tuple[list, np.ndarray]:
        """
        Same as `get_data`, but restricted to sales whose artist name appears in the
        passed-in WikiTokenizer.
        
        Args:
            wiki: The WikiTokenizer (Wikipedia2Vec wrapper) in which we want to query.
            only_entries: if set to True, we will only return names which appear as a
                wikipedia entry. If set to False, then we will return all names which
                appear as either words or entries in wikipedia.
            aggregate: If true, will call get_data_aggregate instead of get_data
            median: Used in conjunction with aggregate. If true, aggregates by median,
            otherwise by mean.
        """
        if only_entries:
            accept = wiki.has_entity
        else:
            accept = lambda name: name in wiki
        if aggregate:
            return self.get_data_aggregate(
                method=method,
                acceptance_function=accept,
            )
        return self.get_data(acceptance_function=accept)
