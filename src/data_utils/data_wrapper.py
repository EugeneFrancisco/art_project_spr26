"""
This file defines an abstract base classes for downloading data.
"""

from abc import ABC, abstractmethod
import numpy as np

class AuctionScraper(ABC):
    """
    This is an abstract base class for getting data about artists and
    their median auction prices.
    """

    @abstractmethod
    def get_data(self) -> tuple[list, np.ndarray]:
        """
        Returns a tuple of X and Y. X here is a list of names of artists. Y is
        the corresponding array of their median auction prices.
        """
        raise NotImplementedError
