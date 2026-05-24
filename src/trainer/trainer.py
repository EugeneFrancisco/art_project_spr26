"""
This class defines the abstract base class for models that train.
"""

from abc import ABC, abstractmethod
import numpy as np
from src.tokenizers.tokenizer import Token

class Vec2Price(ABC):
    """
    An abstract base class for taking embedding vectors and predicting
    prices.
    """
    def __init__(self, embedding_dim: int):
        """
        Inheritors should add in this constructor any necessary
        hyperparameters and all data for testing and training that would
        be needed in the abstract methods below.
        """
        self.embedding_dim = embedding_dim

    @abstractmethod
    def train(self):
        """
        The function to train the model. If this function needs
        hyperparameters, store these as member variables and use
        that. Data should be stored as member variables as well.
        """

    @abstractmethod
    def test(self) -> float:
        """
        The function to test the model on a testing set. This
        should return the MSE on the test set.
        """

    @abstractmethod
    def get_pred(self, name: str | Token) -> float:
        """
        Get a predicted price for the passed in name.
        """

    @abstractmethod
    def get_preds(self, names: list[str | Token]) -> np.ndarray:
        """
        Get an array of predicted prices for the list of passed in names.
        """
