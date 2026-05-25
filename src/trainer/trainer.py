"""
This class defines the abstract base class for models that train.
"""

from abc import ABC, abstractmethod
import numpy as np
from src.tokenizers.tokenizer import Token, Tokenizer

class Vec2Price(ABC):
    """
    An abstract base class for taking embedding vectors and predicting
    prices.
    """
    def __init__(self, tokenizer: Tokenizer):
        """
        Inheritors should add in this constructor any necessary
        hyperparameters and all data for testing and training that would
        be needed in the abstract methods below.
        """
        self.tokenizer = tokenizer
        self.embedding_dim = tokenizer.dim

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
    def test_baseline(self) -> float:
        """
        Return the variance of the test set, i.e. the MSE achieved by
        always predicting the mean. Compare against `test()` to see
        whether the model beats the constant-mean baseline.
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

    @abstractmethod
    def load_model(self, path: str) -> None:
        """
        Load a previously trained model from `path` into this instance.
        After this call, the trainer should be ready to serve predictions
        without first calling `train`.
        """
