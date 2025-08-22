
from abc import ABC, abstractmethod
import numpy as np
from sklearn.preprocessing import LabelEncoder


class MABAlgo(ABC):
    """
    Abstract class for Multi-Armed Bandit algorithms.
    """

    def __init__(self, seed: int = None):
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)
        self.label_encoder = None
        self.num_arms = None
        self.num_features = None
    
    def _update_label_encoder(self, items_ids: np.ndarray, num_features: int):
        """
        Update the label encoder with new item IDs.
        """
        if self.label_encoder is None:
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(items_ids)
            self.num_arms = len(self.label_encoder.classes_)
        else:
            new_classes = np.setdiff1d(items_ids, self.label_encoder.classes_)
            if len(new_classes) > 0:
                all_classes = np.concatenate((self.label_encoder.classes_, new_classes))
                self.label_encoder.fit(all_classes)
            self.num_arms = len(self.label_encoder.classes_)
        
        if self.num_features is None:
            self.num_features = num_features
        
        assert num_features == self.num_features, "Number of features has changed!"
        

    @abstractmethod
    def fit(self, contexts: np.ndarray, items_ids: np.ndarray, rewards: np.ndarray):
        """
        Fit the MAB algorithm with the provided contexts, item IDs, and rewards.
        """
        pass

    @abstractmethod
    def predict(self, contexts: np.ndarray) -> np.ndarray:
        """
        Predict the expected rewards for the given contexts.
        """
        pass