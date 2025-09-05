
from abc import ABC, abstractmethod
import numpy as np


class LabelEncoder:
    def __init__(self):
        self.class_to_index: dict[int, int] = {}
        self.index_to_class: list[int] = []

    def fit(self, decisions: np.ndarray):
        for cls in decisions:
            if cls not in self.class_to_index:
                idx = len(self.index_to_class)
                self.class_to_index[cls] = idx
                self.index_to_class.append(cls)

    def transform(self, decisions: np.ndarray) -> np.ndarray:
        return np.array([self.class_to_index[cls] for cls in decisions])

    def inverse_transform(self, indices: np.ndarray) -> np.ndarray:
        return np.array([self.index_to_class[idx] for idx in indices])

    @property
    def classes_(self):
        return np.array(self.index_to_class)


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
    
    def _update_label_encoder(self, decisions: np.ndarray, num_features: int):
        """
        Update the label encoder with new item IDs.
        """
        if self.label_encoder is None:
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(decisions)
            self.num_arms = len(self.label_encoder.classes_)
        else:
            new_classes = np.setdiff1d(decisions, self.label_encoder.classes_)
            if len(new_classes) > 0:
                all_classes = np.concatenate((self.label_encoder.classes_, new_classes))
                self.label_encoder.fit(all_classes)
            self.num_arms = len(self.label_encoder.classes_)
        
        if self.num_features is None:
            self.num_features = num_features
        
        assert num_features == self.num_features, "Number of features has changed!"
        

    @abstractmethod
    def fit(self, contexts: np.ndarray, decisions: np.ndarray, rewards: np.ndarray):
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

    @abstractmethod
    def reset(self):
        """
        Reset the MAB algorithm to its initial state.
        """
        self.label_encoder = None
        self.num_arms = None
        self.num_features = None
        self.rng = np.random.default_rng(self.seed)