
import torch
import numpy as np
from gobrec.mabs.MABAlgo import MABAlgo


class RandomMAB(MABAlgo):

    def __init__(self, seed: int = None):
        super().__init__(seed)

    def fit(self, contexts: np.ndarray, decisions: np.ndarray, rewards: np.ndarray):
        self._update_label_encoder(decisions, contexts.shape[1])

    def predict(self, contexts: np.ndarray):
        return torch.from_numpy(self.rng.random((contexts.shape[0], self.num_arms))).double()
    
    def reset(self):
        super().reset()