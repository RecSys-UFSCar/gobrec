
from gobrec.mabs.lin_mabs import Lin
import numpy as np
import torch


class LinGreedy(Lin):

    def __init__(self, seed: int = None, epsilon: float = 1.0, l2_lambda: float = 1.0, use_gpu: bool = False):
        
        super().__init__(seed, l2_lambda, use_gpu)
        self.epsilon = epsilon


    def predict(self, contexts: np.ndarray):

        x = torch.tensor(contexts, device=self.device)

        num_arms = self.beta.shape[0]
        num_contexts = contexts.shape[0]

        random_mask = self.rng.random(contexts.shape[0]) < self.epsilon
        random_indexes = random_mask.nonzero()[0]
        not_random_indexes = (~random_mask).nonzero()[0]

        scores = torch.empty((num_contexts, num_arms), device=self.device, dtype=torch.double)

        scores[random_mask] = torch.tensor(self.rng.random((len(random_indexes), num_arms)), device=self.device, dtype=torch.double)
        scores[not_random_indexes] = torch.matmul(x[not_random_indexes], self.beta.T)

        return scores
