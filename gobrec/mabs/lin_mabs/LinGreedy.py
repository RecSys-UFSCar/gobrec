
from gobrec.mabs.lin_mabs import Lin
import numpy as np
import torch


class LinGreedy(Lin):

    def __init__(self, seed: int = None, epsilon: float = 0.1, l2_lambda: float = 1.0, use_gpu: bool = False, items_per_batch: int = 10_000):
        
        super().__init__(seed, l2_lambda, use_gpu, items_per_batch)
        self.epsilon = epsilon


    def predict(self, contexts: np.ndarray):

        x = torch.tensor(contexts, device=self.device, dtype=torch.double)

        scores = torch.empty((contexts.shape[0], self.num_arms), device=self.device, dtype=torch.double)
        random_mask = self.rng.random(contexts.shape[0]) < self.epsilon
        random_indexes = random_mask.nonzero()[0]
        not_random_indexes = (~random_mask).nonzero()[0]

        scores[random_mask] = torch.tensor(self.rng.random((len(random_indexes), self.num_arms)), device=self.device, dtype=torch.double)

        for start in range(0, len(not_random_indexes), self.items_per_batch):
            end = min(start + self.items_per_batch, len(not_random_indexes))
            batch_indexes = not_random_indexes[start:end]
            scores[batch_indexes] = torch.einsum('bd,ad->ba', x[batch_indexes], self.beta)

        return scores
