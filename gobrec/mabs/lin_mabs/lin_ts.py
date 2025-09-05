
from gobrec.mabs.lin_mabs import Lin
import numpy as np
import torch

class LinTS(Lin):

    def __init__(self, seed: int = None, alpha: float = 1.0, l2_lambda: float = 1.0, use_gpu: bool = False, items_per_batch: int = 10_000):
        
        super().__init__(seed, l2_lambda, use_gpu, items_per_batch)
        self.alpha = alpha
    
    def predict(self, contexts: np.ndarray):
        x = torch.tensor(contexts, device=self.device, dtype=torch.double)

        num_arms, num_features = self.beta.shape
        num_contexts = contexts.shape[0]

        scores = torch.empty((num_contexts, num_arms), device=self.device, dtype=torch.double)

        eps = torch.from_numpy(self.rng.standard_normal(size=(num_contexts, num_features))).to(device=self.device, dtype=torch.double)

        for start in range(0, num_arms, self.items_per_batch):
            end = min(start + self.items_per_batch, num_arms)  

            beta_chunk = self.beta[start:end]
            A_chunk = self.A[start:end]
            A_inv_chunk = torch.linalg.inv(A_chunk)

            L_chunk = torch.linalg.cholesky((self.alpha ** 2) * A_inv_chunk)
            beta_sampled = torch.einsum('bd,add->bad', eps, L_chunk) + beta_chunk

            scores[:, start:end] = torch.einsum('bd,bad->ba', x, beta_sampled)

        return scores