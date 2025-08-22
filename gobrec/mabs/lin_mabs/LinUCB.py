
from gobrec.mabs.lin_mabs import Lin
import numpy as np
import torch


class LinUCB(Lin):

    def __init__(self, seed: int = None, alpha: float = 1.0, l2_lambda: float = 1.0, use_gpu: bool = False, items_per_batch: int = 10_000):
        
        super().__init__(seed, l2_lambda, use_gpu, items_per_batch)
        self.alpha = alpha


    def predict(self, contexts: np.ndarray):

        x = torch.tensor(contexts, device=self.device, dtype=torch.double)

        scores = torch.matmul(x, self.beta.T)

        for j in range(0, self.beta.shape[0], self.items_per_batch):
            x_A_inv = torch.matmul(x, torch.linalg.inv(self.A[j: j+self.items_per_batch]))

            # Upper confidence bound = alpha * sqrt(x A^-1 xt). Notice that, x = xt
            # ucb values are claculated for all the contexts in one single go. type(ucb): np.ndarray
            ucb = self.alpha * torch.sqrt(torch.sum(x_A_inv * x, axis=2))

            # Calculate linucb expectation y = x * b + ucb
            scores[:, j: j+self.items_per_batch] += ucb.T
        
        return scores