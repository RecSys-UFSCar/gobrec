
import torch
import numpy as np
from gobrec.mabs.MABAlgo import MABAlgo


class Lin(MABAlgo):

    def __init__(self, seed: int = None, l2_lambda: float = 1.0, use_gpu: bool = False, items_per_batch: int = 10_000):
        
        super().__init__(seed)

        self.l2_lambda = l2_lambda
        self.device = 'cuda' if use_gpu else 'cpu'

        self.items_per_batch = items_per_batch
        self.already_initialized = False
    
    def _update_label_encoder_and_matrices_sizes(self, decisions: np.ndarray, num_features: int):

        self._update_label_encoder(decisions, num_features)

        if not self.already_initialized:
            self.Xty = torch.zeros((self.num_arms, self.num_features), device=self.device, dtype=torch.double)
            self.A = torch.eye(self.num_features, device=self.device, dtype=torch.double).unsqueeze(0).repeat(self.num_arms, 1, 1) * self.l2_lambda
            self.beta = torch.zeros((self.num_arms, self.num_features), device=self.device, dtype=torch.double)
            self.already_initialized = True
        elif self.num_arms != self.beta.shape[0]:
            # the number of arms has changed, we need to update the matrices
            Xty_new = torch.zeros((self.num_arms, self.num_features), device=self.device, dtype=torch.double)
            Xty_new[:self.Xty.shape[0]] = self.Xty
            self.Xty = Xty_new

            A_new = torch.eye(self.num_features, device=self.device, dtype=torch.double).unsqueeze(0).repeat(self.num_arms, 1, 1) * self.l2_lambda
            A_new[:self.A.shape[0]] = self.A
            self.A = A_new

            beta_new = torch.zeros((self.num_arms, self.num_features), device=self.device, dtype=torch.double)
            beta_new[:self.beta.shape[0]] = self.beta
            self.beta = beta_new


    def fit(self, contexts: np.ndarray, decisions: np.ndarray, rewards: np.ndarray):

        self._update_label_encoder_and_matrices_sizes(decisions, contexts.shape[1])

        X_device = torch.tensor(contexts, device=self.device, dtype=torch.double)
        y_device = torch.tensor(rewards, device=self.device, dtype=torch.double)
        decisions_device = torch.tensor(self.label_encoder.transform(decisions), device=self.device, dtype=torch.long)

        self.A.index_add_(0, decisions_device, torch.einsum('ni,nj->nij', X_device, X_device))

        self.Xty.index_add_(0, decisions_device, X_device * y_device.view(-1, 1))

        for j in range(0, self.beta.shape[0], self.items_per_batch):            
            self.beta[j:j+self.items_per_batch] = torch.linalg.solve(
                self.A[j:j+self.items_per_batch],
                self.Xty[j:j+self.items_per_batch]
            )

    def predict(self, contexts: np.ndarray):
        scores = torch.empty((contexts.shape[0], self.num_arms), device=self.device, dtype=torch.double)
        for start in range(0, self.num_arms, self.items_per_batch):
            end = min(start + self.items_per_batch, self.num_arms)  
            scores[:, start:end] = torch.einsum('bd,ad->ba', torch.tensor(contexts, device=self.device, dtype=torch.double), self.beta[start:end])
        
        return scores

    def reset(self):
        '''
        Reset the linear model to its initial state.
        '''
        super().reset()
        self.already_initialized = False
        self.Xty = None
        self.A = None
        self.beta = None