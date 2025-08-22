
import torch
import numpy as np
from sklearn.preprocessing import LabelEncoder


class Lin:

    def __init__(self, seed: int = None, l2_lambda: float = 1.0, use_gpu: bool = False):

        self.seed = seed
        self.rng = np.random.default_rng(self.seed)
        self.l2_lambda = l2_lambda
        self.device = 'cuda' if use_gpu else 'cpu'

        self.label_encoder = None
        self.num_features = None
        self.items_per_batch = 1000
    
    def _update_label_encoder_and_matrices_sizes(self, items_ids: np.ndarray, num_features: int):
        if self.label_encoder is None:
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(items_ids)
            num_arms = len(self.label_encoder.classes_)
            self.num_features = num_features
            self.Xty = torch.zeros((num_arms, num_features), device=self.device, dtype=torch.double)
            self.A = torch.eye(num_features, device=self.device, dtype=torch.double).unsqueeze(0).repeat(num_arms, 1, 1) * self.l2_lambda
            self.beta = torch.zeros((num_arms, num_features), device=self.device, dtype=torch.double)
        else:
            assert num_features == self.num_features, "Number of features has changed!"
            new_classes = np.setdiff1d(items_ids, self.label_encoder.classes_)
            if len(new_classes) > 0:
                all_classes = np.concatenate((self.label_encoder.classes_, new_classes))
                self.label_encoder.fit(all_classes)
                num_arms = len(self.label_encoder.classes_)

                Xty_new = torch.zeros((num_arms, num_features), device=self.device, dtype=torch.double)
                Xty_new[:self.Xty.shape[0]] = self.Xty
                self.Xty = Xty_new

                A_new = torch.eye(num_features, device=self.device, dtype=torch.double).unsqueeze(0).repeat(num_arms, 1, 1) * self.l2_lambda
                A_new[:self.A.shape[0]] = self.A
                self.A = A_new

                beta_new = torch.zeros((num_arms, num_features), device=self.device, dtype=torch.double)
                beta_new[:self.beta.shape[0]] = self.beta
                self.beta = beta_new


    def fit(self, contexts: np.ndarray, items_ids: np.ndarray, rewards: np.ndarray):

        self._update_label_encoder_and_matrices_sizes(items_ids, contexts.shape[1])

        X_device = torch.tensor(contexts, device=self.device, dtype=torch.double)
        y_device = torch.tensor(rewards, device=self.device, dtype=torch.double)
        decisions_device = torch.tensor(self.label_encoder.transform(items_ids), device=self.device, dtype=torch.long)

        self.A.index_add_(0, decisions_device, torch.einsum('ni,nj->nij', X_device, X_device))

        self.Xty.index_add_(0, decisions_device, X_device * y_device.view(-1, 1))

        for j in range(0, self.beta.shape[0], self.items_per_batch):            
            self.beta[j:j+self.items_per_batch] = torch.linalg.solve(
                self.A[j:j+self.items_per_batch],
                self.Xty[j:j+self.items_per_batch]
            )

    def predict(self, contexts: np.ndarray):
        return torch.matmul(torch.tensor(contexts, device=self.device, dtype=torch.double), self.beta.T)