"""Linear base for Multi-Armed Bandit (MAB) Algorithm Implementation."""
import torch
import numpy as np
from gobrec.mabs.mab_algo import MABAlgo


class Lin(MABAlgo):
    """Linear Multi-Armed Bandit (MAB) Algorithm.

    This class implements a linear MAB algorithm that uses ridge regression to
    estimate the expected rewards for each arm. This class is the base class
    for other linear MAB algorithms like LinUCB and LinTS. It can also be
    used directly, not having any exploration strategy (exploit only).

    Attributes
    ----------
    l2_lambda : float
        Regularization parameter for ridge regression.
    device : str
        Device to use for computations ('cpu' or 'cuda').
    items_per_batch : int
        Number of items to process in each batch when updating the model.
        More items per batch means more memory usage but faster computation.
    
    Examples
    --------
    A simple example using Lin to generate scores for items.

    >>> import numpy as np
    >>> from gobrec.mabs.lin_mabs import Lin
    >>> contexts = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1],
    ...                      [1, 0, 0], [0, 1, 0], [0, 0, 1],
    ...                      [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    >>> decisions = np.array(['a', 'a', 'a', 
    ...                       'b', 'b', 'b',
    ...                       'c', 'c', 'c'])
    >>> rewards = np.array([10, 0 , 1 , 
    ...                     1 , 10, 0 ,
    ...                     0 , 1 , 10])
    >>> lin_mab = Lin()
    >>> lin_mab.fit(contexts, decisions, rewards)
    >>> lin_mab.predict(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
    tensor([[5.0000, 0.5000, 0.0000],
            [0.0000, 5.0000, 0.5000],
            [0.5000, 0.0000, 5.0000]], dtype=torch.float64)
    """

    def __init__(self, seed: int = None, l2_lambda: float = 1.0, use_gpu: bool = False, items_per_batch: int = 10_000):
        """Initialize the Lin algorithm.

        Parameters
        ----------
        seed : int, optional
            Random seed for reproducibility. Default is None.
        l2_lambda : float, optional
            Regularization parameter for ridge regression. Default is 1.0.
        use_gpu : bool, optional
            Whether to use GPU for computations if available. Default is False.
        items_per_batch : int, optional
            Number of items to process in each batch when updating the model.
            More items per batch means more memory usage but faster computation. 
            Default is 10,000.
        """
        super().__init__(seed)

        self.l2_lambda = l2_lambda
        self.device = 'cuda' if use_gpu else 'cpu'

        self.items_per_batch = items_per_batch
        self.already_initialized = False
    
    def _update_label_encoder_and_matrices_sizes(self, decisions: np.ndarray, num_features: int):
        """Update the label encoder and initialize or resize internal matrices.

        This method updates the label encoder with new item IDs if any, sets the
        new number of arms, and checks for consistency in the number of features.
        It also initializes or resizes the internal matrices used for ridge
        regression. It is recommended to call this method at the beginning of
        the `fit` method.

        Parameters
        ----------
        decisions : np.ndarray
            A 1D array of item IDs, which can be strings or integers.
        num_features : int
            The number of features in the context vectors.
        """
        self._update_label_encoder(decisions, num_features)

        if not self.already_initialized:
            self.Xty = torch.zeros((self.num_arms, self.num_features), device=self.device, dtype=torch.double)
            self.A = torch.eye(self.num_features, device=self.device, dtype=torch.double).unsqueeze(0).repeat(self.num_arms, 1, 1) * self.l2_lambda
            self.beta = torch.zeros((self.num_arms, self.num_features), device=self.device, dtype=torch.double)
            self.already_initialized = True
        elif self.num_arms != self.beta.shape[0]:
            # TODO: maybe updating after a certain threshold of new arms is better?
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
        """Fit the Lin algorithm with contexts, decisions, and rewards.

        Parameters
        ----------
        contexts : np.ndarray
            A 2D array of shape (n_samples, n_features) representing the context arrays
        decisions : np.ndarray
            A 1D array of item IDs (arms or decisions) of shape (n_samples,)
            where each element can be strings or integers.
        rewards : np.ndarray
            A 1D array of rewards (ratings) of shape (n_samples,). It can be integers
            or floats.
        """
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
        """Predict the expected rewards for each arm given the contexts.

        Parameters
        ----------
        contexts : np.ndarray
            A 2D array where each row represents the context features for which
            predictions are to be made.
        
        Returns
        -------
        expected_rewards : torch.Tensor
            A 2D tensor of shape (n_samples, n_arms) where each element
            is the expected reward for the corresponding context-arm pair.
            The encoded items ids are used here. To get the original item IDs, 
            it is possible to use the `label_encoder.inverse_transform` method.
        """
        scores = torch.empty((contexts.shape[0], self.num_arms), device=self.device, dtype=torch.double)
        for start in range(0, self.num_arms, self.items_per_batch):
            end = min(start + self.items_per_batch, self.num_arms)  
            scores[:, start:end] = torch.einsum('bd,ad->ba', torch.tensor(contexts, device=self.device, dtype=torch.double), self.beta[start:end])
        
        return scores

    def reset(self):
        """Reset the Lin algorithm to its initial state.

        This method clears the labeled encoder and reinitializes the internal
        matrices used for ridge regression.

        After that the algorithm can be fitted again as if it was newly created.
        """
        super().reset()
        self.already_initialized = False
        self.Xty = None
        self.A = None
        self.beta = None