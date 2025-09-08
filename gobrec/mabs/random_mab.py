"""Random Multi-Armed Bandit (MAB) Algorithm Implementation."""
import torch
import numpy as np
from gobrec.mabs.mab_algo import MABAlgo


class RandomMAB(MABAlgo):
    """Random Multi-Armed Bandit (MAB) Algorithm.
    
    This class implements a random MAB algorithm that generates random scores for each arm.
    It can be used as a baseline for comparison with more sophisticated MAB algorithms.

    Examples
    --------
    A simple example using RandomMAB to generate random scores for items.

    >>> import numpy as np
    >>> from gobrec.mabs import RandomMAB
    >>> random_mab = RandomMAB(seed=2)
    >>> random_mab.fit(
    ...    contexts=np.array([[0, 1],[0, 1], [1, 0], [1, 0]]),
    ...    decisions=np.array([0, 1, 0, 1]),
    ...    rewards=np.array([1, 0, 0, 1])
    ... )
    >>> random_mab.predict(np.array([[0, 1],[0, 1], [1, 0], [1, 0]]))
    tensor([[0.2616, 0.2985],
            [0.8142, 0.0919],
            [0.6001, 0.7286],
            [0.1879, 0.0551]], dtype=torch.float64)
    """

    def __init__(self, seed: int = None):
        """Initialize the RandomMAB algorithm.

        Parameters
        ----------
        seed : int, optional
            Random seed for reproducibility. Default is None.
        """
        super().__init__(seed)

    def fit(self, contexts: np.ndarray, decisions: np.ndarray, rewards: np.ndarray):
        """Fit the RandomMAB algorithm with contexts, decisions, and rewards.

        This method just updates the label encoder and internal state. There is
        no actual learning in this random algorithm.

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
        self._update_label_encoder(decisions, contexts.shape[1])

    def predict(self, contexts: np.ndarray):
        """Predict random scores for each arm given the contexts.
        
        Parameters
        ----------
        contexts : np.ndarray
            A 2D array where each row represents the context features for which
            predictions are to be made.
        
        Returns
        -------
        scores : torch.Tensor
            A 2D tensor of shape (n_samples, n_arms) where each element is a random
            score for the corresponding context-arm pair.

        """
        return torch.from_numpy(self.rng.random((contexts.shape[0], self.num_arms))).double()
    
    def reset(self):
        """Reset the RandomMAB algorithm to its initial state.
        
        This method clears the labeled encoder and reinitializes the random
        number generator with the original seed.
        """
        super().reset()