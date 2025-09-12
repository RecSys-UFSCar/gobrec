"""LinGreedy: Linear Contextual Bandit with Epsilon-Greedy Exploration."""


from gobrec.mabs.lin_mabs import Lin
import numpy as np
import torch


class LinGreedy(Lin):
    r"""LinGreedy: Linear Contextual Bandit with Epsilon-Greedy Exploration [1]_.
    
    This class implements a linear MAB algorithm that uses ridge regression to
    estimate the expected rewards for each arm. Then, with probability :math:`\epsilon`,
    the generated scores are random (exploration), and with probability :math:`1 - \epsilon`,
    the generated scores are the expected rewards (exploitation).

    Attributes
    ----------
    epsilon : float
        Probability of choosing a random action (exploration). Value should be in [0, 1].
        1 means always explore, 0 means always exploit.
    l2_lambda : float
        Regularization parameter for ridge regression.
    device : str
        Device to use for computations ('cpu' or 'cuda').
    items_per_batch : int
        Number of items to process in each batch when updating the model.
        More items per batch means more memory usage but faster computation.
    
    References
    ----------
    .. [1] John Langford and Tong Zhang. The epoch-greedy algorithm for contextual multi-armed
       bandits. In Proceedings of the 20th International Conference on Neural Information Pro-
       cessing Systems, NIPS'07, pages 817-824, Red Hook, NY, USA, 2007. Curran Associates
       Inc. doi: 10.5555/2981562.2981665.
    
    Examples
    --------
    An example using LinGreedy with :math:`\epsilon = 1`, which means always explore.

    >>> import numpy as np
    >>> from gobrec.mabs.lin_mabs import LinGreedy
    >>> contexts = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1],
    ...                      [1, 0, 0], [0, 1, 0], [0, 0, 1],
    ...                      [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    >>> decisions = np.array(['a', 'a', 'a', 
    ...                       'b', 'b', 'b',
    ...                       'c', 'c', 'c'])
    >>> rewards = np.array([10, 0 , 1 , 
    ...                     1 , 10, 0 ,
    ...                     0 , 1 , 10])
    >>> lin_greedy_mab = LinGreedy(seed=42, epsilon=1)
    >>> lin_greedy_mab.fit(contexts, decisions, rewards)
    >>> lin_greedy_mab.predict(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
    tensor([[0.6974, 0.0942, 0.9756],
            [0.7611, 0.7861, 0.1281],
            [0.4504, 0.3708, 0.9268]], dtype=torch.float64)
    >>> lin_greedy_mab.predict(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
    tensor([[0.2272, 0.5546, 0.0638],
            [0.8276, 0.6317, 0.7581],
            [0.3545, 0.9707, 0.8931]], dtype=torch.float64)
    
    An example using LinGreedy with :math:`\epsilon = 0`, which means always exploit.

    >>> import numpy as np
    >>> from gobrec.mabs.lin_mabs import LinGreedy
    >>> contexts = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1],
    ...                      [1, 0, 0], [0, 1, 0], [0, 0, 1],
    ...                      [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    >>> decisions = np.array(['a', 'a', 'a', 
    ...                       'b', 'b', 'b',
    ...                       'c', 'c', 'c'])
    >>> rewards = np.array([10, 0 , 1 , 
    ...                     1 , 10, 0 ,
    ...                     0 , 1 , 10])
    >>> lin_greedy_mab = LinGreedy(seed=42, epsilon=0)
    >>> lin_greedy_mab.fit(contexts, decisions, rewards)
    >>> lin_greedy_mab.predict(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
    tensor([[5.0000, 0.5000, 0.0000],
            [0.0000, 5.0000, 0.5000],
            [0.5000, 0.0000, 5.0000]], dtype=torch.float64)
    >>> lin_greedy_mab.predict(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
    tensor([[5.0000, 0.5000, 0.0000],
            [0.0000, 5.0000, 0.5000],
            [0.5000, 0.0000, 5.0000]], dtype=torch.float64)
    """

    def __init__(self, seed: int = None, epsilon: float = 0.1, l2_lambda: float = 1.0, use_gpu: bool = False, items_per_batch: int = 10_000):
        """Initialize the LinGreedy algorithm.

        Parameters
        ----------
        seed : int, optional
            Random seed for reproducibility. Default is None.
        epsilon : float, optional
            Probability of choosing a random action (exploration). Value should be in [0, 1].
            1 means always explore, 0 means always exploit. Default is 0.1.
        l2_lambda : float, optional
            Regularization parameter for ridge regression. Default is 1.0.
        use_gpu : bool, optional
            Whether to use GPU for computations if available. Default is False.
        items_per_batch : int, optional
            Number of items to process in each batch when updating the model.
            More items per batch means more memory usage but faster computation.
            Default is 10,000.
        """
        super().__init__(seed, l2_lambda, use_gpu, items_per_batch)
        self.epsilon = epsilon


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
