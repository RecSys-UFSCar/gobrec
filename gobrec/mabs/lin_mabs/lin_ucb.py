"""LinUCB: Linear Upper Confidence Bound algorithm for Contextual Multi-Armed Bandits."""


from gobrec.mabs.lin_mabs import Lin
import numpy as np
import torch


class LinUCB(Lin):
    r"""LinUCB: Linear Upper Confidence Bound algorithm for Contextual Multi-Armed Bandits [1]_.
    
    This class implements a linear MAB algorithm that uses ridge regression to
    estimate the expected rewards for each arm. Then, it calculates the upper confidence bound
    for each arm based on the estimated coefficients and the design matrix.

    Attributes
    ----------
    alpha : float
        Controls the width of the confidence interval. Higher values lead to more exploration.
    l2_lambda : float
        Regularization parameter for ridge regression.
    device : str
        Device to use for computations ('cpu' or 'cuda').
    items_per_batch : int
        Number of items to process in each batch when updating the model.
        More items per batch means more memory usage but faster computation.

    References
    ----------
    .. [1] Lihong Li, Wei Chu, John Langford, and Robert E. Schapire. A contextual-bandit 
       approach to personalized news article recommendation. In Proceedings of the 19th 
       International Conference on World Wide Web, WWW'09, pages 661-670, New York, NY, 
       USA, 2010. Association for Computing Machinery. doi: 10.1145/1772690.1772758.
    
    Examples
    --------
    An example using LinUCB with :math:`\alpha = 0.5`.

    >>> import numpy as np
    >>> from gobrec.mabs.lin_mabs import LinUCB
    >>> contexts = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1],
    ...                      [1, 0, 0], [0, 1, 0], [0, 0, 1],
    ...                      [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    >>> decisions = np.array(['a', 'a', 'a', 
    ...                       'b', 'b', 'b',
    ...                       'c', 'c', 'c'])
    >>> rewards = np.array([10, 0 , 1 , 
    ...                     1 , 10, 0 ,
    ...                     0 , 1 , 10])
    >>> linucb_mab = LinUCB(seed=42, alpha=0.5)
    >>> linucb_mab.fit(contexts, decisions, rewards)
    >>> linucb_mab.predict(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
    tensor([[5.3536, 0.8536, 0.3536],
            [0.3536, 5.3536, 0.8536],
            [0.8536, 0.3536, 5.3536]], dtype=torch.float64)
    """

    def __init__(self, seed: int = None, alpha: float = 1.0, l2_lambda: float = 1.0, use_gpu: bool = False, items_per_batch: int = 10_000):
        """Initialize LinUCB algorithm.
        
        Parameters
        ----------
        seed : int, optional
            Random seed for reproducibility. Default is None.
        alpha : float, optional
            Controls the width of the confidence interval. More exploration 
            for higher values. Default is 1.0.
        l2_lambda : float, optional
            Regularization parameter for ridge regression. Default is 1.0.
        use_gpu : bool, optional
            Whether to use GPU for computations. Default is False.
        items_per_batch : int, optional
            Number of items to process in each batch when updating the model.
            More items per batch means more memory usage but faster computation.
            Default is 10,000.
        """
        super().__init__(seed, l2_lambda, use_gpu, items_per_batch)
        self.alpha = alpha


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

        scores = torch.matmul(x, self.beta.T)

        for j in range(0, self.beta.shape[0], self.items_per_batch):
            x_A_inv = torch.matmul(x, torch.linalg.inv(self.A[j: j+self.items_per_batch]))

            # Upper confidence bound = alpha * sqrt(x A^-1 xt). Notice that, x = xt
            # ucb values are claculated for all the contexts in one single go. type(ucb): np.ndarray
            ucb = self.alpha * torch.sqrt(torch.sum(x_A_inv * x, axis=2))

            # Calculate linucb expectation y = x * b + ucb
            scores[:, j: j+self.items_per_batch] += ucb.T
        
        return scores