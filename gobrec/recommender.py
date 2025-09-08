"""Implementation of the recommender that uses a MAB algorithm to recommend items based on contexts."""

from gobrec.mabs.mab_algo import MABAlgo
import numpy as np
import torch


class Recommender:
    """GOBRec recommender class.

    Recommender that uses a MAB algorithm to generate top-K recommendations
    based on contexts.

    Attributes
    ----------
    mab_algo : MABAlgo
        The multi-armed bandit algorithm used to generate item scores.
    top_k : int
        The number of items to be recommended. It will return the K 
        highest scored items.
    
    Examples
    --------
    A simple example using LinUCB as the MAB algorithm to recommend items.

    >>> import numpy as np
    >>> import gobrec
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
    >>> recommender = gobrec.Recommender(mab_algo=LinUCB(seed=42, use_gpu=True),top_k=2)
    >>> recommender.fit(contexts, decisions, rewards)
    >>> recommender.recommend(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
    (array([['a', 'b'],  # Top-2 recommendations for each context. Each value is an item ID.
            ['b', 'c'],
            ['c', 'a']]),
     array([[5.70710678, 1.20710678],   # Corresponding scores for each recommended item.
            [5.70710678, 1.20710678],
            [5.70710678, 1.20710678]]))

    Example using LinTS as the MAB algorithm to recommend items with item filtering.

    >>> import numpy as np
    >>> import gobrec
    >>> from gobrec.mabs.lin_mabs import LinTS
    >>> contexts = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1],
    ...                      [1, 0, 0], [0, 1, 0], [0, 0, 1],
    ...                      [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    >>> decisions = np.array(['a', 'a', 'a', 
    ...                       'b', 'b', 'b',
    ...                       'c', 'c', 'c'])
    >>> rewards = np.array([10, 0 , 1 , 
    ...                     1 , 10, 0 ,
    ...                     0 , 1 , 10])
    >>> recommender = gobrec.Recommender(mab_algo=LinTS(seed=42, use_gpu=True),top_k=2)
    >>> recommender.fit(contexts, decisions, rewards)
    >>> recommender.recommend(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
    ...                       decisions_filter=[[0, 1, 2], ['a', 'b', 'c']])
    ...                       # The first element is the indices of contexts to filter
    ...                       # The second element is the list of item IDs to filter
    ...                       # In this example, it will filter item a from context 0, 
    ...                       # item b from context 1 and item c from context 2
    (array([['b', 'c'],  # Top-2 recommendations for each context. Each value is an item ID.
            ['c', 'a'],
            ['a', 'b']]),
     array([[ 0.71546751,  0.21546751],    # Corresponding scores for each recommended item.
            [-0.87959021, -1.37959021],
            [ 0.48811979, -0.01188021]]))
    """

    def __init__(self, mab_algo: MABAlgo, top_k: int):
        """Initialize the GOBRec Recommender.

        Parameters
        ----------
        mab_algo : MABAlgo
            The multi-armed bandit algorithm used to generate item scores.
        top_k : int
            The number of items to be recommended. It will return the K 
            highest scored items.
        """
        self.mab_algo = mab_algo
        self.top_k = top_k
    
    def fit(self, contexts: np.ndarray, decisions: np.ndarray, rewards: np.ndarray):
        """Trains the MAB algorithm with the provided data.

        Parameters
        ----------
        contexts : np.ndarray
            A 2D array where each row represents the context features.
        decisions : np.ndarray
            A 1D array where each element is the item ID (decision) taken for the
            corresponding context.
        rewards : np.ndarray
            A 1D array where each element is the reward (rating) received for the 
            corresponding context-decision pair.
        """
        self.mab_algo.fit(contexts, decisions, rewards)
    
    def recommend(self, contexts: np.ndarray, decisions_filter: 'list[np.ndarray, np.ndarray]' = None):
        """Predicts top-K recommendations for the given contexts.

        Parameters
        ----------
        contexts : np.ndarray
            A 2D array where each row represents the context features for which
            recommendations are to be made.
        decisions_filter : list[np.ndarray, np.ndarray], optional
            A list containing two arrays:

            1. `contexts_indexes`: A 1D array of indices representing the contexts 
            for which certain items should be filtered out.
            2. `items_ids`: A 1D array of item IDs to be filtered out for 
            the corresponding contexts.

            For example, if `contexts_indexes` is `[0, 2]` and `items_ids` is
            `['itemA', 'itemB']`, then 'itemA' will be filtered out from the 
            recommendations for the context at index 0, and 'itemB' will be filtered 
            out for the context at index 2. `contexts_indexes` can contain repeated
            indices if multiple items need to be filtered for the same context.

            If `decisions_filter` is `None`, no items will be filtered from the
            recommendations.
        
        Returns
        -------
        recommendations : np.ndarray
            A 2D array where each row contains the top-K recommended item IDs for 
            the corresponding context.
        scores : np.ndarray
            A 2D array where each row contains the scores of the top-K recommended
            items for the corresponding context.

        """
        expectations = self.mab_algo.predict(contexts)

        if decisions_filter is not None:
            decisions_filter[1] = self.mab_algo.label_encoder.transform(decisions_filter[1])
            expectations[decisions_filter] = -100.

        topk_sorted_expectations = torch.topk(expectations, self.top_k, dim=1)
        recommendations = self.mab_algo.label_encoder.inverse_transform(topk_sorted_expectations.indices.cpu().numpy().flatten()).reshape(contexts.shape[0], self.top_k)
        scores = topk_sorted_expectations.values.cpu().numpy()
        
        return recommendations, scores

    def reset(self):
        """Reset the internal state of the MAB algorithm.

        This method is useful when you want to clear the learned parameters
        of the MAB algorithm and start fresh without any prior knowledge.

        This will make the recommender behave as if it was just initialized.
        """
        self.mab_algo.reset()