
from gobrec.mabs.MABAlgo import MABAlgo
import numpy as np
import torch


class Recommender:

    def __init__(self, mab_algo: MABAlgo, top_k: int):
        self.mab_algo = mab_algo
        self.top_k = top_k
    
    def fit(self, contexts: np.ndarray, decisions: np.ndarray, rewards: np.ndarray):
        self.mab_algo.fit(contexts, decisions, rewards)
    
    def recommend(self, contexts: np.ndarray, decisions_filter: 'list[np.ndarray, np.ndarray]' = None):
        # ITEMS IDS FILTERS is a tuple where the first element is a list of indices (of contexts) to filter and the second element is the items_ids to filter

        expectations = self.mab_algo.predict(contexts)

        if decisions_filter is not None:
            decisions_filter[1] = self.mab_algo.label_encoder.transform(decisions_filter[1])
            expectations[decisions_filter] = -100.

        topk_sorted_expectations = torch.topk(expectations, self.top_k, dim=1)
        recommendations = self.mab_algo.label_encoder.inverse_transform(topk_sorted_expectations.indices.cpu().numpy().flatten()).reshape(contexts.shape[0], self.top_k)
        scores = topk_sorted_expectations.values.cpu().numpy()
        
        return recommendations, scores

    def reset(self):
        self.mab_algo.reset()