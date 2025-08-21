
from gobrec.gobrec.lin_mabs.Lin import Lin
import numpy as np
import torch


class Recommender:

    def __init__(self, mab_algo: Lin, top_k: int):
        self.mab_algo = mab_algo
        self.top_k = top_k
    
    def fit(self, contexts: np.ndarray, items_ids: np.ndarray, rewards: np.ndarray):
        self.mab_algo.fit(contexts, items_ids, rewards)
    
    def recommend(self, contexts: np.ndarray, items_ids_filter: list = None):

        expectations = self.mab_algo.predict(contexts)

        items_ids_filter = self.mab_algo.label_encoder.transform(items_ids_filter) if items_ids_filter is not None else None

        if items_ids_filter is not None:
            expectations[items_ids_filter] = -100.

        topk_sorted_expectations = torch.topk(expectations, self.top_k, dim=1)
        recommendations = self.mab_algo.label_encoder.inverse_transform(topk_sorted_expectations.indices.cpu().numpy())
        scores = topk_sorted_expectations.values.cpu().numpy()
        
        return recommendations, scores