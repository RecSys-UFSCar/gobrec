
from gobrec.mabs.lin_mabs.Lin import Lin
import numpy as np
import torch


class Recommender:

    def __init__(self, mab_algo: Lin, top_k: int):
        self.mab_algo = mab_algo
        self.top_k = top_k
    
    def fit(self, contexts: np.ndarray, items_ids: np.ndarray, rewards: np.ndarray):
        self.mab_algo.fit(contexts, items_ids, rewards)
    
    def recommend(self, contexts: np.ndarray, items_ids_filter: 'list[np.ndarray, np.ndarray]' = None):
        # ITEMS IDS FILTERS is a tuple where the first element is a list of indices (of contexts) to filter and the second element is the items_ids to filter

        expectations = self.mab_algo.predict(contexts)

        if items_ids_filter is not None:
            items_ids_filter[1] = self.mab_algo.label_encoder.transform(items_ids_filter[1])
            expectations[items_ids_filter] = -100.

        topk_sorted_expectations = torch.topk(expectations, self.top_k, dim=1)
        recommendations = self.mab_algo.label_encoder.inverse_transform(topk_sorted_expectations.indices.cpu().numpy().flatten()).reshape(contexts.shape[0], self.top_k)
        scores = topk_sorted_expectations.values.cpu().numpy()
        
        return recommendations, scores