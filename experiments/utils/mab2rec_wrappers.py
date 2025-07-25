
from mab2rec import BanditRecommender, LearningPolicy
from utils.constants import TOP_K, SEED, L2_LAMBDA, LINUCB_ALPHA, LINGREEDY_EPSILON, LINTS_ALPHA
from utils.BaseWrapper import BaseWrapper
import pandas as pd
import numpy as np


class BaseMab2recWrapper(BaseWrapper):

    def __init__(self):
        self.mab2rec_recommender = BanditRecommender(
            learning_policy=self.mab2rec_learning_policy,  # This should be set in the subclass
            top_k=TOP_K,
            seed=SEED
        )
    
    def fit(self, interactions_df: pd.DataFrame, contexts: np.ndarray):
        """
        Fit the MAB2Rec model to the interactions data.
        
        Parameters:
            interactions_df (pd.DataFrame): DataFrame containing 'item_id' and 'reward'.
            contexts (np.ndarray): Numpy array containing contexts for each interaction.
        """
        self.mab2rec_recommender.fit(
            decisions=interactions_df['item_id'],
            rewards=interactions_df['reward'],
            contexts=contexts
        )
    
    def partial_fit(self, interactions_df: pd.DataFrame, contexts: np.ndarray):
        """
        Incrementally fit the MAB2Rec model with new interactions data.
        
        Parameters:
            interactions_df (pd.DataFrame): DataFrame containing 'item_id' and 'reward'.
            contexts (np.ndarray): Numpy array containing contexts for each interaction.
        """
        self.mab2rec_recommender.partial_fit(
            decisions=interactions_df['item_id'],
            rewards=interactions_df['reward'],
            contexts=contexts
        )
    
    def recommend(self, contexts: np.ndarray):
        """
        Recommend items based on the current model state and contexts.
        
        Parameters:
            contexts (np.ndarray): Numpy array containing contexts for each interaction.
        
        Returns:
            list: List of recommended item IDs.
        """
        return self.mab2rec_recommender.recommend(contexts=contexts)
        

class LinUCBMab2RecWrapper(BaseMab2recWrapper):
    """
    Wrapper for the LinUCB algorithm in MAB2Rec.
    """
    def __init__(self):
        self.mab2rec_learning_policy = LearningPolicy.LinUCB(l2_lambda=L2_LAMBDA, alpha=LINUCB_ALPHA)
        super().__init__()

class LinGreedyMab2RecWrapper(BaseMab2recWrapper):
    """
    Wrapper for the LinGreedy algorithm in MAB2Rec.
    """
    def __init__(self):
        self.mab2rec_learning_policy = LearningPolicy.LinGreedy(l2_lambda=L2_LAMBDA, epsilon=LINGREEDY_EPSILON)
        super().__init__()

class LinTSMab2RecWrapper(BaseMab2recWrapper):
    """
    Wrapper for the LinTS algorithm in MAB2Rec.
    """
    def __init__(self):
        self.mab2rec_learning_policy = LearningPolicy.LinTS(l2_lambda=L2_LAMBDA, alpha=LINTS_ALPHA)
        super().__init__()