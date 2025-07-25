
import pandas as pd
import numpy as np

class BaseWrapper:
    
    def fit(self, interactions_df: pd.DataFrame, contexts: np.ndarray):
        """
        Fit the model to the interactions data.
        
        Parameters:
            interactions_df (pd.DataFrame): DataFrame containing 'item_id' and 'reward'.
            contexts (np.ndarray): Numpy array containing contexts for each interaction.
        """
        raise NotImplementedError("This method should be implemented in subclasses.")
    
    def partial_fit(self, interactions_df: pd.DataFrame, contexts: np.ndarray):
        """
        Incrementally fit the model with new interactions data.
        
        Parameters:
            interactions_df (pd.DataFrame): DataFrame containing 'item_id' and 'reward'.
            contexts (np.ndarray): Numpy array containing contexts for each interaction.
        """
        raise NotImplementedError("This method should be implemented in subclasses.")
    
    def recommend(self, contexts: np.ndarray):
        """
        Recommend items based on the current model state and contexts.
        
        Parameters:
            contexts (np.ndarray): Numpy array containing contexts for each interaction.
        
        Returns:
            list: List of recommended item IDs.
        """
        raise NotImplementedError("This method should be implemented in subclasses.")