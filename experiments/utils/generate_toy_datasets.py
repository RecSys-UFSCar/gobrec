
import pandas as pd
import numpy as np
import os
from utils.constants import SEED, ITEM_ID_COLUMN, RATING_COLUMN

def load_or_generate_toy_dataset(num_items: int, num_interactions: int, save_path: str) -> tuple:
    """
    Loads a toy dataset from the specified path if it exists, otherwise generates it.

    Parameters:
        num_items (int): Number of items in the dataset.
        num_interactions (int): Number of interactions in the dataset.
        save_path (str): Path where the dataset is saved or will be saved.
    
    Returns:
        tuple: A tuple containing:
            - interactions (pd.DataFrame): DataFrame containing item IDs and their corresponding ratings.
            - contexts (np.ndarray): Numpy array containing contexts for each interaction.
    """
    if not (os.path.exists(f'{save_path}/interactions.csv') and os.path.exists(f'{save_path}/contexts.npy')):
        # Generate new dataset
        generate_and_save_toy_dataset(num_items, num_interactions, save_path)
    
    # Load existing dataset
    interactions = pd.read_csv(f'{save_path}/interactions.csv')
    contexts = np.load(f'{save_path}/contexts.npy')

    return interactions, contexts


def generate_and_save_toy_dataset(num_items: int, num_interactions: int, save_path: str):
    """
    Generates a toy dataset with the specified number of items and interactions,
    and saves it to the given path. This will generate the following files:

    - interactions.csv: Contains item and the reward of each interaction. The item IDs are random strings of 10 characters. Its garanteed that will be exactly num_items items in the dataset. The rewards are 0 or 1, with 0 having a 70% chance and 1 having a 30% chance.
    - contexts.npy: Contains the contexts for each interaction. The contexts are random arrays of 32 random floats between -5 and 5. The final shape of the array is (num_interactions, 32).

    Parameters:
        num_items (int): Number of items in the dataset.
        num_interactions (int): Number of interactions in the dataset.
        save_path (str): Path where the dataset will be saved.
    """
    os.makedirs(save_path, exist_ok=True)  # Ensure the save directory exists

    np.random.seed(SEED)  # For reproducibility

    # Generate item IDs
    items = []
    while len(items) < num_items:
        item_id = ''.join(np.random.choice(list('abcdefghijklmnopqrstuvwxyz0123456789'), size=10))
        if item_id not in items:  # Ensure unique item IDs
            items.append(item_id)
    
    # Generate interactions
    
    interactions = pd.DataFrame({
        ITEM_ID_COLUMN: np.concatenate([
            np.random.choice(items, size=num_interactions),
        ]),
        RATING_COLUMN: np.random.choice([0, 1], size=num_interactions, p=[0.7, 0.3])
    })
    
    # Save interactions to CSV
    interactions.to_csv(f'{save_path}/interactions.csv', index=False)
    
    # Generate contexts
    contexts = np.random.uniform(-5, 5, size=(num_interactions, 32))
    
    # Save contexts to .npy file
    np.save(f'{save_path}/contexts.npy', contexts)