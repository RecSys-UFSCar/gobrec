
import pandas as pd
import numpy as np
import os


def generate_toy_datasets_if_they_do_not_exist(save_dir: str):
    '''
    Generates varios datasets with different number of items and interactions. It will generate the dataset with the combination of the following quantities of items and interactions:

    - items: 1k, 10k, 100k
    - interactions: 1M, 5M, 15M, 30M

    If the datasets already exist, it will not generate them again.

    Parameters:
        save_dir (str): Directory where the datasets will be saved. The datasets will be saved in the following format: <save_dir>/toy_dataset_<num_items>_<num_interactions>. The datasets will be saved in the following format:
            - interactions.csv: Contains item and the reward of each interaction. The item IDs are random strings of 10 characters. Its garanteed that will be exactly num_items items in the dataset. The rewards are 0 or 1, with 0 having a 70% chance and 1 having a 30% chance.
            - contexts.npy: Contains the contexts for each interaction. The contexts are random arrays of 32 random floats between -5 and 5. The final shape of the array is (num_interactions, 32).
    '''
    items_quantities_and_name = {
        1_000: '1k',
        10_000: '10k',
        100_000: '100k'
    }
    interactions_quantities_and_name = {
        1_000_000: '1M',
        5_000_000: '5M',
        15_000_000: '15M',
        30_000_000: '30M'
    }

    for num_items, items_name in items_quantities_and_name.items():
        for num_interactions, interactions_name in interactions_quantities_and_name.items():
            dataset_name = f'toy_dataset_{items_name}_{interactions_name}'
            dataset_path = os.path.join(save_dir, dataset_name)
            if not (os.path.exists(f'{dataset_path}/interactions.csv') and os.path.exists(f'{dataset_path}/contexts.npy')):
                print(f'Generating dataset {dataset_name}...')
                generate_and_save_toy_dataset(num_items, num_interactions, dataset_path)
            else:
                print(f'Dataset {dataset_name} already exists. Skipping generation.')

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
    # Generate item IDs
    items = []
    while len(items) < num_items:
        item_id = ''.join(np.random.choice(list('abcdefghijklmnopqrstuvwxyz0123456789'), size=10))
        if item_id not in items:  # Ensure unique item IDs
            items.append(item_id)
    
    # Generate interactions
    
    interactions = pd.DataFrame({
        'item': np.concatenate([
            items,  # Ensure all items are included
            np.random.choice(items, size=num_interactions - len(items)),
        ]),
        'reward': np.random.choice([0, 1], size=num_interactions, p=[0.7, 0.3])
    })
    
    # Save interactions to CSV
    interactions.to_csv(f'{save_path}/interactions.csv', index=False)
    
    # Generate contexts
    contexts = np.random.uniform(-5, 5, size=(num_interactions, 32))
    
    # Save contexts to .npy file
    np.save(f'{save_path}/contexts.npy', contexts)