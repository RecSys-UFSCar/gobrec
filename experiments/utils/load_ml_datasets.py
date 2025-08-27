
import os
import numpy as np
import pandas as pd
from tqdm import tqdm


def preprocess_mldata(ml_data_loader, save_path):
    
    interactions_df, items_df = ml_data_loader()
    genres = ['Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

    items_df = items_df[['item_id'] + genres]
    interactions_items_df = interactions_df.merge(items_df, on='item_id', how='left')
    interactions_items_df['genres_sum'] = 1

    cumsum = interactions_items_df.groupby('user_id')[['genres_sum'] + genres].cumsum()
    contexts = cumsum[genres].to_numpy() / cumsum['genres_sum'].to_numpy()[:, None]

    for user_id in tqdm(interactions_df['user_id'].unique(), desc='Loading ML dataset'):
        user_indices = interactions_df[interactions_df['user_id'] == user_id].index.to_numpy()
        user_contexts = contexts[user_indices]
        user_contexts_shifted = np.roll(user_contexts, shift=1, axis=0)
        user_contexts_shifted[0] = 0
        contexts[user_indices] = user_contexts_shifted
    
    os.makedirs(save_path, exist_ok=True)
    interactions_df[['item_id', 'rating']].to_csv(os.path.join(save_path, 'interactions.csv'), index=False)
    np.save(os.path.join(save_path, 'contexts.npy'), contexts)


def load_ml100k_raw_data():
    interactions = pd.read_csv('./datasets/ml-100k/u.data', sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp']).sort_values(by=['timestamp']).reset_index(drop=True)
    items = pd.read_csv('./datasets/ml-100k/u.item', sep='|', names=['item_id', 'title', 'release_date', 'video_release_date', 'IMDb_URL', 'unknown', 'Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'], encoding='latin-1')
    return interactions, items

def load_ml100k():
    if not os.path.exists('./datasets/ml-100k/preprocessed/interactions.csv') and not os.path.exists('./datasets/ml-100k/preprocessed/contexts.npy'):
        preprocess_mldata(load_ml100k_raw_data, './datasets/ml-100k/preprocessed')

    interactions = pd.read_csv('./datasets/ml-100k/preprocessed/interactions.csv')
    contexts = np.load('./datasets/ml-100k/preprocessed/contexts.npy')
    return interactions, contexts


def load_ml1m_raw_data():
    genres_map = {
        'Action': 0,
        'Adventure': 1,
        'Animation': 2,
        "Children's": 3,
        'Comedy': 4,
        'Crime': 5,
        'Documentary': 6,
        'Drama': 7,
        'Fantasy': 8,
        'Film-Noir': 9,
        'Horror': 10,
        'Musical': 11,
        'Mystery': 12,
        'Romance': 13,
        'Sci-Fi': 14,
        'Thriller': 15,
        'War': 16,
        'Western': 17
    }
    interactions = pd.read_csv('./datasets/ml-1m/ratings.dat', sep='::', names=['user_id', 'item_id', 'rating', 'timestamp'], engine='python').sort_values(by=['timestamp']).reset_index(drop=True)
    items = pd.read_csv('./datasets/ml-1m/movies.dat', sep='::', names=['item_id', 'title', 'genres'], engine='python', encoding='latin-1')
    for genre in genres_map.keys():
        items[genre] = items['genres'].apply(lambda x: 1 if genre in x.split('|') else 0)
    return interactions, items

def load_ml1m():
    if not os.path.exists('./datasets/ml-1m/preprocessed/interactions.csv') and not os.path.exists('./datasets/ml-1m/preprocessed/contexts.npy'):
        preprocess_mldata(load_ml1m_raw_data, './datasets/ml-1m/preprocessed')

    interactions = pd.read_csv('./datasets/ml-1m/preprocessed/interactions.csv')
    contexts = np.load('./datasets/ml-1m/preprocessed/contexts.npy')
    return interactions, contexts


def load_ml10m_raw_data():
    genres_map = {
        'Action': 0,
        'Adventure': 1,
        'Animation': 2,
        "Children's": 3,
        'Comedy': 4,
        'Crime': 5,
        'Documentary': 6,
        'Drama': 7,
        'Fantasy': 8,
        'Film-Noir': 9,
        'Horror': 10,
        'Musical': 11,
        'Mystery': 12,
        'Romance': 13,
        'Sci-Fi': 14,
        'Thriller': 15,
        'War': 16,
        'Western': 17
    }
    interactions = pd.read_csv('./datasets/ml-10m/ratings.dat', sep='::', names=['user_id', 'item_id', 'rating', 'timestamp'], engine='python').sort_values(by=['timestamp']).reset_index(drop=True)
    items = pd.read_csv('./datasets/ml-10m/movies.dat', sep='::', names=['item_id', 'title', 'genres'], engine='python', encoding='latin-1')
    for genre in genres_map.keys():
        items[genre] = items['genres'].apply(lambda x: 1 if genre in x.split('|') else 0)
    return interactions, items

def load_ml10m():
    if not os.path.exists('./datasets/ml-10m/preprocessed/interactions.csv') and not os.path.exists('./datasets/ml-10m/preprocessed/contexts.npy'):
        preprocess_mldata(load_ml10m_raw_data, './datasets/ml-10m/preprocessed')

    interactions = pd.read_csv('./datasets/ml-10m/preprocessed/interactions.csv')
    contexts = np.load('./datasets/ml-10m/preprocessed/contexts.npy')
    return interactions, contexts