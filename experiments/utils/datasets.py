
import os
import pandas as pd
from utils.generate_toy_datasets import load_or_generate_toy_dataset

from utils.constants import TOY_DATASETS_SAVE_PATH

DATASETS_TABLE = pd.DataFrame(
    [[1,     'toy_dataset_1k_1M',      lambda: load_or_generate_toy_dataset(1_000,   1_000_000,  os.path.join(TOY_DATASETS_SAVE_PATH, 'toy_dataset_1k_1M'))],
     [2,     'toy_dataset_1k_5M',      lambda: load_or_generate_toy_dataset(1_000,   5_000_000,  os.path.join(TOY_DATASETS_SAVE_PATH, 'toy_dataset_1k_5M'))],
     [3,     'toy_dataset_1k_15M',     lambda: load_or_generate_toy_dataset(1_000,   15_000_000, os.path.join(TOY_DATASETS_SAVE_PATH, 'toy_dataset_1k_15M'))],
     [4,     'toy_dataset_1k_30M',     lambda: load_or_generate_toy_dataset(1_000,   30_000_000, os.path.join(TOY_DATASETS_SAVE_PATH, 'toy_dataset_1k_30M'))],
     [5,     'toy_dataset_10k_1M',     lambda: load_or_generate_toy_dataset(10_000,  1_000_000,  os.path.join(TOY_DATASETS_SAVE_PATH, 'toy_dataset_10k_1M'))],
     [6,     'toy_dataset_10k_5M',     lambda: load_or_generate_toy_dataset(10_000,  5_000_000,  os.path.join(TOY_DATASETS_SAVE_PATH, 'toy_dataset_10k_5M'))],
     [7,     'toy_dataset_10k_15M',    lambda: load_or_generate_toy_dataset(10_000,  15_000_000, os.path.join(TOY_DATASETS_SAVE_PATH, 'toy_dataset_10k_15M'))],
     [8,     'toy_dataset_10k_30M',    lambda: load_or_generate_toy_dataset(10_000,  30_000_000, os.path.join(TOY_DATASETS_SAVE_PATH, 'toy_dataset_10k_30M'))],
     [9,     'toy_dataset_100k_1M',    lambda: load_or_generate_toy_dataset(100_000, 1_000_000,  os.path.join(TOY_DATASETS_SAVE_PATH, 'toy_dataset_100k_1M'))],
     [10,    'toy_dataset_100k_5M',    lambda: load_or_generate_toy_dataset(100_000, 5_000_000,  os.path.join(TOY_DATASETS_SAVE_PATH, 'toy_dataset_100k_5M'))],
     [11,    'toy_dataset_100k_15M',   lambda: load_or_generate_toy_dataset(100_000, 15_000_000, os.path.join(TOY_DATASETS_SAVE_PATH, 'toy_dataset_100k_15M'))],
     [12,    'toy_dataset_100k_30M',   lambda: load_or_generate_toy_dataset(100_000, 30_000_000, os.path.join(TOY_DATASETS_SAVE_PATH, 'toy_dataset_100k_30M'))]],
    columns=['id', 'name', 'get_dataset']
).set_index('id')