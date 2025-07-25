
from utils.datasets import DATASETS_TABLE
from utils.wrappers_table import WRAPPERS_TABLE
from utils.constants import RESULTS_SAVE_PATH, NUM_EXECUTIONS_PER_EXPERIMENT, TRAIN_TIME_COLUMN, RECS_TIME_COLUMN, TOTAL_TIME_COLUMN
from utils.parameters_handle import get_input
from utils.BaseWrapper import BaseWrapper

import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from time import time



def execute_not_incremental_experiment(wrapper: BaseWrapper, interactions_df: pd.DataFrame, contexts: np.ndarray, save_path: str):
    num_necessary_executions = NUM_EXECUTIONS_PER_EXPERIMENT
    df_save_path = os.path.join(save_path, 'not_incremental.csv')
    if os.path.exists(df_save_path):
        num_necessary_executions -= pd.read_csv(os.path.join(save_path, 'not_incremental.csv')).shape[0]
    
    split_index = int(len(interactions_df) * 0.5)

    train_df = interactions_df.copy()[:split_index]

    train_contexts = contexts[:split_index]
    test_contexts = contexts[split_index:]
    
    for _ in tqdm(range(num_necessary_executions), desc='Executing not incremental experiment'):
        start_time = time()
        wrapper.fit(train_df, train_contexts)
        fit_time = time() - start_time

        start_time = time()
        wrapper.recommend(test_contexts)
        recommend_time = time() - start_time

        full_time = fit_time + recommend_time

        results = {
            TRAIN_TIME_COLUMN: fit_time,
            RECS_TIME_COLUMN: recommend_time,
            TOTAL_TIME_COLUMN: full_time
        }
        results_df = pd.DataFrame([results])
        results_df.to_csv(df_save_path, mode='a', header=not os.path.exists(df_save_path), index=False)
        

def execute_incremental_experiment(wrapper: BaseWrapper, interactions_df: pd.DataFrame, contexts: np.ndarray, save_path: str):
    pass 


def get_experiment_save_path(dataset_name, wrapper_name):
    save_path = os.path.join(RESULTS_SAVE_PATH, dataset_name, wrapper_name)
    os.makedirs(save_path, exist_ok=True)
    return save_path




EXPERIMENTS_TABLE = pd.DataFrame(
    [[1, 'not_incremental', execute_not_incremental_experiment],
     [2, 'incremental', execute_incremental_experiment]],
    columns=['id', 'name', 'experiment_function']
).set_index('id')


datasets_options, wrappers_options, experiments_options = get_input(
    'Select the options to be used in the experiments',
    [
        {
            'name': 'datasets',
            'description': 'Datasets to be used in the experiments',
            'name_column': 'name',
            'options': DATASETS_TABLE
        },
        {
            'name': 'wrappers',
            'description': 'Wrappers to be used in the experiments',
            'name_column': 'name',
            'options': WRAPPERS_TABLE
        },
        {
            'name': 'experiments',
            'description': 'Experiments to be executed',
            'name_column': 'name',
            'options': EXPERIMENTS_TABLE
        }
    ]
)

for dataset_option in datasets_options:
    dataset_name = DATASETS_TABLE.loc[dataset_option, 'name']
    dataset_getter = DATASETS_TABLE.loc[dataset_option, 'get_dataset']
    print(f'Loading dataset {dataset_name}...')
    interactions_df, contexts = dataset_getter()
    print(f'Dataset {dataset_name} loaded...')

    for wrapper_option in wrappers_options:
        wrapper_name = WRAPPERS_TABLE.loc[wrapper_option, 'name']
        WrapperClass = WRAPPERS_TABLE.loc[wrapper_option, 'AlgoWrapper']
        print(f'Using wrapper {wrapper_name}...')
        wrapper = WrapperClass()

        for experiment_option in experiments_options:
            experiment_name = EXPERIMENTS_TABLE.loc[experiment_option, 'name']
            experiment_function = EXPERIMENTS_TABLE.loc[experiment_option, 'experiment_function']
            print(f'Executing experiment {experiment_name} with wrapper {wrapper_name} on dataset {dataset_name}...')
            experiment_function(wrapper, interactions_df, contexts, get_experiment_save_path(dataset_name, wrapper_name))
            print(f'Experiment {experiment_name} completed.')