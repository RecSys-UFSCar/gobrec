
from utils.datasets import DATASETS_TABLE
from utils.wrappers_table import WRAPPERS_TABLE

from utils.parameters_handle import get_input

import pandas as pd




def execute_not_incremental_experiment(wrapper, interactions_df, contexts):
    pass

def execute_incremental_experiment(wrapper, interactions_df, contexts):
    pass 






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
            experiment_function(wrapper, interactions_df, contexts)
            print(f'Experiment {experiment_name} completed.')