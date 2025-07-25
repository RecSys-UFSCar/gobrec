
from utils.datasets import DATASETS_TABLE
from utils.parameters_handle import get_input




datasets_options = get_input(
    'Select the options to be used in the experiments',
    [
        {
            'name': 'datasets',
            'description': 'Datasets to be used in the experiments',
            'name_column': 'name',
            'options': DATASETS_TABLE
        },
    ]
)

datasets_options = datasets_options[0]

for dataset_option in datasets_options:
    dataset_name = DATASETS_TABLE.loc[dataset_option, 'name']
    dataset_getter = DATASETS_TABLE.loc[dataset_option, 'get_dataset']
    print(f'Loading dataset {dataset_name}...')
    interactions_df, contexts = dataset_getter()
    print(f'Dataset {dataset_name} loaded...')