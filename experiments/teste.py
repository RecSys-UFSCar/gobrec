
from utils.datasets import DATASETS_TABLE
from utils.wrappers_table import WRAPPERS_TABLE
from utils.constants import RESULTS_SAVE_PATH, NUM_EXECUTIONS_PER_EXPERIMENT, TRAIN_TIME_COLUMN, RECS_TIME_COLUMN, TOTAL_TIME_COLUMN, CONTEXTS_PER_BATCH
from utils.parameters_handle import get_input
from utils.BaseWrapper import BaseWrapper
from utils.load_ml_datasets import load_ml100k

import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from time import time

from utils.gobrec_wrappers import LinGobrecWrapper
from utils.mab2rec_wrappers import LinMab2RecWrapper

def execute_not_incremental_experiment(wrapper: BaseWrapper, interactions_df: pd.DataFrame, contexts: np.ndarray, save_path: str):

    split_index = int(len(interactions_df) * 0.5)

    train_df = interactions_df.copy()[:split_index]

    train_contexts = contexts[:split_index]
    test_contexts = contexts[split_index:]

    all_recs = []
    all_scores = []
    
    for _ in tqdm(range(2), desc='Executing not incremental experiment'):
        recs = []
        scores = []
        start_time = time()
        for start in range(0, len(train_contexts), CONTEXTS_PER_BATCH):
            if start == 0:
                wrapper.fit(train_df[start:start+CONTEXTS_PER_BATCH], train_contexts[start:start+CONTEXTS_PER_BATCH])
            else:
                wrapper.partial_fit(train_df[start:start+CONTEXTS_PER_BATCH], train_contexts[start:start+CONTEXTS_PER_BATCH])
        fit_time = time() - start_time

        start_time = time()
        for start in range(0, len(test_contexts), CONTEXTS_PER_BATCH):
            rec, score = wrapper.recommend(test_contexts[start:start+CONTEXTS_PER_BATCH])
            recs.append(rec)
            scores.append(score)
        recommend_time = time() - start_time

        full_time = fit_time + recommend_time

        wrapper.reset()

        all_recs.append(np.concatenate(recs))
        all_scores.append(np.concatenate(scores))
    
    return all_recs, all_scores
        

def execute_incremental_experiment(wrapper: BaseWrapper, interactions_df: pd.DataFrame, contexts: np.ndarray, save_path: str):
    NUM_WINDOWS = 10

    split_index = int(len(interactions_df) * 0.5)

    train_df = interactions_df.copy()[:split_index]
    test_df = interactions_df.copy()[split_index:]

    train_contexts = contexts[:split_index]
    test_contexts = contexts[split_index:]

    results = {}

    all_recs = []
    all_scores = []

    for _ in tqdm(range(2), desc='Executing incremental experiment'):

        results = {}

        start_time = time()

        recs = []
        scores = []
        
        for start in range(0, len(train_contexts), CONTEXTS_PER_BATCH):
            if start == 0:
                wrapper.fit(train_df[start:start+CONTEXTS_PER_BATCH], train_contexts[start:start+CONTEXTS_PER_BATCH])
            else:
                wrapper.partial_fit(train_df[start:start+CONTEXTS_PER_BATCH], train_contexts[start:start+CONTEXTS_PER_BATCH])
        fit_time = time() - start_time
        results[TRAIN_TIME_COLUMN] = fit_time

        for window_number in range(NUM_WINDOWS):

            current_window_start_index = int(len(test_df) * (window_number / NUM_WINDOWS))
            current_window_end_index = int(len(test_df) * ((window_number + 1) / NUM_WINDOWS))

            current_window_df = test_df.iloc[current_window_start_index:current_window_end_index]
            current_window_contexts = test_contexts[current_window_start_index:current_window_end_index]

            start_time = time()
            for start in range(0, len(current_window_contexts), CONTEXTS_PER_BATCH):
                rec, score = wrapper.recommend(current_window_contexts[start:start+CONTEXTS_PER_BATCH])
                recs.append(rec)
                scores.append(score)
            recommend_time = time() - start_time
            results[RECS_TIME_COLUMN + f'_{window_number+1}'] = recommend_time

            if window_number != NUM_WINDOWS - 1:
                start_time = time()
                for start in range(0, len(current_window_df), CONTEXTS_PER_BATCH):
                    wrapper.partial_fit(current_window_df[start:start+CONTEXTS_PER_BATCH], current_window_contexts[start:start+CONTEXTS_PER_BATCH])
                partial_fit_time = time() - start_time
                results[TRAIN_TIME_COLUMN + f'_{window_number+1}'] = partial_fit_time

        full_time = sum(results.values())
        results[TOTAL_TIME_COLUMN] = full_time
        wrapper.reset()

        all_recs.append(np.concatenate(recs))
        all_scores.append(np.concatenate(scores))

    return all_recs, all_scores


interactions_df, contexts = load_ml100k()
size = 100_000
interactions_df = interactions_df[:size]
contexts = contexts[:size]

gobrec_lin = LinGobrecWrapper()
mab2rec_lin = LinMab2RecWrapper()

incremental_recs_gobrec, incremental_scores_gobrec = execute_incremental_experiment(gobrec_lin, interactions_df, contexts, RESULTS_SAVE_PATH)
not_incremental_recs_gobrec, not_incremental_scores_gobrec = execute_not_incremental_experiment(gobrec_lin, interactions_df, contexts, RESULTS_SAVE_PATH)

incremental_recs_mab2rec, incremental_scores_mab2rec = execute_incremental_experiment(mab2rec_lin, interactions_df, contexts, RESULTS_SAVE_PATH)
not_incremental_recs_mab2rec, not_incremental_scores_mab2rec = execute_not_incremental_experiment(mab2rec_lin, interactions_df, contexts, RESULTS_SAVE_PATH)

# Is gobrec incremental recs is close to mab2rec incremental recs?
for i in range(len(incremental_recs_gobrec)):
    print(f'Incremental execution {i+1}:')
    print('GOBRec recs:', incremental_recs_gobrec[i][:10])
    print('MAB2Rec recs:', incremental_recs_mab2rec[i][:10])
    print('GOBRec scores:', incremental_scores_gobrec[i][:10])
    print('MAB2Rec scores:', incremental_scores_mab2rec[i][:10])
    print('Recs are equal:', np.array_equal(incremental_recs_gobrec[i], incremental_recs_mab2rec[i]))
    print('Scores are close:', np.allclose(incremental_scores_gobrec[i], incremental_scores_mab2rec[i], atol=1e-5))
    print()

# Is gobrec not incremental recs is close to mab2rec not incremental recs?
for i in range(len(not_incremental_recs_gobrec)):
    print(f'Not incremental execution {i+1}:')
    print('GOBRec recs:', not_incremental_recs_gobrec[i][:10])
    print('MAB2Rec recs:', not_incremental_recs_mab2rec[i][:10])
    print('GOBRec scores:', not_incremental_scores_gobrec[i][:10])
    print('MAB2Rec scores:', not_incremental_scores_mab2rec[i][:10])
    print('Recs are equal:', np.array_equal(not_incremental_recs_gobrec[i], not_incremental_recs_mab2rec[i]))
    print('Scores are close:', np.allclose(not_incremental_scores_gobrec[i], not_incremental_scores_mab2rec[i], atol=1e-5))
    print()