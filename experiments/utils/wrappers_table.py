
import pandas as pd
from utils.mab2rec_wrappers import LinUCBMab2RecWrapper, LinGreedyMab2RecWrapper, LinTSMab2RecWrapper


WRAPPERS_TABLE = pd.DataFrame(
    [[1,     'mab2rec_LinUCB',      LinUCBMab2RecWrapper],
     [2,     'mab2rec_LinGreedy',   LinGreedyMab2RecWrapper],
     [3,     'mab2rec_LinTS',       LinTSMab2RecWrapper]],
    columns=['id', 'name', 'AlgoWrapper']
).set_index('id')