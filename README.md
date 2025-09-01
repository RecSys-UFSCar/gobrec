# gobrec
GOBRec: GPU Optimized Bandits Recommender

## Usage

### Using a MAB Algorithm individually to generate arm scores

```python
import numpy as np
from gobrec.mabs.lin_mabs import LinUCB                 # Import LinUCB as an example, it could be also LinTS or LinGreedy

contexts = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # A batch of contexts for training.
decisions = np.array(['a', 1, 2])                       # Corresponding decisions taken, it can be str or int
rewards = np.array([1, 0, 1])                           # Corresponding rewards received

bandit = LinUCB(seed=42, use_gpu=True)                  # Initialize the bandit. A seed is set for reproducibility and GPU usage can be on or off

bandit.fit(contexts, decisions, rewards)                # Fit the model with the training data

bandit.predict(np.array([[1, 1, 0], [0, 1, 1]]))        # Predict scores for each arm (item) given a batch of contexts
```

### Using a MAB Algorithm to generate recommendations

```python
import numpy as np
import gobrec
from gobrec.mabs.lin_mabs import LinUCB                 # Import LinUCB as an example, it could be also LinTS or LinGreedy

contexts = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # A batch of contexts for training.
decisions = np.array(['a', 1, 2])                       # Corresponding decisions taken, it can be str or int
rewards = np.array([1, 0, 1])                           # Corresponding rewards received

recommender = gobrec.Recommender(
    mab_algo=LinUCB(seed=42, use_gpu=True),             # The recommender can use any implementation following the MABAlgo interface
    top_k=2                                             # Number of items to recommend
)

recommender.fit(contexts, decisions, rewards)           # Fit the model with the training data

recommender.recommend(np.array([[1, 1, 0], [0, 1, 1]])) # Recommend top_k items given a batch of contexts
```
