# Quick start

This section shows two examples of how to use GOBRec. You can also use the available [Jupyter notebook](https://github.com/RecSys-UFSCar/gobrec/blob/main/notebooks/usage_tutorial.ipynb) to reproduce the examples and verify the generated output.

## Using an MAB Algorithm individually to generate arm scores

It is possible to generate item (arm) expectations by using an MAB Algorithm alone. That way it is possible to use these algorithms for others tasks than recommendation.

```python
import numpy as np
# Import LinUCB as an example, it could be also LinTS or LinGreedy
from gobrec.mabs.lin_mabs import LinUCB

# A batch of contexts for training
contexts = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
# Corresponding decisions (items) taken, it can be str or int
decisions = np.array(['a', 1, 2])
# Corresponding rewards (ratings) received                     
rewards = np.array([1, 0, 1])

# Initialize the bandit. A seed is set for reproducibility and GPU usage can be switched
bandit = LinUCB(seed=42, use_gpu=True)

# Fit the model with the training data
bandit.fit(contexts, decisions, rewards)

# Predict scores for each arm (item) given a batch of contexts
bandit.predict(np.array([[1, 1, 0], [0, 1, 1]]))
```

## Using an MAB Algorithm to generate recommendations

It is possible to use an MAB Algorithm with the Recommender class to efficiently generate top-K recommendations.

```python
import numpy as np
import gobrec
# Import LinUCB as an example, it could be also LinTS or LinGreedy
from gobrec.mabs.lin_mabs import LinUCB

# A batch of contexts for training.
contexts = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
# Corresponding decisions (items) taken, it can be str or int
decisions = np.array(['a', 1, 2])
# Corresponding rewards (ratings) received
rewards = np.array([1, 0, 1])

recommender = gobrec.Recommender(
    # The recommender can use any implementation following the MABAlgo interface
    mab_algo=LinUCB(seed=42, use_gpu=True),
    # Number of items to recommend
    top_k=2
)

# Fit the model with the training data
recommender.fit(contexts, decisions, rewards)

# Recommend top_k items given a batch of contexts
recommender.recommend(np.array([[1, 1, 0], [0, 1, 1]]))
```