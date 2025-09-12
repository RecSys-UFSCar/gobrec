# GOBRec: GPU Optimized Bandits Recommender

GOBRec is a Python library with an optimized implementation of contextual multi-armed bandits (CMABs) for recommender systems. The library has a simple API that allows you to use the CMAB algorithms to generate item (arms expectations, using it for tasks other than recommendations. You can also use any of the implemented CMABs inside the Recommender to efficiently generate top-K recommendations.

The main contribution of GOBRec is its efficient implementation. With the vectorized code, using only CPU, our implementation was up to X times faster than other libraries. Using GPU optimization, our library achieved a speed gain of Y times. More details about these comparisons can be found in the ["performance comparison" section](#performance-comparison) or in our [paper]().

The GOBRec documentation is available at: [https://recsys-ufscar.github.io/gobrec](https://recsys-ufscar.github.io/gobrec/).

## Installation

GOBRec is available on [PyPI](https://pypi.org/project/gobrec/) and can be installed by the command below:

```
pip install gobrec
```

The recommended Python version to use is 3.8.20 (but newer versions should work too). For using GPU optimization, it is important to install PyTorch with CUDA implementation. More details on installing PyTorch with CUDA can be found in the [PyTorch documentation](https://pytorch.org/get-started/locally/). The recommended PyTorch version to use is 2.4.1.

More installation options can be found in the [documentation]().

## Performance comparison

EXPLICAR SIMPLIFICADAMENTE OS EXPERIMENTOS.

COLOCAR A TABELA COM OS RESULTADOS EM MARKDOWN

## Usage

This section shows two examples of how to use GOBRec. You can also use the available [Jupyter notebook](notebooks/usage_tutorial.ipynb) to reproduce the examples and verify the generated output.

### Using an MAB Algorithm individually to generate arm scores

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

### Using an MAB Algorithm to generate recommendations

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

## Available algorithms

LISTAR OS ALGORITMOS DISPONIVEIS E REFERENCIAS

## Contributing

LINKAR ISSUES
LINKAR PULL REQUESTS
LINKAR README COM INSTRUÇÕES DE DESENVOLVIMENTO

## License

LINKAR LICENSA

## Citation

COLOCAR CÓDIGO PARA CITAÇÃO

## References

COLOCAR AS REFERÊNCIAS UTILIZADAS
