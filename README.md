# GOBRec: GPU Optimized Bandits Recommender

GOBRec is a Python library with optimized implementation of contextual multi-armed bandits (CMABs) for recommender systems. The library has a simple API where you can use the CMABs algorithms to generate items (arms) expectations, using it for other tasks than recommendations. You can also use any of the implemented CMABs inside the Recommender to efficiently generate top-K recommendations.

The main contribuition of GOBRec is its efficient implementation. With the vectorized code, using only CPU, our implementation was up to X times faster than other libraries. Using the GPU optimization, our library was able to obtain a speed gain of Y times. More details about these comparisions can be found in the ["performance comparision" section](#performance-comparison) or in our [paper]().

The GOBRec documentation is available at: [https://recsys-ufscar.github.io/gobrec](https://recsys-ufscar.github.io/gobrec/).

## Installation

INSTALACAO COM O PIP INSTALL.
FALAR SOBRE CUDA E TORCH PARA PODER USAR A OTIMIZACAO DE GPU
LINK PARA MAIS DETALHES NA DOCUMENTACAO

## Performance comparison

EXPLICAR SIMPLIFICADAMENTE OS EXPERIMENTOS.

COLOCAR A TABELA COM OS RESULTADOS EM MARKDOWN

## Usage

EXPLICAR MELHOR CADA EXEMPLO

### Using a MAB Algorithm individually to generate arm scores

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

### Using a MAB Algorithm to generate recommendations

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
