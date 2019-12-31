# pytorch-ract
PyTorch Implementation of Ranking-Critical Training for Collaborative Filtering

## FILES:
* `RaCT.py` : PyTorch Implementation of Ranking-Critical Training for Collaborative Filtering algorithm.
* `models.py`: Implementation of Variational Auto-Encoders and a Custom NN(Critic)
* `evaluation_metrics.py` : Borrowed&modified [implementation](https://github.com/samlobel/RaCT_CF/blob/master/utils/evaluation_functions.py) of NDCG metric for ranking.
* `RaCT-CF_Implementation_Test.ipynb` : Notebook that tests and discusses this implementation. Note that data preparations are borrowed from[this repository](https://github.com/dawenl/vae_cf)

## TODO:
* CUDA
* Better Performance Visualization
* Solving the problems discussed in the notebook
