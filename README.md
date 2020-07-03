## All in the Exponential Family: Bregman Duality in Thermodynamic Variational Inference

 Code for [All in the Exponential Family: Bregman Duality in Thermodynamic Variational Inference](https://arxiv.org/)

```
@inproceedings{brekelmans2020allin,
  title     = {All in the Exponential Family: Bregman Duality in Thermodynamic Variational Inference},
  author    = {Rob Brekelmans and Vaden Masrani and Frank Wood and Greg Ver Steeg and Aram Galstyan},
  booktitle = {International Conference on Machine Learning},
  year      = {2020},
}
```


To download datasets, run scripts in ```data/``` folder

Example run script:
```
python3 main.py with loss='tvo' schedule='moments' dataset='mnist' S=50 K=10 batch_size=1000 epochs=1000 seed=2 record=True verbose=True
```
``` 
loss in ['tvo', 'tvo_reparam', 'iwae', 'iwae_dreg', 'elbo']
schedule in ['moments', 'coarse_grain', 'log', 'linear']
dataset in ['mnist', 'omniglot']
```
