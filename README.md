# Decohering tensor network QML models
## Unitary Tree Tensor Network and MERA



Efficient unitary tree tensor network (TTN) and Multi-scale Entanglement Renormalization Ansatz (MERA) built with Tensorflow, with tunable local dephasing channels at every layer of the tensor networks and tunable number of ancillas, benchmarked on compressed MNIST, KMNIST, and Fashion-MNIST. Code comments included.

To setup, 
```
git clone https://github.com/HaoranLiao/dephased_ttn_mera.git
cd dephased_ttn_mera/
conda create -n tnqml python=3.8
conda activate tnqml
pip install -r requirements.txt
```
To use GPU,
```
pip install -r requirements_dev.txt
```

Add this to ```~/.bashrc```:

```export PYTHONPATH="${PYTHONPATH}:HOME_DIR/dephased_ttn_mera/"```

and ```source ~/.bashrc```.

$~$

To run the MERA,
```cd dephased_ttn_mera/mera```.


Configure ```config_example.yaml```, and run
```python model.py```


To run the unitary TTN,
```cd dephased_ttn_mera/uni_ttn/tf2```.


Configure ```config_example.yaml```, and run
```python model.py```

$~$

The main scripts to construct the tensor networks and to do the training are:
- ```data.py``` (for dataset loading and preprocessing), 
- ```model.py``` (define the workflow), 
- ```network.py``` (construct the network),
- under the folders ```cd dephased_ttn_mera/uni_ttn/tf2``` for unitary TTN and ```cd dephased_ttn_mera/mera``` for MERA

$~$

<!-- {% raw %} -->
Using the code please consider citing:
```
@article{Liao_2022,
  url = {https://arxiv.org/abs/2209.01195},
  author = {Liao, Haoran and Convy, Ian and Yang, Zhibo and Whaley, K. Birgitta},
  title = {Decohering Tensor Network Quantum Machine Learning Models},
  publisher = {arXiv:2209.01195},
  year = {2022}}
```

Reference: [Liao et al., Decohering Tensor Network Quantum Machine Learning Models, arXiv:2209.01195 (2022)](https://arxiv.org/abs/2209.01195)
<!-- {% endraw %} -->
