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
The compatible tensorflow versions should be around 2.4 - 2.7
For Apple M1, do the following,
```
conda install -c conda-forge tensorflow==2.6.0
conda install -c conda-forge tensorflow==2.4.0
```

Add this to ```~/.bashrc``` replacing the ```HOME_DIR```:

```export PYTHONPATH="${PYTHONPATH}:HOME_DIR/dephased_ttn_mera/"```

and ```source ~/.bashrc```.


To use GPUs,
```
pip install -r requirements-gpu.txt
```

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
 @article{Liao_2023,
  title={Decohering tensor network quantum machine learning models},
  author={Liao, Haoran and Convy, Ian and Yang, Zhibo and Whaley, K. Birgitta},
  journal={Quantum Machine Intelligence},
  volume={5},
  number={1},
  pages={7},
  year={2023},
  publisher={Springer},
  doi={https://doi.org/10.1007/s42484-022-00095-9},
}
```

Reference: [Liao et al., Decohering Tensor Network Quantum Machine Learning Models, Quantum Mach. Intell. 5(1), 7 (2023)](https://doi.org/10.1007/s42484-022-00095-9)
<!-- {% endraw %} -->
