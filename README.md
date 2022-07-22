# Decohering tensor network QML models


To setup 
```
cd dephased_ttn_project/uni_ttn/tf2/dependency
conda env create -f environment_tf2.7.yaml
conda activate tf2
```

Add this to ```~/.bashrc```:

```export PYTHONPATH="${PYTHONPATH}:HOME_DIR/dephased_ttn_project/"```

and ```source ~/.bashrc```

To run, for example, the MERA,
```cd dephased_ttn_project/mera```


Configure ```config_example.yaml```, and run
```python model.py```
