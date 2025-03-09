# scPEGCSC
scPEGCSC: Proximity Enhanced Graph Convolutional Subspace Clustering method for scRNA-seq Data 
## Requirement
The python environment and the main packages needed to run scPEDSSC are as follows:

* python 3.8.16

* pytorch 2.0.0

* pandas 1.5.3

* scanpy 1.9.3

* scipy 1.10.1

* scikit-learn 1.2.2

The above python packages can be installed via Anaconda or pip commands.For example:

```
pip install scanpy 1.9.3
```

## Program execution process 
1.You need to download the datasets and codes locally.

2.Open the main.py file in the PyCharm client.Take the “X.h5” file as the input to the main.py program, and run it to get the results nmi and ari,where X is the specific name of the dataset.

Here,take the Human1 dataset as an example. 

Step1: Open the main.py file in the PyCharm client. Replace the first parameter of functoin "h5py.File" with files "Human1.h5". Run file main.py in the PyCharm client to get the final clustering results.

```python
data_mat = h5py.File('Human1.h5')
x = np.array(data_mat['X'])
y = np.array(data_mat['Y'])
data_mat.close()
```

The final output is as follows:

```
NMI 0.879 , ARI 0.883
```
