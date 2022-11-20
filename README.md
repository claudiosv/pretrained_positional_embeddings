# Towards Learned Positional Encodings (MAE-Derived Positional Embeddings)

In this repository, you will find several notebooks and python scripts related to the MAE-Derived Positional Embedding research project.

The various tasks and code to be used for replication of this work are described and detailed below. The relevant files and notebooks to be examined are listed below, other files include requirements/configuration, or supplementary project work.

> **Note**
> During the project creation and evaluation, the MAE-Derived positional embeddings were originally referred to as "*MAED*" embeddings. Any reference to *MAED* or *MAED* position embeddings in code or files can be used synonymously with MAE-Derived position embeddings.

<br></br>


### MAE-Derived Positional Embedding Creation
-----
Two notebooks are used to generate the MAE-Derived positional embedding similarity matrices. The similarity weights can then be reduced in dimensionality to a specified scale of choosing.

[Colab_Training_AE_CIFAR10.ipynb](Colab_Training_AE_CIFAR10.ipynb) - This notebook uses masked auto-encoder to output a similarity matrix for the CIFAR-10 dataset.

[Colab_Training_AE_IMDB.ipynb](Colab_Training_AE_IMDB.ipynb) - This notebook uses a masekd auto-encoder to outputs a similarity matrix for the CIFAR-10 dataset.


<br></br>

### Perceiver Positional Embedding Training & Evaluation
-----
#### IMDB
[Perceiver_IMDB_vanilla_training.ipynb](Perceiver_IMDB_vanilla_training.ipynb) - This notebook includes the training and evaluation tasks including training run output for the IMDB classification task using the default Perceiver specified positional embeddings.

[Perceiver_IMDB_MAED_training.ipynb](Perceiver_IMDB_MAED_training.ipynb) - This notebook includes the training and evaluation tasks including training run output for the IMDB classification task using the MAE-Derived positional embeddings.

#### CIFAR-10
[run_perceiver_w_MAED_encoder.ipynb](run_perceiver_w_MAED_encoder.ipynb) - This notebook includes the training and evaluation tasks including training run output for the IMDB classification task using the default Perceiver specified positional embeddings. Related .py scripts are used to run the training process on servers outside of a Jupyter Notebook environment

[baseline_perceiver_cifar.py](baseline_perceiver_cifar.py) - This includes the training and evaluation tasks including training run output for the CIFAR-10 classification task using the MAE-Derived positional embeddings.

<br></br>

### Visualization Tasks
-----
[visualize_positional_embeddings.ipynb](visualize_positional_embeddings.ipynb) - This notebook is used to visualize the weight similarity matrices outputed from the MAE preprocessing task. 


[visualize_autoencoder.ipynb](visualize_autoencoder.ipynb) - This notebook is used to visualize outputs from the masked autoencoder pretraining task. The included outputs show results acheived through 99% masking for image recreation of the CIFAR-10 dataset.

<br></br>
### Position Embeddings
-----
[pos_embeddings_IMDB_tSNE_2048x64.pth](pos_embeddings_IMDB_tSNE_2048x64.pth) - This file contains the t-SNE dimensionality reduced weights used for the MAE-Derived position embeddings. Weights are loaded through the related IMDB training notebooks.
