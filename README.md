# Semi-supervised learning of convolutional neural networks by solving jigsaw puzzles

## Main structure

The file `run_experiment.py` was used to run the different experiments. 
It internally calls a function defined in `jigsaw_pretrain_stl10.py` to pretrain on the jigsaw task.
Then this network is finetuned on a classification task using functionality from `supervised_train_pretrained_stl10.py`.

## Components 

Below we will further describe the individual components that are used in this repository.

### `permutation.py`
Class that interfaces the permutations used during training.

### `pipeline.py`
Collection of functions that aid the training process.

### `preprocessing.py`
Custom preprocessing steps for images and image tiles.

### `tests`
Tests for the individual components.

### `logger`
Logger class that was used used to track the results of different experiments.
 
### `jigsaw_model`
Dataset and CNN models for the context-free network.

### `helpers.py`
Helper functions for debugging and visualizing the model.

### `constants.py`
Defines constants used at different locations during training.


## Data
Due to its size, the image data is not included in this repository.
The pre-calculated permutation file can be found under `data/permutations_d_1000.csv`.