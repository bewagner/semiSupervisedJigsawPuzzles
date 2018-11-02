# Semi-supervised learning of convolutional neural networks by solving jigsaw puzzles

## Main structure

The file `run_experiment.py` was used to run the different experiments. 
It internally calls a function defined in `jigsaw_pretrain_stl10.py` to pretrain on the jigsaw task.
Then this network is finetuned on a classification task using functionality from `supervised_train_pretrained_stl10.py`.

## Components 

### `constants.py`

### `helpers.py`

### `permutation.py`

### `pipeline.py`

### `preprocessing.py`

### `tests`

### `logger`

### `jigsaw_model`

## Data

Due to its size, the image data is not included in this repository.
The pre-calculated permutation file can be found under `data/permutations_d_1000.csv`.