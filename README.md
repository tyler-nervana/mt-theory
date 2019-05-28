## Purpose
This repo provides the code used to generate the data found in *Generalization in multitask deep neural classifiers: a statistical physics approach*. It is provided as-is, but we will do our best to answer any questions added to the repo.

## Details
- [`docker/Dockerfile`](docker/Dockerfile): The Dockerfile declaring all dependencies for the repo.
- [`multitask/train.py`](multitask/train.py): The main training script. It has a lot of options that can be viewed with `train.py -h`
- [`multitask/data.py`](multitask/data.py): The data generating function for the teacher networks