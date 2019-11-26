# Contextual Bandits with Amazon SageMaker RL

This Notebook demonstrates how you can manage your own contextual multi-armed bandit workflow on SageMaker using the built-in Vowpal Wabbit (VW) container to train and deploy contextual bandit models. We show how to train these models that interact with a live environment (using a simulated client application based on MovieLens dataset) and continuously update the model with efficient exploration.


## Contents

- `bandits_movielens.ipynb`: Notebook used for running the contextual bandit notebook.<br>
- `config.yaml`: The configuration parameters used for the bandit example.<br>
- `common`: Code that manages the different AWS components required for training workflow.<br>
- `src`:
    - `train-vw.py`: Script for training with Vowpal Wabbit library.
    - `env.py`: Script containing the MovieLens environment.
