import os
import random
from typing import List

import numpy as np

# for movielens env
import torch
from spotlight.cross_validation import random_train_test_split
from spotlight.datasets.movielens import get_movielens_dataset
from spotlight.interactions import Interactions
from spotlight.factorization.implicit import ImplicitFactorizationModel
from spotlight.factorization.explicit import ExplicitFactorizationModel

from recobanditsgym.embeddings import EmbeddingManager


EMBEDDING_DIM = "Dim"
NUM_USERS = "Users"
NUM_ITEMS = "Items"
TOP_K = "TopK"
ITEM_POOL_SIZE = "ItemPool"
USER_GROUPS = "UserGroups"
ITEM_GROUPS = "ItemGroups"

DEFAULT_CONFIG = {NUM_USERS: 100,
                  USER_GROUPS: 10,
                  NUM_ITEMS: 1000,
                  ITEM_GROUPS: 50,
                  EMBEDDING_DIM: 32,
                  ITEM_POOL_SIZE: 100,
                  TOP_K: 1}


class SimpleRecoEnv():
    def __init__(self, config=None):
        if config is None:
            config = DEFAULT_CONFIG
        self.embedding_dim = config.get("Dim", 32)
        self.user_config = {"number": config.get(NUM_USERS, 100),
                            "groups": config.get(USER_GROUPS, 10)}
        self.item_config = {"number": config.get(NUM_ITEMS, 1000),
                            "groups": config.get(ITEM_GROUPS, 50)}

        self.user_manager = EmbeddingManager(dim=self.embedding_dim, config=self.user_config)
        self.item_manager = EmbeddingManager(dim=self.embedding_dim, config=self.item_config)
        self.top_k = config.get(TOP_K, 1)
        self.item_pool_size = config.get(ITEM_POOL_SIZE, 100)

        self.current_user = 0
        self.item_pool = []
        self.total_regret = 0

    def _regulate_item_pool(self):
        # Randomly generate a candidate list of items
        self.item_pool = random.sample(self.item_manager.embeddings, self.item_pool_size)

    @property
    def current_user_id(self) -> int:
        return self.user_manager.embeddings[self.current_user].id

    def step(self, actions: List[int]):
        # TODO: Generalize for slate recommendation
        action = actions[0]
        recommended_item = self.item_pool[action]
        scores = self.user_manager.embeddings[self.current_user].get_preference_scores(self.item_pool)
        reward = scores[action]
        regret = np.max(scores) - reward
        self.total_regret += regret

        done = False
        info = {"total_regret": self.total_regret}
        if self.current_user == self.user_config["number"] - 1:
            self.current_user = 0
        else:
            self.current_user += 1
        self._regulate_item_pool()
        return (self.user_manager.get_embedding(index=self.current_user),
                *[item.embedding for item in self.item_pool]), reward, done, info

    def reset(self):
        self.current_user = 0
        self.total_regret = 0
        random.shuffle(self.user_manager.embeddings)
        self._regulate_item_pool()
        return (self.user_manager.get_embedding(index=self.current_user),
                *[item.embedding for item in self.item_pool])



class MovieLensEnv():
    def __init__(self, variant='100K', factorization='explicit',
                 rating_threshold=3.0, item_pool_size=50, top_k=5, embedding_dim=16):

        dataset = get_movielens_dataset(variant=variant)
        self.top_k = top_k

        if item_pool_size is not None:
            self.item_pool_size = item_pool_size
        else:
            # Use all items as the candidate list
            self.item_pool_size = dataset.num_items - 1

        self.total_regret = 0

        train, test = None, None

        # Note: Since we want to model item attractiveness, we treat
        # rating > rating_threshold as positive and others as negatives.
        if factorization == 'implicit':
            indices_to_keep = dataset.ratings > rating_threshold
            user_ids = dataset.user_ids[indices_to_keep]
            item_ids = dataset.item_ids[indices_to_keep]
            dataset = Interactions(user_ids=user_ids,
                                   item_ids=item_ids,
                                   num_users=dataset.num_users,
                                   num_items=dataset.num_items)
            train, test = random_train_test_split(dataset, test_percentage=0.5)

            self.model_full = ImplicitFactorizationModel(n_iter=3, loss='pointwise',
                                                         embedding_dim=embedding_dim)
            self.model_train = ImplicitFactorizationModel(n_iter=3, loss='pointwise',
                                                          embedding_dim=embedding_dim)

        elif factorization == 'explicit':
            dataset.ratings = np.where(dataset.ratings > rating_threshold, 1.0, 0.0).astype("float32")
            train, test = random_train_test_split(dataset, test_percentage=0.5)

            self.model_full = ExplicitFactorizationModel(n_iter=3, loss='logistic',
                                                         embedding_dim=embedding_dim)
            self.model_train = ExplicitFactorizationModel(n_iter=3, loss='logistic',
                                                          embedding_dim=embedding_dim)

        print('Training model with full data')
        self.model_full.fit(dataset)

        print('Training model with training data')
        self.model_train.fit(train)

        self.attractiveness_means = np.zeros((dataset.num_users, dataset.num_items))

        for user_id in range(0, dataset.num_users):
            if factorization == 'explicit':
                self.attractiveness_means[user_id, :] = self.model_full.predict(user_ids=user_id)
            else:
                predictions = torch.sigmoid(torch.from_numpy(self.model_full.predict(user_ids=user_id)))
                self.attractiveness_means[user_id, :] = predictions.numpy()


        self.user_embeddings = self.model_train._net.user_embeddings.weight.data.numpy()
        self.item_embeddings = self.model_train._net.item_embeddings.weight.data.numpy()

        self.total_users = test.num_users - 1  # Spotlight creates a dummy user with id=0
        self.total_items = test.num_items - 1  # Spotlight creates a dummy item with id=0
        self.done = False
        self.current_user_id = None
        self.current_user_embedding = None
        self.current_item_pool = None
        self.current_items_embedding = None
        self.step_count = 1

    def reset(self):
        self.done = False
        self.current_user_id = None
        self.current_user_embedding = None
        self.current_item_pool = None
        self.current_items_embedding = None
        self.step_count = 1
        self.total_regret = 0
        self.total_regret_random = 0
        self._regulate_item_pool()
        return self.current_user_embedding, self.current_items_embedding

    def _regulate_item_pool(self):
        if self.step_count > self.total_users:
            self.step_count = 1
        # TODO: Randomize user selection
        self.current_user_id = self.step_count
        self.current_user_embedding = self.user_embeddings[self.current_user_id]
        # Randomly generate a candidate list of items
        self.current_item_pool = np.array(random.sample(range(1, self.total_items+1), self.item_pool_size))
        self.current_items_embedding = self.item_embeddings[self.current_item_pool]

    def step(self, actions):
        assert len(actions) == self.top_k, "Size of recommended items list does not match top-k"
        rewards, regret, regret_random = self.get_feedback(actions)
        self.total_regret += regret
        self.total_regret_random += regret_random
        info = {"total_regret": self.total_regret, "random_regret": self.total_regret_random}

        # Last user reached
        self.step_count += 1
        self._regulate_item_pool()
        return (self.current_user_embedding, self.current_items_embedding), rewards, self.done, info

    def get_feedback(self, actions, click_model="cascade"):
        """
        Return rewards: List[float] and regret for the current recommended list - actions

        Args:
            actions: A list of top-k actions indices picked by the agent from candidate list
            click_model: One of 'cascade', 'pbm'

        Returns:
            rewards: A reward corresponding to each item in the list
            regret: Expected regret calculated based on the recommended actions
        """

        recommended_item_ids = self.current_item_pool[actions]
        attraction_probs = self.attractiveness_means[self.step_count][recommended_item_ids]

        random_indices = np.random.choice(len(recommended_item_ids), size=self.top_k, replace=False)
        random_item_ids = self.current_item_pool[random_indices]
        random_attraction_probs = self.attractiveness_means[self.step_count][random_item_ids]
        # Simulate user behavior using a cascading click model.
        # User scans the list top-down and clicks on an item with prob = attractiveness_means.
        # User stops seeing the list after the first click.
        clicks = np.random.binomial(1, attraction_probs)
        if clicks.sum() > 1:
            first_click = np.flatnonzero(clicks)[0]
            clicks = clicks[:first_click + 1]

        expected_reward_random = 1 - np.prod(1 - random_attraction_probs)
        expected_reward = 1 - np.prod(1 - attraction_probs)


        current_pool_probs = self.attractiveness_means[self.step_count][self.current_item_pool]
        optimal_attraction_probs = np.sort(current_pool_probs)[::-1][:self.top_k]
        expected_optimal_reward = 1 - np.prod(1 - optimal_attraction_probs)
        regret = expected_optimal_reward - expected_reward
        regret_random = expected_optimal_reward - expected_reward_random

        return clicks, regret, regret_random