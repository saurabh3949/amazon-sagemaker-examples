import os
import random
from typing import List

import numpy as np

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


class MSLREnv():
    def __init__(self, mslr_npz_path, item_pool_size=10, ordering=0):
        # Use https://github.com/akshaykr/oracle_cb/blob/master/PreloadMSLR.py to create MSLR npz file
        mslr = np.load(mslr_npz_path)
        self.relevances = mslr["relevances"]
        self.features = mslr["features"]
        self.docs_per_query = mslr["docsPerQuery"]
        self.item_pool_size = item_pool_size
        self.total_regret = 0
        self.query_index = 0
        mslr_dir = os.path.dirname(mslr_npz_path)
        self.orderings = [np.load(os.path.join(mslr_dir, f'mslr30k_train_{i}.npz'))["order"] for i in range(20)]
        self.ordering_index = ordering
        self.current_ordering = self.orderings[ordering]
        self.current_candidate_features = None
        self.skipped_queries = 0
        self.done = False

    def reset(self):
        self.done = False
        self.query_index = 0
        self.current_ordering = self.orderings[self.ordering_index]
        self._regulate_document_pool()
        return self.current_candidate_features

    def get_feedback(self, actions):
        last_query_id = self.current_ordering[self.query_index]
        relevances = np.array([self.relevances[last_query_id][i] for i in actions])

        rewards = relevances / 4
        regret = np.max(self.relevances[last_query_id][:self.item_pool_size]) - relevances.max()
        return rewards, regret

    def get_oracle_rewards(self):
        last_query_id = self.current_ordering[self.query_index]
        relevances = self.relevances[last_query_id][:self.item_pool_size]
        return np.sort(relevances)[-3:].sum()

    def step(self, actions):
        rewards, regret = self.get_feedback(actions)
        self.total_regret += regret
        info = {"total_regret": self.total_regret}

        # Last query reached
        if self.query_index == len(self.current_ordering) - 1:
            return None, rewards, True, info
        else:
            self.query_index += 1
            self._regulate_document_pool()
            return self.current_candidate_features, rewards, self.done, info

    def _regulate_document_pool(self):
        if self.query_index == len(self.current_ordering) - 1:
            self.current_candidate_features = None
            self.done = True
            return
        new_query_id = self.current_ordering[self.query_index]
        num_docs = self.docs_per_query[new_query_id]
        num_docs = min(num_docs, self.item_pool_size)
        if self.relevances[new_query_id][:num_docs].sum() < 0:
            print(f"No relevant docs found for query ID: {new_query_id}. Skipping this query.")
            self.query_index += 1
            self.skipped_queries += 1
            self._regulate_document_pool()
        else:
            self.current_candidate_features = self.features[new_query_id][:num_docs]


class YahooEnv():
    def __init__(self, yahoo_npz_path, item_pool_size=6, ordering=0):
        yahoo = np.load(yahoo_npz_path)
        self.relevances = yahoo["relevances"]
        self.features = yahoo["features"]
        self.docs_per_query = yahoo["docsPerQuery"]
        self.item_pool_size = item_pool_size
        self.total_regret = 0
        self.query_index = 0
        yahoo_dir = os.path.dirname(yahoo_npz_path)
        self.orderings = [np.load(os.path.join(yahoo_dir, f'yahoo_train_{i}.npz'))["order"] for i in range(20)]
        self.ordering_index = ordering
        self.current_ordering = self.orderings[ordering]
        self.current_candidate_features = None
        self.skipped_queries = 0
        self.done = False

    def reset(self):
        self.done = False
        self.query_index = 0
        self.current_ordering = self.orderings[self.ordering_index]
        self._regulate_document_pool()
        return self.current_candidate_features

    def get_feedback(self, actions):
        last_query_id = self.current_ordering[self.query_index]
        relevances = np.array([self.relevances[last_query_id][i] for i in actions])

        rewards = relevances / 4
        regret = np.max(self.relevances[last_query_id][:self.item_pool_size]) - relevances.max()
        return rewards, regret

    def step(self, actions):
        rewards, regret = self.get_feedback(actions)
        self.total_regret += regret
        info = {"total_regret": self.total_regret}

        # Last query reached
        if self.query_index == len(self.current_ordering) - 1:
            return None, rewards, True, info
        else:
            self.query_index += 1
            self._regulate_document_pool()
            return self.current_candidate_features, rewards, self.done, info

    def _regulate_document_pool(self):
        if self.query_index == len(self.current_ordering) - 1:
            self.current_candidate_features = None
            self.done = True
            return
        new_query_id = self.current_ordering[self.query_index]
        num_docs = self.docs_per_query[new_query_id]
        num_docs = min(num_docs, self.item_pool_size)
        if self.relevances[new_query_id][:num_docs].sum() < 0 or num_docs < 2:
            # print(f"No relevant docs found for query ID: {new_query_id}. Skipping this query.")
            self.query_index += 1
            self.skipped_queries += 1
            self._regulate_document_pool()
        else:
            self.current_candidate_features = self.features[new_query_id][:num_docs]