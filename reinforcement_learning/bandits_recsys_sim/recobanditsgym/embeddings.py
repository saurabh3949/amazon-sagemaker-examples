import json
from typing import List, Dict, Tuple
import numpy as np
from sklearn.preprocessing import normalize

from recobanditsgym.utils import uniform_feature_vec


class Embedding:
    def __init__(self, id: int, embedding=None, dim: int = None) -> None:
        self.id = id
        self.embedding = embedding
        if self.embedding is not None:
            if not type(embedding) == np.ndarray:
                self.embedding = np.array(embedding)
            self.embedding = self.embedding.reshape((1, -1))
            self.dim = self.embedding.size
        else:
            assert dim, "Please provide feature dimension: dim when creating an Embedding"
            self.dim = dim

    def choose_closest_embedding(self, candidates_list):
        if self.embedding is None:
            self.init_embeddings()
        best_item = None
        best_item_index = 0
        best_reward = float('-inf')
        for i, candidate in enumerate(candidates_list):
            reward = np.dot(self.embedding, candidate.embedding.T).item()
            if best_reward < reward:
                best_item_index = i
                best_reward = reward
                best_item = candidate
        return best_item_index, best_reward, best_item

    def get_preference_scores(self, candidates_list):
        if self.embedding is None:
            self.init_embeddings()
        matrix = np.array([i.embedding.squeeze() for i in candidates_list])
        return matrix.dot(self.embedding.T).squeeze()

    def init_embeddings(self, init_fn="random"):
        if init_fn == "random":
            self.embedding = np.random.rand(1, self.dim)
            self.embedding = normalize(self.embedding, norm='l2', axis=1)
        else:
            self.embedding = np.zeros((1, self.dim))


class EmbeddingManager:
    def __init__(self, dim, config, argv=None) -> None:
        self.dim = dim
        self.init_fn = eval(config['init_fn']) if 'init_fn' in config else uniform_feature_vec
        self.num_embeddings = config['number'] if 'number' in config else 100
        self.num_groups = config['groups'] if 'groups' in config else 5
        self.argv = argv
        self.signature = "A-" + "+PA" + "+TF-" + self.init_fn.__name__
        if 'load' in config and config['load']:
            # Load from user file
            self.embeddings = self.load_embeddings(config['filename']) if 'filename' in config else \
                self.load_embeddings(
                    config['default_file'])
        else:
            # Simulate random users
            self.embeddings = self.generate_embeddings()
            if 'save' in config and config['save']:
                self.save_embeddings(self.embeddings, config['default_file'], force=False)

    def get_embeddings(self) -> List[Embedding]:
        return self.embeddings

    def get_embedding(self, index: int):
        return self.embeddings[index].embedding

    def save_embeddings(self, users, filename, force=False):
        # TODO: Implement this
        raise NotImplementedError

    @staticmethod
    def load_embeddings(filename) -> List[Embedding]:
        users = []
        with open(filename, 'r') as f:
            for line in f:
                user_id, theta = json.loads(line)
                users.append(Embedding(user_id, np.array(theta)))
        return users

    def generate_masks(self):
        mask = {}
        for i in range(self.num_groups):
            mask[i] = np.random.randint(2, size=(1, self.dim))
        return mask

    def generate_embeddings(self) -> List[Embedding]:
        group_to_user_dict = {}
        embeddings = []
        mask = self.generate_masks()

        if self.num_groups == 0:
            for key in range(self.num_embeddings):
                embedding = self.init_fn(self.dim)
                embedding = normalize(embedding, norm='l2', axis=1)
                embeddings.append(Embedding(key, embedding))
        else:
            for i in range(self.num_groups):
                group_to_user_dict[i] = range(self.num_embeddings * i // self.num_groups,
                                              (self.num_embeddings * (i + 1)) // self.num_groups)

                for key in group_to_user_dict[i]:
                    embedding = np.multiply(self.init_fn(self.dim), mask[i])
                    embedding = normalize(embedding, norm='l2', axis=1)
                    embeddings.append(Embedding(key, embedding))
        return embeddings
