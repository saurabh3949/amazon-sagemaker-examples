from typing import List, Tuple, Iterable

import numpy as np

from recobanditsgym.agents import RecoBanditAgent


class LinUCBUserStruct:
    def __init__(self, embedding_dim, lambda_, init_fn="zero", update_schedule=1):
        self.d = embedding_dim
        self.A = lambda_ * np.identity(n=self.d)
        self.b = np.zeros(self.d)
        self.A_inv = np.linalg.inv(self.A)
        if init_fn == "random":
            self.user_theta = np.random.rand(self.d)
        else:
            self.user_theta = np.zeros(self.d)
        self.time = 0
        self.update_schedule = update_schedule
        self.A_delta = 0
        self.b_delta = 0

    def update_weights(self, item_vector, reward):
        self.time += 1
        self.A_delta += np.outer(item_vector, item_vector)
        self.b_delta += item_vector * reward
        if self.time % self.update_schedule == 0:
            self.A += self.A_delta
            self.b += self.b_delta
            self.A_delta = 0
            self.b_delta = 0
            self.A_inv = np.linalg.inv(self.A)
            self.user_theta = np.dot(self.A_inv, self.b)

    def get_theta(self):
        return self.user_theta

    def get_A(self):
        return self.A

    def get_prob(self, alpha, item_vector):
        if alpha == -1:
            alpha = 0.1 * np.sqrt(np.log(self.time + 1))
        mean = np.dot(self.user_theta, item_vector)
        var = np.sqrt(np.dot(np.dot(item_vector, self.A_inv), item_vector))
        pta = mean + alpha * var
        return pta

    def get_prob_plot(self, alpha, item_vector):
        mean = np.dot(self.user_theta, item_vector)
        var = np.sqrt(np.dot(np.dot(item_vector, self.A_inv), item_vector))
        pta = mean + alpha * var
        return pta, mean, alpha * var


class LinUCBAgent(RecoBanditAgent):
    def __init__(self, env_config, init_fn="zero", alpha=0.3, lambda_=0.1,
                 update_schedule=1):
        RecoBanditAgent.__init__(self, env_config)
        self.users = []
        self.alpha = alpha
        self.lambda_ = lambda_
        self.update_schedule = update_schedule

        # Ensure that at least one model is created
        self.num_users = max(1, self.num_users)

        # Create a separate model for each user
        for i in range(self.num_users):
            self.users.append(LinUCBUserStruct(self.embedding_dim, self.lambda_, init_fn,
                                               self.update_schedule))

    def learn(self, shared_features, candidate_arms_features, reward, action_index,
              action_prob=None, user_id=0, candidate_ids=None, cost_fn=None):
        self._verify(user_id)
        self.users[user_id].update_weights(candidate_arms_features[action_index], reward)

    def _verify(self, user_id):
        assert user_id is not None, "LinUCBAgent requires user_id as it learns a separate model for each user."
        assert user_id < len(self.users), "LinUCBAgent doesn't support adding new users dynamically."

    def choose_actions(self, shared_features, candidate_arms_features,
                       user_id=0, candidate_ids=None) -> Tuple[Iterable[int], Iterable[float]]:
        self._verify(user_id)
        user_theta = self.users[user_id].get_theta()
        mean_vector = np.dot(candidate_arms_features, user_theta)

        # VARIANCE
        var_matrix = np.sqrt(np.dot(np.dot(candidate_arms_features,
                                           self.users[user_id].A_inv), candidate_arms_features.T).clip(0))
        item_scores = mean_vector + self.alpha * np.diag(var_matrix)

        top_k_action_indices = list(reversed(item_scores.argsort()[-1 * self.top_k:]))
        action_probs = [1.0] * self.top_k
        return top_k_action_indices, action_probs

    def get_prob(self, candidate_arms_features, user_id=0):
        means = []
        vars = []
        for x in candidate_arms_features:
            x_pta, mean, var = self.users[user_id].get_prob_plot(self.alpha, x)
            means.append(mean)
            vars.append(var)
        return means, vars

    def get_theta(self, user_id=0):
        self._verify(user_id)
        return self.users[user_id].get_theta()
