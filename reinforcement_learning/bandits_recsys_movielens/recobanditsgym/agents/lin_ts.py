import numpy as np
from scipy.stats import invgamma

from recobanditsgym.agents import RecoBanditAgent


class LinTSAgent(RecoBanditAgent):
    """Contextual Linear Thompson Sampling
    http://proceedings.mlr.press/v28/agrawal13.pdf
    """

    def __init__(self, env_config, alpha=1, use_sherman_updates=False,
                 update_schedule=1):
        RecoBanditAgent.__init__(self, env_config)
        self.alpha = alpha
        self.b = np.eye(self.embedding_dim)
        self.b_inv = np.eye(self.embedding_dim)
        self.f = np.zeros((self.embedding_dim, 1))
        self.t = 0
        self.update_schedule = update_schedule
        self.use_sherman_updates = use_sherman_updates
        self.delta_f = 0
        self.delta_b = 0

    def _sherman_morrison_update(self, x):
        # x should have shape (n, 1)
        self.b_inv -= np.linalg.multi_dot([self.b_inv, x, x.T, self.b_inv]) / (1.0 + np.linalg.multi_dot([x.T,
                                                                                                          self.b_inv,
                                                                                                          x]))

    def learn(self, shared_features, candidate_arms_features, action_index, reward, action_prob=None, user_id=None,
              candidate_ids=None, cost_fn=None):
        x = candidate_arms_features[action_index]
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        x = x.reshape((-1, 1))
        if self.use_sherman_updates:
            self._sherman_morrison_update(x)
            self.f += reward * x
        else:
            self.t += 1
            self.delta_f += reward * x
            self.delta_b += np.dot(x, x.T)
            # Can follow an update schedule if not doing sherman morison updates
            if self.t % self.update_schedule == 0:
                self.b += self.delta_b
                self.f += self.delta_f
                self.delta_b = 0
                self.delta_f = 0
                self.b_inv = np.linalg.inv(self.b)

    def choose_actions(self, shared_features, candidate_arms_features, user_id=None, candidate_ids=None):
        mu = (self.b_inv.dot(self.f)).reshape(-1)
        mu = np.random.multivariate_normal(mu, self.alpha * self.b_inv)
        item_scores = candidate_arms_features.dot(mu).reshape(-1)
        top_k_action_indices = list(reversed(item_scores.argsort()[-1 * self.top_k:]))
        action_probs = [1.0] * self.top_k
        return top_k_action_indices, action_probs

