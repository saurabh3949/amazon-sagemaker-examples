import numpy as np

from recobanditsgym.agents import RecoBanditAgent


class RandomAgent(RecoBanditAgent):
    def __init__(self, env_config):
        RecoBanditAgent.__init__(self, env_config)

    def learn(self, shared_features, candidate_arms_features, action_index, action_prob, reward, user_id=None,
              candidate_ids=None, cost_fn=None):
        pass

    def choose_actions(self, shared_features, candidate_arms_features, user_id=None, candidate_ids=None):
        action_indices = np.random.choice(len(candidate_arms_features), size=self.top_k, replace=False)
        actions_probs = [1.0] * self.top_k
        return action_indices, actions_probs
