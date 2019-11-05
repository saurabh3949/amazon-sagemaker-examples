from recobanditsgym.envs import EMBEDDING_DIM, NUM_ITEMS, NUM_USERS, TOP_K


class RecoBanditAgent:
    def __init__(self, env_config, *args, **kwargs):
        self.env_config = env_config
        self.embedding_dim = env_config.get(EMBEDDING_DIM, None)
        assert self.embedding_dim, "embedding_dim cannot be none. Specify 'dim' in environment config"
        self.num_users = env_config.get(NUM_USERS, 1)
        self.num_items = env_config.get(NUM_ITEMS, 0)
        self.top_k = env_config.get(TOP_K, 1)

    def learn(self, shared_features, action_features, action_index, action_prob,
              reward, user_id=None, candidate_ids=None, cost_fn=None):
        raise NotImplementedError

    def choose_actions(self, shared_features, action_features, user_id=None, candidate_ids=None):
        raise NotImplementedError
