import numpy as np

from recobanditsgym.agents import RecoBanditAgent
import tempfile, json
import importlib
import pandas as pd
import random
import copy

_hyp_template = {
    'model.vocab_size' : 10, # dummy - total number of unique arms if available
    'data.cand_size' : 3,    # number of impression arms per event
    'model.bias_features' : True, # build model from candidate_arms_features
    'model.num_hidden' : 1,       # dummy - user embedding dimensions
    'model.core_prefix' : 'baseline/', # dummy - user embedding by MLP
    'init.svd' : 0,               # for faster initialization
    'init.limit' : 10,            # for faster initialization
    'training.explicit_label' : 1, # use explicit labels for multi-label events
    'loss.prefix' : 'L2Loss',     # use L2Loss for multi-label events
    'training.epochs' : 6,
    'split.train_valid_ratio' : 4, # validation data for early stopping
    'training.lam' : 1.0, # use 0% item embedding and 100% feature embedding
}


def _argmax_with_random_tie_break(rewards, cand_ids=None):
    argmax_natural = np.random.choice(np.flatnonzero(
            np.asarray(rewards) == max(rewards)
        ))
    if cand_ids is None:
        return int(argmax_natural)
    else:
        return int(cand_ids[argmax_natural])


def _list(arr):
    return np.asarray(arr).tolist()


def _normalize_vector(shared_features):
    return np.divide(shared_features, np.linalg.norm(shared_features))


def _normalize_matrix(candidate_arms_features, axis=1):
    norms = np.linalg.norm(candidate_arms_features, axis=axis, keepdims=True)
    return np.divide(candidate_arms_features, norms)


class HRNNAgent(RecoBanditAgent):
    def __init__(self, env_config, update_schedule,
        import_module='aws_concierge_hrnn_model.sagemaker_wrapper',
        hyp=_hyp_template,
        eps=0,
        normalize=True,
        allow_group_learn=True
        ):
        RecoBanditAgent.__init__(self, env_config)
        self.update_schedule = update_schedule
        self.import_module = import_module
        self.hyp = hyp
        self.tmp_data_dir = tempfile.TemporaryDirectory()
        self.tmp_model_dir = tempfile.TemporaryDirectory()
        self.model_transform = None
        self.reservoir = {}
        self.eps = eps
        self.normalize = normalize
        self.cur_group = {'user_id':None, 'group_id':None}
        self.allow_group_learn = allow_group_learn


    def learn(self, shared_features, candidate_arms_features, action_index, action_prob, reward, user_id=None, group_id=None,
              candidate_ids=None, cost_fn=None):

        if group_id != self.cur_group['group_id']:
            if self.cur_group['group_id'] is not None:
                self.cur_group.pop('group_id')
                self.group_learn(**self.cur_group)

            self.cur_group = {'user_id'  : user_id,
                              'group_id' : group_id}

        self.cur_group.setdefault('shared_features', shared_features)
        self.cur_group.setdefault('rewards', [])
        self.cur_group['rewards'].append(reward)

        if candidate_ids is not None:
            self.cur_group.setdefault('candidate_ids', [])
            self.cur_group['candidate_ids'].append(candidate_ids[action_index])
        else:
            self.cur_group.setdefault('candidate_ids', None)

        if candidate_arms_features is not None:
            self.cur_group.setdefault('candidate_arms_features', [])
            self.cur_group['candidate_arms_features'].append(candidate_arms_features[action_index])
        else:
            self.cur_group.setdefault('candidate_arms_features', None)


    def group_learn(self,
        shared_features,    # user_dynamics
        candidate_arms_features, # cand_features
        rewards,            # labels
        candidate_ids=None, # cand_ids
        user_id=None,
        cost_fn=None):

        _i           = importlib.import_module(self.import_module)
        train        = _i.train
        save         = _i.save
        model_fn     = _i.model_fn
        transform_fn = _i.transform_fn


        item_id = _argmax_with_random_tie_break(rewards, candidate_ids)

        user_data = self.reservoir.setdefault(user_id, {})

        # labels for each individual arm may or may not be provided. For this interface, they are always provided for simplicity.
        user_data.setdefault('item_ids', [])
        user_data['item_ids'].append(item_id)
        user_data.setdefault('labels', [])
        user_data['labels'].append(_list(rewards))

        # add personalize-ranking candidate arms and user-item contexts
        if candidate_ids is not None and candidate_arms_features is not None:
            assert len(candidate_ids) == len(candidate_arms_features), "please provide only relevant candidate arms"

        if candidate_ids is not None:
            user_data.setdefault('cand_ids', [])
            user_data['cand_ids'].append(_list(candidate_ids))

        if candidate_arms_features is not None:
            if self.normalize:
                candidate_arms_features = _normalize_matrix(candidate_arms_features)
            user_data.setdefault('cand_features', [])
            user_data['cand_features'].append(_list(candidate_arms_features))

        # add user context features
        if shared_features is not None:
            if self.normalize:
                shared_features = _normalize_vector(shared_features)
            user_data.setdefault('user_dynamics', [])
            user_data['user_dynamics'].append(_list(shared_features))


        if (user_id % self.update_schedule == 0) and user_id>0:

            with open(self.tmp_data_dir.name + f'/reservoir.json', 'w') as f:
                f.write('\n'.join(map(json.dumps, self.reservoir.values())))

            inputs = {'train' : self.tmp_data_dir.name}
            if self.model_transform is not None:
                inputs['pretrained'] = self.tmp_model_dir.name

            ret_args = train(None, inputs, self.hyp.copy())
            save(ret_args, self.tmp_model_dir.name)
            self.net = model_fn(self.tmp_model_dir.name)
            self.model_transform = lambda x: json.loads(transform_fn(self.net, x)[0])


    def choose_actions(self,
        shared_features,    # user_dynamics
        candidate_arms_features, # cand_features
        candidate_ids=None, # cand_ids
        user_id=None        # associate user item_ids
        ):

        candidate_arms_features = candidate_arms_features / np.linalg.norm(candidate_arms_features, axis=1, keepdims=True)

        if self.model_transform is None:
            action_indices = np.random.choice(len(candidate_arms_features), size=self.top_k, replace=False)
            actions_probs = [1.0] * self.top_k
            return action_indices, actions_probs

        else:
            # match by user_id
            user_data = copy.deepcopy(self.reservoir.get(user_id, {}))
            user_data.setdefault('item_ids', [])

            # add personalize-ranking candidate arms at the request time
            if candidate_ids is not None and candidate_arms_features is not None:
                assert len(candidate_ids) == len(candidate_arms_features), "please provide only relevant candidate arms"

            user_data.pop('cand_ids', None)
            if candidate_ids is not None:
                user_data['cand_ids'] = _list(candidate_ids)

            user_data.pop('cand_features', None)
            if candidate_arms_features is not None:
                if self.normalize:
                    candidate_arms_features = _normalize_matrix(candidate_arms_features)
                user_data['cand_features'] = _list(candidate_arms_features)

            # add user dynamic contexts
            if shared_features is not None:
                if self.normalize:
                    shared_features = _normalize_vector(shared_features)
                user_data['user_dynamics'].append(_list(shared_features))

            # do inference
            model_output = self.model_transform({
                'configuration':{
                    'topk':self.top_k, 'log_level':30, 'lam':self.hyp.get('training.lam', 0.5)},
                'instances':[{'data':user_data}],
            })
            action_indices = model_output['outputs'][0]['item_ids']
            actions_probs = model_output['outputs'][0]['scores']

            # eps greedy
            eps_actions = set(range(len(candidate_arms_features))) - set(action_indices)

            if len(eps_actions):
                eps_actions = list(np.random.permutation(list(eps_actions)))

                for k in range(self.top_k):
                    if np.random.rand() < self.eps:
                        actions_probs[k] = self.eps / len(eps_actions)
                        action_indices[k] = eps_actions.pop()

            return action_indices, actions_probs
