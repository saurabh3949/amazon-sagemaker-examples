import numpy as np

from recobanditsgym.agents import RecoBanditAgent
import tempfile, json
import sagemaker.mxnet

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


class HRNNContainerAgent(RecoBanditAgent):
    def __init__(self, env_config, update_schedule, hyp=_hyp_template, eps=0,
        training_image_name = None,
        inference_image_name = None,
        hrnn_source_dir='aws_concierge_hrnn_model',
        ):
        RecoBanditAgent.__init__(self, env_config)
        self.update_schedule = update_schedule
        self.hyp = hyp
        self.tmp_data_dir = tempfile.TemporaryDirectory()
        self.tmp_model_dir = tempfile.TemporaryDirectory()
        self.model_transform = None
        self.reservoir = []
        self.eps = eps
        self.training_image_name = training_image_name
        self.inference_image_name = inference_image_name
        self.hrnn_source_dir = hrnn_source_dir # path to hrnn source code
        self.latest_models = {} # keep track of the latest model and predictor


    def learn(self, shared_features, candidate_arms_features, action_index, action_prob, reward, user_id=None,
              candidate_ids=None, cost_fn=None):
        pass

    def group_learn(self, shared_features, candidate_arms_features, actions, probs, rewards, user_id=None,
              candidate_ids=None, cost_fn=None):
        from aws_concierge_hrnn_model.sagemaker_wrapper import train, save, model_fn, transform_fn

        candidate_arms_features = candidate_arms_features / np.linalg.norm(candidate_arms_features, axis=1, keepdims=True)

        seq_data = {
            'labels' : [rewards.tolist()],
            'cand_ids' : [[-1] * len(rewards)],
            'cand_features' : [candidate_arms_features[actions].tolist()],
            'item_ids' : [-1],
        }

        self.reservoir.append(seq_data)

        if (user_id % self.update_schedule == 0) and user_id>0:

            with open(self.tmp_data_dir.name + f'/reservoir.json', 'w') as f:
                f.write('\n'.join(map(json.dumps, self.reservoir)))

            SOURCE_DIR = self.hrnn_source_dir

            # create data and/or pretrained model
            inputs = {'train' : 'file://' + self.tmp_data_dir.name}
            if 'm' in self.latest_models:
                inputs['pretrained'] = self.latest_models['m'].model_data

            # use custom image to pre-install the training dependencies
            self.latest_models['m'] = sagemaker.mxnet.MXNet(
                SOURCE_DIR + "/sagemaker_wrapper.py",
                role=sagemaker.get_execution_role(), # role
                image_name = self.training_image_name, # training container
                train_instance_count=1,
                train_instance_type='local', # local model
                framework_version="1.4.1",   # python version
                py_version="py3",
                hyperparameters=self.hyp.copy())

            self.latest_models['m'].fit(inputs)

            # deploy or update model
            if 'predictor' in self.latest_models:
                self.latest_models['predictor'].delete_endpoint()

            # use custom image to pre-install the inference dependencies
            inf_model = sagemaker.mxnet.model.MXNetModel(
                model_data = self.latest_models['m'].model_data,
                image = self.inference_image_name, # docker images
                role=sagemaker.get_execution_role(), 
                py_version='py3',            # python version
                framework_version='1.4.1',   # mxnet version
                entry_point=SOURCE_DIR + '/sagemaker_wrapper.py',
                source_dir=SOURCE_DIR)

            self.latest_models['predictor'] = inf_model.deploy(1, 'local')
            self.model_transform = self.latest_models['predictor'].predict


    def choose_actions(self, shared_features, candidate_arms_features, user_id=None, candidate_ids=None):

        candidate_arms_features = candidate_arms_features / np.linalg.norm(candidate_arms_features, axis=1, keepdims=True)

        if self.model_transform is None:
            action_indices = np.random.choice(len(candidate_arms_features), size=self.top_k, replace=False)
            actions_probs = [1.0] * self.top_k
            return action_indices, actions_probs

        else:
            request_data = {
                'cand_ids' : list(range(len(candidate_arms_features))),
                'cand_features' : candidate_arms_features.tolist(),
                'item_ids' : [],
            }

            model_output = self.model_transform({
                'configuration':{
                    'topk':self.top_k, 'log_level':30, 'lam':self.hyp.get('training.lam', 0.5)},
                'instances':[{'data':request_data}],
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
