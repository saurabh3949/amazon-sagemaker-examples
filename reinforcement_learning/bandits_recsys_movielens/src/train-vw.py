import argparse
import json
import os
from pathlib import Path
import logging

from vw_agent import VWAgent

from io_utils import extract_model, CSVReader, validate_experience
from vw_utils import TRAIN_CHANNEL, MODEL_CHANNEL, MODEL_OUTPUT_PATH, MODEL_OUTPUT_DIR

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def main():
    """ Train a Vowpal Wabbit (VW) model through C++ process. """
    
    channel_names = json.loads(os.environ['SM_CHANNELS'])
    hyperparameters = json.loads(os.environ['SM_HPS'])
    num_arms = int(hyperparameters.get("num_arms", 0))
    num_policies = int(hyperparameters.get("num_policies", 3))
    exploration_policy = hyperparameters.get("exploration_policy", "egreedy").lower()
    epsilon = float(hyperparameters.get("epsilon", 0))
    arm_features = bool(hyperparameters.get("arm_features", True))

    if num_arms is 0:
        raise ValueError("Customer Error: Please provide a non-zero value for 'num_arms'")
    logging.info("channels %s" % channel_names)
    logging.info("hps: %s" % hyperparameters)

    # Different exploration policies in VW
    # https://github.com/VowpalWabbit/vowpal_wabbit/wiki/Contextual-Bandit-algorithms
    valid_policies = ["egreedy", "bag", "regcbopt"]
    if exploration_policy not in valid_policies:
        raise ValueError(f"Customer Error: exploration_policy must be one of {valid_policies}.")
    
    if exploration_policy == "egreedy":
        vw_args_base = f"--cb_explore_adf --cb_type mtr --epsilon {epsilon}"
    elif exploration_policy == "regcbopt":
        vw_args_base = f"--cb_explore_adf --cb_type mtr --regcbopt --mellowness 0.01"
    else:
        vw_args_base = f"--cb_explore_adf --cb_type mtr --{exploration_policy} {num_policies}"

    # No training data. Initialize and save a random model
    if TRAIN_CHANNEL not in channel_names:
        logging.info("No training data found. Saving a randomly initialized model!")
        vw_model = VWAgent(cli_args=vw_args_base,
                           model_path=None,
                           test_only=False,
                           quiet_mode=False,
                           output_dir=MODEL_OUTPUT_DIR,
                           adf_mode=arm_features,
                           num_actions=num_arms
                           )
        vw_model.start()
        vw_model.save_model(close=True)
    
    # If training data is present
    else:
        if MODEL_CHANNEL not in channel_names:
            logging.info(f"No pre-trained model has been specified in channel {MODEL_CHANNEL}."
                         f"Training will start from scratch.")
            vw_model = VWAgent(cli_args=vw_args_base,
                               output_dir=MODEL_OUTPUT_DIR,
                               model_path=None,
                               test_only=False,
                               quiet_mode=False,
                               adf_mode=arm_features,
                               num_actions=num_arms)
        else:
            # Load the pre-trained model for training.
            model_folder = os.environ[f'SM_CHANNEL_{MODEL_CHANNEL.upper()}']
            metadata_path, weights_path = extract_model(model_folder)
            logging.info(f"Loading model from {weights_path}")
            vw_model = VWAgent.load_model(metadata_loc=metadata_path,
                                          weights_loc=weights_path,
                                          test_only=False,
                                          quiet_mode=False,
                                          output_dir=MODEL_OUTPUT_DIR)                 
        # Init a class instance that communicates with C++ VW process using pipes

        vw_model.start()

        # Load training data
        training_data_dir = Path(os.environ["SM_CHANNEL_%s" % TRAIN_CHANNEL.upper()])
        training_files = [i for i in training_data_dir.rglob("*") if i.is_file() and i.suffix == ".csv"]
        logging.info("Processing training data: %s" % training_files)

        data_reader = CSVReader(input_files=training_files)
        data_iterator = data_reader.get_iterator()

        count = 0
        for experience in data_iterator:
            is_valid = validate_experience(experience)
            if not is_valid:
                continue
            vw_model.learn(user_embedding=json.loads(experience.get("shared_context", "null")),
                           candidate_embeddings=json.loads(experience.get("actions_context", "null")),
                           action_index=int(experience["action"]),
                           reward=experience["reward"],
                           action_prob=experience["action_prob"],
                           user_id=experience["user_id"])
            count += 1
        
        stdout = vw_model.save_model(close=True)
        print(stdout.decode())
        logging.info(f"Model learned using {count} training experiences.")


if __name__ == '__main__':
    main()
