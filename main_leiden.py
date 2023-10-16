# pylint: disable=protected-access
# pylint: disable=ungrouped-imports

"""Evaluates the performance of pre-trained agents."""
import yaml
import torch
import datetime
import gym
import os
from loguru import logger
from argparse import ArgumentParser

from agents.sac.agent import SoftActorCritic, load_sac_agent
from agents.sac.replay_buffer import SoftActorCriticReplayBuffer
from agents.workspaces import SACWorkspace
from agents.utils import set_seed_everywhere

from cubes.constants import BASE_DIR
from cubes.package.core import register_environment
from cubes.construct.buildingconfig import load_building_config
from cubes.construct.building import Building
from cubes.construct.core import materials_evaluator, windows_evaluator
from cubes.package.utilities import get_envconfig_leiden
from cubes.cubesgym.utils.wrappers import LoggerWrapperCubes

parser = ArgumentParser()
parser.add_argument("--case", type=int)
parser.add_argument("--year", type=int)
parser.add_argument("--rep", type=int)
parser.add_argument("--temp_weight", type=int)
parser.add_argument("--load_agent", type=str, default="False")
parser.add_argument("--wandb_logging", type=str, default="True")
args = parser.parse_args()

config_path = BASE_DIR / "agents" / "sac" / "config.yaml"
model_dir = BASE_DIR / "agents" / "sac" / "saved_models"
time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
cwd_path = os.getcwd()

with open(config_path, "rb") as f:
    config = yaml.safe_load(f)

set_seed_everywhere(config["seed"])
config["device"] = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_built() else "cpu")
)
config.update(vars(args))

if args.wandb_logging == "True":
    args.wandb_logging = True
else:
    args.wandb_logging = False

# set torch threads
torch.set_num_threads(1)

if args.load_agent == "False":
    load_agent = False
    test_save_path = ""
    logger.info(
        "Training model for case "
        + str(config["case"])
        + ", year "
        + str(config["year"])
        + ", rep "
        + str(config["rep"])
        + ", T weight "
        + str(config["temp_weight"])
    )
else:
    load_agent = True
    test_save_path = config["temp_weight"]
    logger.info(
        "Evaluating model for case "
        + str(config["case"])
        + ", year "
        + str(config["year"])
        + ", rep "
        + str(config["rep"])
    )

# register environments:
complete_input_file_path = (
    "exp/hannes/Leiden-study/01_evaluate_input/evaluation/case_"
    + str(config["case"])
    + "/year_"
    + str(config["year"])
    + "/rep_"
    + str(0)
    + "/input_c.json"
)
bc = load_building_config(complete_input_file_path)
# bc = load_building_config("input_new.json")
ec = get_envconfig_leiden(config["case"])
ec.map_t_setpoints_to_comfort_space = True
ec.emissions_weight = config["emissions_weight"]
ec.air_quality_weight = config["air_quality_weight"]
ec.temperature_weight = config["temperature_weight"]
# ec.episode_end_date = (3, 1)

building = Building(bc, materials_evaluator(), windows_evaluator())
building.build()
idf = building.get_idf()

environment = (
    "Leiden-case_"
    + str(config["case"])
    + "-year_"
    + str(config["year"])
    + "-rep_"
    + str(config["year"])
)

register_environment(environment, idf, bc, ec)
env = gym.make(environment)
env = LoggerWrapperCubes(env)
# env = DatetimeWrapperCubes(env)

observation_length = env.observation_space.shape[0]
action_length = env.action_space.shape[0]

action_range = [
    env.action_space.low[0],
    env.action_space.high[0],
]

if load_agent:
    agent = load_sac_agent(
        save_path=test_save_path,
        observation_length=observation_length,
        action_length=action_length,
        config=config,
        action_range=action_range,
    )
else:
    agent = SoftActorCritic(
        observation_length=observation_length,
        action_length=action_length,
        device=config["device"],
        name=config["name"],
        batch_size=config["batch_size"],
        discount=config["discount"],
        critic_hidden_dimension=config["critic_hidden_dimension"],
        critic_hidden_layers=config["critic_hidden_layers"],
        critic_betas=config["critic_betas"],
        critic_tau=config["critic_tau"],
        critic_learning_rate=config["critic_learning_rate"],
        critic_target_update_frequency=config["critic_target_update_frequency"],
        actor_hidden_dimension=config["actor_hidden_dimension"],
        actor_hidden_layers=config["actor_hidden_layers"],
        actor_betas=config["actor_betas"],
        actor_learning_rate=config["actor_learning_rate"],
        actor_log_std_bounds=config["actor_log_std_bounds"],
        alpha_learning_rate=config["alpha_learning_rate"],
        alpha_betas=config["alpha_betas"],
        actor_update_frequency=config["actor_update_frequency"],
        init_temperature=config["init_temperature"],
        learnable_temperature=config["learnable_temperature"],
        activation=config["activation"],
        action_range=action_range,
    )

replay_buffer = SoftActorCriticReplayBuffer(
    capacity=config["buffer_capacity"],
    observation_length=observation_length,
    action_length=action_length,
    device=config["device"],
)

workspace = SACWorkspace(
    env=env,
    eval_frequency=config["eval_frequency"],
    eval_rollouts=config["eval_rollouts"],
    model_dir=model_dir,
    seed_steps=config["seed_steps"],
    learning_steps=config["learning_steps"],
    wandb_logging=args.wandb_logging,
)

if __name__ == "__main__":
    if load_agent:
        metrics = workspace.eval(agent=agent, replay_buffer=replay_buffer)
        print(metrics)
    else:
        workspace.train(agent, agent_config=config, replay_buffer=replay_buffer)
