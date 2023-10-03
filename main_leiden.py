# pylint: disable=protected-access
# pylint: disable=ungrouped-imports

"""Evaluates the performance of pre-trained agents."""
import yaml
import torch
import datetime
import gym
import os
import sys
from cubes.constants import BASE_DIR
from cubes.package.core import register_environment

from agents.sac.agent import SoftActorCritic, load_sac_agent
from agents.sac.replay_buffer import SoftActorCriticReplayBuffer
from agents.workspaces import SACWorkspace
from agents.utils import set_seed_everywhere

from cubes.construct.buildingconfig import load_building_config
from cubes.construct.building import Building
from cubes.construct.core import materials_evaluator, windows_evaluator
from cubes.package.utilities import get_envconfig_leiden
from cubes.cubesgym.utils.wrappers import LoggerWrapperCubes

from loguru import logger


config_path = BASE_DIR / "agents" / "sac" / "config.yaml"
model_dir = BASE_DIR / "agents" / "sac" / "saved_models"
time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
cwd_path = os.getcwd()

with open(config_path, "rb") as f:
    config = yaml.safe_load(f)

set_seed_everywhere(config["seed"])
config["device"] = torch.device(
    "cuda" if torch.cuda.is_available() else ("mps" if torch.has_mps else "cpu")
)

# set torch threads
torch.set_num_threads(1)


if len(sys.argv) < 4:
    logger.error("not enough input arguments")
    sys.exit()

case = int(sys.argv[1])
year = int(sys.argv[2])
rep = int(sys.argv[3])

config["case"] = case
config["year"] = year
config["rep"] = rep


if len(sys.argv) == 4:
    load_agent = False
    test_save_path = ""
    logger.info(
        "Training model for case "
        + str(case)
        + ", year "
        + str(year)
        + ", rep "
        + str(rep)
    )

else:
    load_agent = True
    test_save_path = sys.argv[4]
    logger.info(
        "Evaluating model for case "
        + str(case)
        + ", year "
        + str(year)
        + ", rep "
        + str(rep)
    )

# register environments:
complete_input_file_path = (
    "exp/hannes/Leiden-study/01_evaluate_input/evaluation/case_"
    + str(case)
    + "/year_"
    + str(year)
    + "/rep_"
    + str(0)
    + "/input_c.json"
)
bc = load_building_config(complete_input_file_path)
# bc = load_building_config("input_new.json")
ec = get_envconfig_leiden(case)
ec.map_t_setpoints_to_comfort_space = True
ec.emissions_weight = config["emissions_weight"]
ec.air_quality_weight = config["air_quality_weight"]
ec.temperature_weight = config["temperature_weight"]
# ec.episode_end_date = (3,1)

building = Building(bc, materials_evaluator(), windows_evaluator())
building.build()
idf = building.get_idf()

environment = "Leiden-case_" + str(case) + "-year_" + str(year) + "-rep_" + str(rep)

register_environment(environment, idf, bc, ec)
env = gym.make(environment)
env = LoggerWrapperCubes(env)

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
)

if __name__ == "__main__":
    if load_agent:
        metrics = workspace.eval(agent=agent, replay_buffer=replay_buffer)
        print(metrics)
    else:
        workspace.train(agent, agent_config=config, replay_buffer=replay_buffer)
