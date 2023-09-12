# pylint: disable=protected-access
# pylint: disable=ungrouped-imports

"""Evaluates the performance of pre-trained agents."""
import yaml
import torch
import datetime
import gym
import os
from cubes.constants import BASE_DIR
from cubes.package.core import register_environment

from agents.sac.agent import SoftActorCritic
from agents.sac.replay_buffer import SoftActorCriticReplayBuffer
from agents.workspaces import SACWorkspace
from agents.utils import set_seed_everywhere

from cubes.package import envconfig
from cubes.construct.buildingconfig import load_building_config
from cubes.construct.building import Building
from cubes.construct.core import materials_evaluator, windows_evaluator

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

case = 0
run_name = "scott-sac"
episodes = 50
one_per_year = True
year = 2022


def make_env(env_id_base, idx):
    def _init():
        env_id = env_id_base + str(idx)
        made_env = gym.make(env_id)
        made_env.reset()

        return made_env

    return _init


environment = "scott-test-env"
input_file_path = (
    BASE_DIR / f"exp/hannes/Leiden-study/00_base_input/case{str(case)}.json"
)
case_path = cwd_path + "/" + run_name + "/case_" + str(case)
complete_input_file_path = case_path + "/input.json"
if not os.path.exists(case_path):
    os.makedirs(case_path)

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
control_vent = True
observe_vent = False
control_observe_battery = False
observe_gcf = [1]
observe_gci = True
if case in [3, 4, 8, 9, 13, 14]:
    control_vent = False
    observe_vent = False
if case >= 10:
    control_observe_battery = True
if case < 5:
    observe_gcf = []
    # observe_gci = False
ec = envconfig.EnvConfig(
    observe_zone_temperature=True,
    observe_electricity_demand=True,
    observe_outside_temperature=True,
    observe_zone_occupancy=True,
    observe_zone_co2=True,
    observe_grid_carbon_intensity=observe_gci,
    observe_zone_thermostat_setpoints=True,
    observe_zone_ventilation=True,
    observe_battery_charge=control_observe_battery,
    observe_batter_charging=control_observe_battery,
    observe_pv_power=control_observe_battery,
    control_battery_charging=control_observe_battery,
    control_ventilation=control_vent,
    control_thermostat_setpoints=True,
    observe_outside_temperature_in_x_hours_forecast=[1],
    observe_grid_carbon_in_x_hours_forecast=observe_gcf,
    emissions_weight=0.5,
    air_quality_weight=0.15,
    episode_end_date=(15, 1),
    timesteps_per_hour=6,
)

building = Building(bc, materials_evaluator(), windows_evaluator())
building.build()
idf = building.get_idf()

environment = "test_env-v1"

register_environment(environment, idf, bc, ec)
env = gym.make(environment)

observation_length = env.observation_space.shape[0]
action_length = env.action_space.shape[0]

action_range = [
    env.action_space.low[0],
    env.action_space.high[0],
]

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
    workspace.train(agent, agent_config=config, replay_buffer=replay_buffer)
