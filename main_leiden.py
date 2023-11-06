# pylint: disable=protected-access
# pylint: disable=ungrouped-imports

"""Evaluates the performance of pre-trained agents."""
import yaml
import torch
import uuid
import gym
import os
from os import makedirs
from loguru import logger
from argparse import ArgumentParser

from agents.sac.agent import SoftActorCritic
from agents.sac.replay_buffer import SoftActorCriticReplayBuffer
from agents.workspaces import LeidenSACWorkspace, DataCollectionWorkspace, RBCWorkspace
from agents.utils import set_seed_everywhere, pull_model_from_wandb

from cubes.rbcs.rbc import GeneralRBC
from cubes.rbcs.constants import (
    zone_names,
    t_control_name,
    occ_name,
    produced_electricity_name,
    electricity_demand_name,
    battery_charging_state_name,
    charge_control_name,
    discharge_control_name,
)

from cubes.constants import BASE_DIR
from cubes.package.core import register_environment
from cubes.construct.buildingconfig import load_building_config
from cubes.construct.building import Building
from cubes.construct.core import materials_evaluator, windows_evaluator
from cubes.package.utilities import get_envconfig_leiden
from cubes.cubesgym.utils.wrappers import LoggerWrapperCubes, DatetimeWrapperCubes

parser = ArgumentParser()
parser.add_argument("--case", type=int)
parser.add_argument("--year", type=int)
parser.add_argument("--rep", type=int, default=0)
parser.add_argument("--algorithm", type=str)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--temperature_weight", type=int, default=1)
parser.add_argument("--emissions_weight", type=int, default=25)
parser.add_argument("--air_quality_weight", type=int, default=1)
parser.add_argument("--load_agent", type=str, default="False")
parser.add_argument("--wandb_logging", type=str, default="True")
parser.add_argument("--collect_dataset", type=str, default="False")
parser.add_argument("--number_logged_rollouts", type=float, default=3)
parser.add_argument("--wandb_run_id", type=str)
parser.add_argument("--wandb_model_id", type=str)
args = parser.parse_args()

# create run dir for running and logging; running in this dir
# allows for parallelization on the cluster
# run dir is a random 128 bit UUID
run_id = str(uuid.uuid4())
run_dir = BASE_DIR / "train" / "runs" / run_id
makedirs(str(run_dir))
os.chdir(run_dir)


if args.algorithm == "sac":
    config_path = BASE_DIR / "agents" / "sac" / "config.yaml"
    model_dir = BASE_DIR / "agents" / "sac" / "saved_models"

elif args.algorithm == "rbc":
    config_path = BASE_DIR / "cubes" / "rbcs" / "config.yaml"

else:
    raise ValueError(f"Unknown algorithm: {args.algorithm}.")

cwd_path = os.getcwd()

with open(config_path, "rb") as f:
    config = yaml.safe_load(f)

config.update(vars(args))
config["run_id"] = run_id

if args.wandb_logging == "True":
    args.wandb_logging = True
else:
    args.wandb_logging = False

if args.collect_dataset == "True":
    args.collect_dataset = True
    complete_input_file_path = (
        BASE_DIR
        / f"train/configs/case_{config['case']}/year_{config['year']}/input_c.json"
    )

else:
    args.collect_dataset = False
    complete_input_file_path = (
        BASE_DIR / f"exp/hannes/Leiden-study/01_evaluate_input/evaluation_new"
        f"/case_{config['case']}/year_{config['year']}/rep_0/input_c.json"
    )

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
        + ", emissions weight "
        + str(config["emissions_weight"])
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

set_seed_everywhere(config["seed"])
config["device"] = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_built() else "cpu")
)

# register environments:
environment = (
    "Leiden-case_"
    + str(config["case"])
    + "-year_"
    + str(config["year"])
    + "-rep_"
    + str(config["year"])
    + "-seed_"
    + str(config["seed"])
)
files_dir = str(BASE_DIR / "inputs" / environment)
makedirs(files_dir, exist_ok=True)

complete_input_file_path = (
    BASE_DIR / f"exp/hannes/Leiden-study/01_evaluate_input/"
    f"evaluation_new/case_{config['case']}/year_{config['year']}"
    f"/rep_0/input_c.json"
)

bc = load_building_config(complete_input_file_path)
# bc = load_building_config("input_new.json")
ec = get_envconfig_leiden(case_number=config["case"], files_dir=files_dir)
ec.map_t_setpoints_to_comfort_space = True

ec.emissions_weight = config["emissions_weight"]
ec.air_quality_weight = config["air_quality_weight"]
ec.temperature_weight = config["temperature_weight"]
# ec.episode_end_date = (3, 1)

building = Building(bc, materials_evaluator(), windows_evaluator())
building.build()
idf = building.get_idf()

register_environment(environment, idf, bc, ec)
env = gym.make(environment)
env = LoggerWrapperCubes(env)
env = DatetimeWrapperCubes(env)

# save config data to run dir
if args.collect_dataset:
    with open(run_dir / "building_config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(bc, f)

    with open(run_dir / "env_config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(ec, f)

observation_length = env.observation_space.shape[0]
action_length = env.action_space.shape[0]

action_range = [
    env.action_space.low[0],
    env.action_space.high[0],
]

if load_agent:
    agent = pull_model_from_wandb(
        algorithm="sac",
        wandb_run_id=args.wandb_run_id,
        wandb_model_id=args.wandb_model_id,
        observation_length=observation_length,
        action_length=action_length,
        config=config,
    )
else:
    if args.algorithm == "sac":
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

        workspace = LeidenSACWorkspace(
            env=env,
            eval_frequency=config["eval_frequency"],
            eval_rollouts=config["eval_rollouts"],
            model_dir=model_dir,
            seed_steps=config["seed_steps"],
            learning_steps=config["learning_steps"],
            wandb_logging=args.wandb_logging,
        )

    elif args.algorithm == "rbc":
        agent = GeneralRBC(
            action_variable_names=env.variables["action"],
            action_ranges=env.setpoints_space,
            observation_variable_names=env.variables["observation"],
            zone_names=zone_names,
            temp_control_names=t_control_name,
            occupancy_variable_names=occ_name,
            electricity_demand_variable_name=electricity_demand_name,
            electricity_supply_variable_name=produced_electricity_name,
            battery_state_variable_name=battery_charging_state_name,
            battery_charge_variable_name=charge_control_name,
            battery_discharge_variable_name=discharge_control_name,
            control_ventilation=ec.control_ventilation,
            control_battery=ec.control_battery_charging,
            temperature_control_method=config["temperature_control_method"],
            ventilation_control_method=config["ventilation_control_method"],
            battery_control_method=config["battery_control_method"],
            open_window_co2=config["open_window_co2"],
            close_window_co2=config["close_window_co2"],
            comfort_temp_setpoint=config["comfort_temp_setpoint"],
            setback_temp_setpoint=config["setback_temp_setpoint"],
            battery_capacity=bc.battery_energy_storage,
            charging_power=bc.battery_power_rating,
            user_type_vent=config["user_type_vent"],
            user_type_temp=config["user_type_temp"],
        )

        workspace = RBCWorkspace(
            env=env,
            wandb_logging=args.wandb_logging,
            eval_rollouts=config["eval_rollouts"],
        )

        replay_buffer = None


if args.collect_dataset:
    workspace = DataCollectionWorkspace(
        env=env,
        eval_frequency=config["eval_frequency"],
        eval_rollouts=config["eval_rollouts"],
        run_dir=run_dir,
        seed_steps=config["seed_steps"],
        learning_steps=config["learning_steps"],
        wandb_logging=args.wandb_logging,
        building_config=bc,
        number_logged_rollouts=config["number_logged_rollouts"],
        building_id=run_id,
    )


if __name__ == "__main__":
    if load_agent or args.algorithm == "rbc":
        metrics = workspace.eval(agent=agent, replay_buffer=replay_buffer)
        print(metrics)
    else:
        workspace.train(agent, agent_config=config, replay_buffer=replay_buffer)
