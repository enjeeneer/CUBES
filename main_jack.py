# pylint: disable=protected-access
# pylint: disable=ungrouped-imports

"""Evaluates the performance of pre-trained agents."""
import yaml
import torch
import uuid
import gym
import os
import json
from os import makedirs
from loguru import logger
from argparse import ArgumentParser
from agents.sac.agent import SoftActorCritic
from agents.sac.replay_buffer import SoftActorCriticReplayBuffer
from agents.workspaces import (
    LeidenSACWorkspace,
    LeidenPEARLWorkspace,
    DataCollectionWorkspace,
    CostSACWorkspace,
    CostWorkspace,
)
from agents.utils import set_seed_everywhere, pull_model_from_wandb, load_obs_rms

from agents.pearl.agent import PEARL
from agents.pearl.replay_buffer import PEARLReplayBuffer

from cubes.rbcs.rbc import GeneralRBC
from cubes.rbcs.constants import (
    get_temp_name,
    get_t_control_name,
    get_occ_name,
    produced_electricity_name,
    electricity_demand_name,
    battery_charging_state_name,
    charge_control_name,
    discharge_control_name,
    utility_demand_target_control_name,
)

from cubes.constants import BASE_DIR
from cubes.package.core import register_environment
from cubes.construct.buildingconfig import load_building_config
from cubes.construct.building import Building
from cubes.construct.core import materials_evaluator, windows_evaluator
from cubes.package.utilities import (
    get_envconfig_leiden,
    get_envconfig_leiden_minimal,
    get_envconfig_jack,
)
from cubes.cubesgym.utils.wrappers import (
    LoggerWrapperCubes,
    ScaleObservationCubes,
    NormalizeObservationCUBES,
)
import datetime

parser = ArgumentParser()
parser.add_argument("--exp_type", type=str, required=True)
parser.add_argument("--heating_set_temp_seed", type=int, default=42)
parser.add_argument("--heating_on_off_seed", type=int, default=42)
parser.add_argument("--secondary_temp_control", type=str)
parser.add_argument("--iter", type=int)
parser.add_argument("--zone", type=int, default=0)
parser.add_argument("--case", type=int)
parser.add_argument("--year", type=int)
parser.add_argument("--rep", type=int, default=0)
parser.add_argument("--algorithm", type=str)
parser.add_argument("--wandb_entity", type=str, required=True)
parser.add_argument("--wandb_project", type=str, required=True)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--seed_steps", type=int, default=2000)
parser.add_argument("--temperature_weight", type=float, default=1)
parser.add_argument("--emissions_weight", type=float, default=1)
parser.add_argument("--cost_weight", type=float, default=1)
parser.add_argument("--lambda_cost", type=float, default=0.02727)
parser.add_argument("--air_quality_weight", type=float, default=1)
parser.add_argument("--load_agent", type=str, default="False")
parser.add_argument("--wandb_logging", type=str, default="False")
parser.add_argument("--collect_dataset", type=str, default="False")
parser.add_argument("--control_ventilation", type=str, default="True")
parser.add_argument("--reward_function_type", type=str, default="LinearCost")
parser.add_argument("--number_logged_rollouts", type=float, default=3)
parser.add_argument("--wandb_run_id", type=str)
parser.add_argument("--wandb_model_id", type=str)
parser.add_argument("--log_frequency", type=int, default=10)
parser.add_argument("--rbc_switch", type=int, default=1)
parser.add_argument("--comfort_temp_setpoint", type=int, default=20)
parser.add_argument("--comfort_temp_bounds", type=float, default=2)
parser.add_argument("--temperature_margin", type=float, default=1)
parser.add_argument("--setback_temp_setpoint", type=int, default=15)
parser.add_argument("--discount", type=float, default=0.99)
parser.add_argument("--critic_hidden_layers", type=int, default=8)
parser.add_argument("--critic_hidden_dimension", type=int, default=128)
parser.add_argument("--actor_hidden_layers", type=int, default=8)
parser.add_argument("--actor_hidden_dimension", type=int, default=128)
parser.add_argument("--actor_learning_rate", type=float, default=0.0001)
parser.add_argument("--alpha_learning_rate", type=float, default=0.0001)
parser.add_argument("--init_temperature", type=float, default=0.1)
parser.add_argument("--learnable_temperature", type=str, default="True")
parser.add_argument("--critic_learning_rate", type=float, default=0.0001)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--occupancy_schedule", type=str)
parser.add_argument("--normalise_inputs", type=str, default="False")
parser.add_argument("--normalise_observations", type=str, default="False")
parser.add_argument("--scale_observations", type=str, default="False")
parser.add_argument("--normalise_rewards", type=str, default="True")
parser.add_argument("--n_frame_stack", type=int, default=4)
parser.add_argument("--map_setpoints_to_comfort_space", type=str, default="True")
parser.add_argument("--history_length", type=int, default=0)
parser.add_argument("--no_ventilation", type=str, default="False")
parser.add_argument("--battery_only", type=str, default="False")
parser.add_argument("--wandb_tags", nargs="+", type=str, default=[])
parser.add_argument("--wandb_name", type=str, required=True)
parser.add_argument("--timesteps_per_hour", type=int, default=6)
parser.add_argument("--short_episode", type=str, default="False")
parser.add_argument("--critic_target_update_frequency", type=int, default=2)
parser.add_argument("--actor_update_frequency", type=int, default=1)
parser.add_argument("--forecast_length", type=int, default=12)
parser.add_argument("--sleep_hours", type=str, default="True")
parser.add_argument("--emissions_reward_avg_timesteps", type=int, default=1)
parser.add_argument("--minimal_setup", type=str, default="False")
parser.add_argument("--thermal_comfort_bonus", type=float, default=0.0)
parser.add_argument("--thermal_comfort_constant_penalty", type=str, default="False")
parser.add_argument("--discrete_actions", type=str, default="True")
parser.add_argument("--incremental_actions", type=str, default="True")
parser.add_argument("--enforce_ventilation", type=str, default="True")
parser.add_argument("--run_id", type=str, required=True)

args = parser.parse_args()

rbc_name = False

if args.algorithm == "sac":
    config_path = BASE_DIR / "agents" / "sac" / "config.yaml"
    model_dir = BASE_DIR / "agents" / "sac" / "saved_models"

elif args.algorithm == "pearl":
    config_path = BASE_DIR / "agents" / "pearl" / "config.yaml"
    model_dir = BASE_DIR / "agents" / "pearl" / "saved_models"

elif args.algorithm == "rbc":
    if args.rbc_switch == 0:
        rbc_name = "manual"
        config_name = "config_manual.yaml"
    elif args.rbc_switch == 1:
        rbc_name = "comfort"
        config_name = "config_comfort.yaml"
    elif args.rbc_switch == 2:
        rbc_name = "eco"
        config_name = "config_eco.yaml"
    elif args.rbc_switch == 3:
        rbc_name = "constant"
        config_name = "config_constant.yaml"

    config_path = BASE_DIR / "cubes" / "rbcs" / config_name

else:
    raise ValueError(f"Unknown algorithm: {args.algorithm}.")

iters = getattr(args, "iter", False)

args.wandb_name = (
    args.exp_type
    + "_"
    + args.algorithm
    + ("_" + rbc_name if rbc_name else "_" + args.reward_function_type)
    + "_year_"
    + str(args.year)
    + "_case_"
    + str(args.case)
    + "_rep_"
    + str(args.rep)
    + "_zone_"
    + str(args.zone)
    + "_onoffseed_"
    + str(args.heating_on_off_seed)
    + "_tempseed_"
    + str(args.heating_set_temp_seed)
    + "_"
    + args.wandb_name
)

# create a naming structure
current_date = datetime.datetime.now().strftime("%m-%d")
args.wandb_name = current_date + "_" + args.wandb_name

# create run dir for running and logging; running in this dir
# allows for parallelization on the cluster
if args.run_id == "True":
    run_id = args.wandb_name
else:
    run_id = str(uuid.uuid4())

run_dir = BASE_DIR / "train" / "runs" / run_id
makedirs(str(run_dir), exist_ok=True)
os.chdir(run_dir)

cwd_path = os.getcwd()

with open(config_path, "rb") as f:
    config = yaml.safe_load(f)

config.update(vars(args))
config["run_id"] = run_id
if config["short_episode"] == "False":
    config["eval_frequency"] = int(config["timesteps_per_hour"] * 8760)
else:
    config["eval_frequency"] = int(config["timesteps_per_hour"] * 360)
    # config["seed_steps"] = int(2 * config["timesteps_per_hour"] * 360)
    # config["seed_steps"] = 10

if args.wandb_logging == "True":
    args.wandb_logging = True
else:
    args.wandb_logging = False

if args.collect_dataset == "True":
    args.collect_dataset = True
else:
    args.collect_dataset = False

if args.control_ventilation == "True":
    config["control_ventilation"] = True
else:
    config["control_ventilation"] = False

if args.no_ventilation == "True":
    config["air_quality_weight"] = 0
    config["control_ventilation"] = False


if args.normalise_inputs == "True":
    config["normalisation_samples"] = config["seed_steps"]
else:
    config["normalisation_samples"] = None

if args.exp_type == "thermostat":

    # base_path = (
    #    BASE_DIR / f"exp/jack/paper/thermostat_experiment/input/"
    #    f"case{config['case']}/rep{config['rep']}"
    # )
    # if isinstance(iters, int):
    #    complete_input_file_path = base_path / f"iter{iters}.json"
    # else:
    #    complete_input_file_path = base_path / "baseline.json"

    complete_input_file_path = (
        BASE_DIR / f"exp/jack/paper/thermostat_experiment/input/case{config['case']}"
        f"/zone{config['zone']}.json"
    )

elif args.exp_type == "cost":
    complete_input_file_path = (
        BASE_DIR / f"exp/jack/paper/cost_experiment/input/case{config['case']}"
        f"/case_{config['case']}_{config['zone']}.json"
    )

elif args.exp_type == "zoning":
    if iters:
        complete_input_file_path = (
            BASE_DIR / f"exp/jack/paper/zoning_experiment/input/rep{config['rep']}"
            f"/case0_iter{config['iter']}.json"
        )
    else:
        complete_input_file_path = (
            BASE_DIR / "exp/jack/paper/zoning_experiment/input/case_0_0.json"
        )
else:
    raise Exception(f"Unknow experiment type {args.exp_type}")

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
        + ", t comfort "
        + str(config["comfort_temp_setpoint"])
        + ", t setback "
        + str(config["setback_temp_setpoint"])
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
        + ", t comfort "
        + str(config["comfort_temp_setpoint"])
        + ", t setback "
        + str(config["setback_temp_setpoint"])
    )

results_path = BASE_DIR / "results"
if not os.path.exists(str(results_path)):
    os.makedirs(str(results_path))

results_name = (
    "case_"
    + str(config["case"])
    + "_year_"
    + str(config["year"])
    + "_rep_"
    + str(config["rep"])
    + "_emissions_weight_"
    + str(config["emissions_weight"])
    + "_t_comfort_"
    + str(config["comfort_temp_setpoint"])
    + "_t_setback_"
    + str(config["setback_temp_setpoint"])
    + "_tags_"
    + "-".join(config["wandb_tags"])
)

if args.algorithm != "rbc":
    set_seed_everywhere(config["seed"])

config["device"] = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_built() else "cpu")
)

files_dir = str(BASE_DIR / "inputs" / run_id)
makedirs(files_dir, exist_ok=True)

bc = load_building_config(
    path_to_datafile=complete_input_file_path, files_dir=files_dir
)

if args.exp_type == "zoning":
    bc.occupant_schedule_file_name = f"zoning_schedule_rep_{config['rep']}.sch"

bc.occupant_schedule_file_name = (
    f"thermostat_exp/rep{config['rep']}/schedule_rep_{config['rep']}.sch"
)

if args.year == 2023:
    bc.year = 2023
    bc.weather_file_name = "Cambridgeshire_CC_2023.epw"
    bc.grid_carbon_intensity_file_name = "grid_carbon_GB_10min_2023.csv"
    bc.gas_pricing_file_name = "csv_gastracker_A_Eastern_England_2023.csv"
    bc.electricity_pricing_file_name = "csv_agile_A_Eastern_England_2023.csv"


bc.heating_setpoint = config["comfort_temp_setpoint"]
bc.heating_setback = config["setback_temp_setpoint"]
if config["no_ventilation"] == "True":
    bc.natural_ventilation_rate_open_windows = 0

# bc.use_operative_temperature = False

if args.algorithm == "rbc":
    ec = get_envconfig_jack(
        case_number=config["case"],
        comfort_temp=config["comfort_temp_setpoint"],
        reward_function_type=args.reward_function_type,
        rbc_setup=True,
        files_dir=files_dir,
        short_test=config["short_episode"] == "True",
        forecast_length=0,
        sleep_hours=config["sleep_hours"] == "True",
    )
else:
    if config["minimal_setup"] == "True":
        ec = get_envconfig_leiden_minimal(
            case_number=config["case"],
            comfort_temp=config["comfort_temp_setpoint"],
            files_dir=files_dir,
            short_test=config["short_episode"] == "True",
            forecast_length=config["forecast_length"],
            sleep_hours=config["sleep_hours"] == "True",
        )
    elif config["reward_function_type"] in ["LinearCost", "LinearEmissions"]:
        ec = get_envconfig_jack(
            case_number=config["case"],
            comfort_temp=config["comfort_temp_setpoint"],
            reward_function_type=args.reward_function_type,
            files_dir=files_dir,
            short_test=config["short_episode"] == "True",
            forecast_length=config["forecast_length"],
            sleep_hours=config["sleep_hours"] == "True",
        )
    else:
        ec = get_envconfig_leiden(
            case_number=config["case"],
            comfort_temp=config["comfort_temp_setpoint"],
            files_dir=files_dir,
            short_test=config["short_episode"] == "True",
            forecast_length=config["forecast_length"],
            sleep_hours=config["sleep_hours"] == "True",
        )


if args.map_setpoints_to_comfort_space == "True":
    ec.map_t_setpoints_to_comfort_space = False  # TODO: check if this is necessary
else:
    ec.map_t_setpoints_to_comfort_space = False

ec.enforce_ventilation = config["enforce_ventilation"] == "True"

ec.emissions_weight = config["emissions_weight"]
ec.cost_weight = config["cost_weight"]
ec.lambda_cost = config["lambda_cost"] * ec.lambda_cost
ec.air_quality_weight = config["air_quality_weight"]
ec.temperature_weight = config["temperature_weight"]
ec.timesteps_per_hour = config["timesteps_per_hour"]
ec.temperature_margin = config["temperature_margin"]
ec.emissions_reward_avg_n_timesteps = config["emissions_reward_avg_timesteps"]
ec.thermal_comfort_bonus = config["thermal_comfort_bonus"]
ec.thermal_comfort_constant_penalty = (
    config["thermal_comfort_constant_penalty"] == "True"
)
# fix battery storage strategy to be charge/discharge
ec.battery_storage_operation = "DemandLevelling"
ec.discrete_battery_actions = config["discrete_actions"] == "True"
ec.discrete_window_actions = config["discrete_actions"] == "True"
ec.incremental_actions = config["incremental_actions"] == "True"  # "False"
if config["reward_function_type"] in [
    "Tolerance",
    "Linear",
    "LinearCost",
    "LinearEmissions",
]:
    ec.reward_function_type = config["reward_function_type"]
else:
    raise ValueError(f"Unknown reward function type: {config['reward_function_type']}.")
# ec.episode_end_date = (3, 1)

building = Building(bc, materials_evaluator(), windows_evaluator())
building.build()
idf = building.get_idf()

pearl_reward_function = register_environment(run_id, idf, bc, ec)
env = gym.make(run_id)
env = LoggerWrapperCubes(env)

# if args.algorithm == "sac":
#    env = DatetimeWrapperCubes(env)

# env = ObservationFilterCubes(env)

if args.algorithm == "sac" and config["n_frame_stack"] > 1:
    env = gym.wrappers.FrameStack(env, num_stack=config["n_frame_stack"])
    env = gym.wrappers.FlattenObservation(env)
if args.algorithm == "sac" and config["scale_observations"] == "True":
    env = ScaleObservationCubes(env)
if args.algorithm == "sac" and config["normalise_observations"] == "True":
    if load_agent:
        obs_rms = load_obs_rms(
            algorithm="sac",
            wandb_run_id=args.wandb_run_id,
            wandb_model_id=args.wandb_model_id,
        )
        env = NormalizeObservationCUBES(env, obs_rms=obs_rms)

    else:
        env = gym.wrappers.NormalizeObservation(env)
if args.algorithm == "sac" and config["normalise_rewards"] == "True":
    env = gym.wrappers.NormalizeReward(env)


# save config data to run dir
if args.collect_dataset:
    with open(run_dir / "building_config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(bc, f)

    with open(run_dir / "env_config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(ec, f)

observation_length = env.observation_space.shape[0]
action_length = env.action_space.shape[0]
if args.battery_only == "True" and config["case"] > 10:  # cases > 10 have battery
    action_length = action_length - 2  # remove thermostats
elif args.battery_only == "True" and config["case"] <= 10:
    raise ValueError("Battery only not possible for case <= 10.")

if ec.battery_storage_operation == "TrackChargeDischargeSchedules":
    action_length = (
        action_length - 1
    )  # make agent output one charge/discharge action instead of 2

action_range = [
    env.action_space.low[0],
    env.action_space.high[0],
]

if load_agent:
    agent = pull_model_from_wandb(
        algorithm="sac",
        wandb_entity=args.wandb_entity,
        wandb_project_id=args.wandb_project,
        wandb_run_id=args.wandb_run_id,
        wandb_model_id=args.wandb_model_id,
        observation_length=observation_length,
        action_length=action_length,
        config=config,
    )
    workspace = LeidenSACWorkspace(
        env=env,
        eval_frequency=config["eval_frequency"],
        eval_rollouts=config["eval_rollouts"],
        model_dir=model_dir,
        seed_steps=config["seed_steps"],
        learning_steps=config["learning_steps"],
        wandb_logging=args.wandb_logging,
        log_frequency=config["log_frequency"],
        wandb_entity=args.wandb_entity,
        wandb_project=args.wandb_project,
        wandb_tags=args.wandb_tags,
        wandb_name=args.wandb_name,
        action_length=action_length,
        battery_only=args.battery_only == "True",
        thermostat_setpoint=config["comfort_temp_setpoint"],
        action_variable_names=env.variables["action"],
        battery_demand_levelling=ec.battery_storage_operation == "DemandLevelling",
        normalized_observations=config["normalise_observations"] == "True",
    )

    replay_buffer = None

else:
    if args.algorithm == "sac" and (
        ec.reward_function_type in ["LinearCost", "LinearEmissions"]
    ):
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
            learnable_temperature=config["learnable_temperature"] == "True",
            activation=config["activation"],
            action_range=action_range,
            history_length=config["history_length"],
            normalisation_samples=config["normalisation_samples"],
        )

        replay_buffer = SoftActorCriticReplayBuffer(
            capacity=config["buffer_capacity"],
            observation_length=observation_length,
            action_length=action_length,
            device=config["device"],
            history_length=config["history_length"],
        )
        workspace = CostSACWorkspace(
            env=env,
            eval_frequency=config["eval_frequency"],
            eval_rollouts=config["eval_rollouts"],
            model_dir=model_dir,
            seed_steps=config["seed_steps"],
            learning_steps=config["learning_steps"],
            wandb_logging=args.wandb_logging,
            log_frequency=config["log_frequency"],
            wandb_entity=args.wandb_entity,
            wandb_project=args.wandb_project,
            wandb_tags=args.wandb_tags,
            wandb_name=args.wandb_name,
            action_length=action_length,
            battery_only=args.battery_only == "True",
            thermostat_setpoint=config["comfort_temp_setpoint"],
            action_variable_names=env.variables["action"],
            battery_demand_levelling=ec.battery_storage_operation == "DemandLevelling",
            normalized_observations=config["normalise_observations"] == "True",
        )

    elif args.algorithm == "sac" and ec.reward_function_type == "Linear":
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
            learnable_temperature=config["learnable_temperature"] == "True",
            activation=config["activation"],
            action_range=action_range,
            history_length=config["history_length"],
            normalisation_samples=config["normalisation_samples"],
        )

        replay_buffer = SoftActorCriticReplayBuffer(
            capacity=config["buffer_capacity"],
            observation_length=observation_length,
            action_length=action_length,
            device=config["device"],
            history_length=config["history_length"],
        )
        workspace = LeidenSACWorkspace(
            env=env,
            eval_frequency=config["eval_frequency"],
            eval_rollouts=config["eval_rollouts"],
            model_dir=model_dir,
            seed_steps=config["seed_steps"],
            learning_steps=config["learning_steps"],
            wandb_logging=args.wandb_logging,
            log_frequency=config["log_frequency"],
            wandb_entity=args.wandb_entity,
            wandb_project=args.wandb_project,
            wandb_tags=args.wandb_tags,
            wandb_name=args.wandb_name,
            action_length=action_length,
            battery_only=args.battery_only == "True",
            thermostat_setpoint=config["comfort_temp_setpoint"],
            action_variable_names=env.variables["action"],
            battery_demand_levelling=ec.battery_storage_operation == "DemandLevelling",
            normalized_observations=config["normalise_observations"] == "True",
        )

    elif args.algorithm == "pearl":
        config["update_frequency"] = (
            ec.timesteps_per_hour * config["hours_between_update"]
        )

        agent = PEARL(
            observation_length=observation_length,
            action_length=action_length,
            history_length=config["history_length"],
            device=config["device"],
            name=config["name"],
            batch_size=config["batch_size"],
            discount=config["discount"],
            ensemble_size=config["ensemble_size"],
            dynamics_hidden_dimension=config["dynamics_hidden_dimension"],
            dynamics_hidden_layers=config["dynamics_hidden_layers"],
            dynamics_learning_rate=config["dynamics_learning_rate"],
            dynamics_activation=config["dynamics_activation"],
            dynamics_betas=config["dynamics_betas"],
            observation_space=env.observation_space,
            planning_particles=config["planning_particles"],
            planning_population=config["planning_population"],
            planning_init_mean=config["planning_init_mean"],
            planning_init_var=config["planning_init_var"],
            planning_horizon=config["planning_horizon"],
            planning_iterations=config["planning_iterations"],
            planning_elite_fraction=config["planning_elite_fraction"],
            planning_temperature=config["planning_temperature"],
            planning_momentum=config["planning_momentum"],
            forecast_idxs=None,
            reward_function=pearl_reward_function,
            learning_steps_per_update=config["learning_steps_per_update"],
        )

        replay_buffer = PEARLReplayBuffer(
            capacity=config["buffer_capacity"],
            observation_length=observation_length,
            history_length=config["history_length"],
            action_length=action_length,
            device=config["device"],
        )

        workspace = LeidenPEARLWorkspace(
            env=env,
            training_steps=config["training_steps"],
            model_dir=model_dir,
            eval_frequency=config["eval_frequency"],
            update_frequency=config["update_frequency"],
            wandb_logging=args.wandb_logging,
            log_frequency=config["log_frequency"],
            wandb_entity=args.wandb_entity,
            wandb_project=args.wandb_project,
            wandb_tags=args.wandb_tags,
            wandb_name=args.wandb_name,
            eval_rollouts=config["eval_rollouts"],
            seed_steps=config["seed_steps"],
        )

    elif args.algorithm == "rbc":
        zone_names_flattened = [zone for sublist in bc.zone_names for zone in sublist]
        no_vent_con = (config["case"] in [3, 4, 8, 9, 13, 14]) or not config[
            "control_ventilation"
        ]
        ventilation_control = (
            None if no_vent_con else config["ventilation_control_method"]
        )
        batt_con = config["battery_control_method"] if config["case"] >= 10 else None
        if ec.battery_storage_operation == "TrackChargeDischargeSchedules":
            batt_con = "excess_storage"
        Tset = (
            config["comfort_temp_setpoint"] + 0.3
            if no_vent_con
            else config["comfort_temp_setpoint"]
        )

        # Set seed for random temp setpoint and heating times
        heating_set_temp_seed = args.heating_set_temp_seed
        heating_on_off_seed = args.heating_on_off_seed

        # Set how the other zones should be controlled

        # secondary_temp_control = args.secondary_temp_control

        secondary_temp_control_names = get_t_control_name(bc.secondary_controlled_zones)

        if bc.secondary_controlled_zones:
            secondary_temp_control = "switch_onoff"
            print(secondary_temp_control)
        else:
            secondary_temp_control = None
        temperature_control_method = "occupancy"

        agent = GeneralRBC(
            action_variable_names=env.variables["action"],
            action_ranges=env.setpoints_space,
            observation_variable_names=env.variables["observation"],
            primary_temp_zone_names=bc.primary_controlled_zones,
            secondary_temp_zone_names=bc.secondary_controlled_zones,
            primary_temp_control_names=get_t_control_name(bc.primary_controlled_zones),
            temperature_names=get_temp_name(
                bc.use_operative_temperature, zone_names_flattened
            ),
            secondary_temp_control_names=get_t_control_name(
                bc.secondary_controlled_zones
            ),
            heating_set_temp_seed=heating_set_temp_seed,
            heating_on_off_seed=heating_on_off_seed,
            occupancy_variable_names=get_occ_name(zone_names_flattened),
            electricity_demand_variable_name=electricity_demand_name,
            electricity_supply_variable_name=produced_electricity_name,
            battery_state_variable_name=battery_charging_state_name,
            battery_charge_variable_name=charge_control_name,
            battery_discharge_variable_name=discharge_control_name,
            utility_demand_target_control_name=utility_demand_target_control_name,
            control_ventilation=ec.control_ventilation,
            control_battery=ec.control_battery_charging,
            temperature_control_method=temperature_control_method,
            secondary_temp_control=secondary_temp_control,
            ventilation_control_method=ventilation_control,
            battery_control_method=batt_con,
            open_window_co2=config["open_window_co2"],
            close_window_co2=config["close_window_co2"],
            comfort_temp_setpoint=Tset,
            setback_temp_setpoint=config["setback_temp_setpoint"],
            battery_capacity=bc.battery_energy_storage,
            charging_power=bc.battery_power_rating,
            t_switch_onoff_times="random",
            sleep_hours=ec.sleep_hours,
        )

        workspace = CostWorkspace(
            env=env,
            wandb_logging=args.wandb_logging,
            wandb_entity=args.wandb_entity,
            wandb_project=args.wandb_project,
            wandb_tags=args.wandb_tags,
            wandb_name=args.wandb_name,
            eval_rollouts=config["eval_rollouts"],
        )

        # workspace = LeidenWorkspace(
        #    env=env,
        #    wandb_logging=args.wandb_logging,
        #    wandb_entity=args.wandb_entity,
        #    wandb_project=args.wandb_project,
        #    wandb_tags=args.wandb_tags,
        #    wandb_name=args.wandb_name,
        #    eval_rollouts=config["eval_rollouts"],
        # )

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
        metrics = workspace.eval(
            agent=agent,
            replay_buffer=replay_buffer,
            checkpoints=False,
            agent_config=config,
            full_logging=True,
        )
        with open(
            str(results_path) + "/" + results_name + ".json", "w", encoding="utf-8"
        ) as fp:
            json.dump(metrics, fp)

    else:
        workspace.train(agent, agent_config=config, replay_buffer=replay_buffer)
        metrics = workspace.eval(
            agent=agent,
            replay_buffer=replay_buffer,
            checkpoints=False,
            agent_config=config,
            full_logging=True,
        )
