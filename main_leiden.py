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
from agents.workspaces import (
    LeidenWorkspace,
    LeidenSACWorkspace,
    LeidenPEARLWorkspace,
    DataCollectionWorkspace,
)
from agents.utils import set_seed_everywhere, pull_model_from_wandb

from agents.pearl.agent import PEARL
from agents.pearl.replay_buffer import PEARLReplayBuffer

from cubes.rbcs.rbc import GeneralRBC
from cubes.rbcs.constants import (
    zone_names,
    get_temp_name,
    t_control_name,
    occ_name,
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
from cubes.package.utilities import get_envconfig_leiden
from cubes.cubesgym.utils.wrappers import LoggerWrapperCubes, DatetimeWrapperCubes

parser = ArgumentParser()
parser.add_argument("--case", type=int)
parser.add_argument("--year", type=int)
parser.add_argument("--rep", type=int, default=0)
parser.add_argument("--algorithm", type=str)
parser.add_argument("--wandb_entity", type=str, required=True)
parser.add_argument("--wandb_project", type=str, required=True)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--seed_steps", type=int, default=2000)
parser.add_argument("--temperature_weight", type=int, default=1)
parser.add_argument("--emissions_weight", type=int, default=1)
parser.add_argument("--air_quality_weight", type=int, default=1)
parser.add_argument("--load_agent", type=str, default="False")
parser.add_argument("--wandb_logging", type=str, default="True")
parser.add_argument("--collect_dataset", type=str, default="False")
parser.add_argument("--control_ventilation", type=str, default="False")
parser.add_argument("--reward_function_type", type=str, default="Tolerance")
parser.add_argument("--number_logged_rollouts", type=float, default=3)
parser.add_argument("--wandb_run_id", type=str)
parser.add_argument("--wandb_model_id", type=str)
parser.add_argument("--log_frequency", type=int, default=10)
parser.add_argument("--rbc_switch", type=int, default=1)
parser.add_argument("--comfort_temp_setpoint", type=int, default=20)
parser.add_argument("--comfort_temp_bounds", type=int, default=2)
parser.add_argument("--temperature_margin", type=int, default=1)
parser.add_argument("--setback_temp_setpoint", type=int, default=17)
parser.add_argument("--discount", type=float, default=0.99)
parser.add_argument("--critic_hidden_layers", type=int, default=2)
parser.add_argument("--critic_hidden_dimension", type=int, default=128)
parser.add_argument("--actor_hidden_layers", type=int, default=2)
parser.add_argument("--actor_hidden_dimension", type=int, default=128)
parser.add_argument("--actor_learning_rate", type=float, default=0.0001)
parser.add_argument("--alpha_learning_rate", type=float, default=0.0001)
parser.add_argument("--init_temperature", type=float, default=0.1)
parser.add_argument("--critic_learning_rate", type=float, default=0.00005)
parser.add_argument("--occupancy_schedule", type=str)
parser.add_argument("--map_setpoints_to_comfort_space", type=str, default="True")
parser.add_argument("--history_length", type=int, default=0)
parser.add_argument("--no_ventilation", type=str, default="True")
parser.add_argument("--wandb_tags", nargs="+", type=str, default=[])
parser.add_argument("--timesteps_per_hour", type=int, default=6)
parser.add_argument("--short_episode", type=str, default="False")
parser.add_argument("--critic_target_update_frequency", type=int, default=2)
parser.add_argument("--actor_update_frequency", type=int, default=1)
parser.add_argument("--forecast_length", type=int, default=0)
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

cwd_path = os.getcwd()

with open(config_path, "rb") as f:
    config = yaml.safe_load(f)

config.update(vars(args))
config["run_id"] = run_id
if config["short_episode"] == "False":
    config["eval_frequency"] = int(config["timesteps_per_hour"] * 8760)
else:
    config["eval_frequency"] = int(config["timesteps_per_hour"] * 360)
    config["seed_steps"] = int(2 * config["timesteps_per_hour"] * 360)

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


# occupancy
assert args.occupancy_schedule in [
    "always_occupied",
    "daytime_occupancy",
    "deterministic_occupancy",
    "stochastic_occupancy",
]
eplus_config_dir = f"evaluation_{args.occupancy_schedule}"

complete_input_file_path = (
    BASE_DIR / f"exp/hannes/Leiden-study/01_evaluate_input/{eplus_config_dir}"
    f"/case_{config['case']}/year_{config['year']}/rep_{config['rep']}/input_c.json"
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
bc.heating_setpoint = config["comfort_temp_setpoint"]
bc.heating_setback = config["setback_temp_setpoint"]

ec = get_envconfig_leiden(
    case_number=config["case"],
    control_vent=config["control_ventilation"],
    comfort_temp_setpoint=config["comfort_temp_setpoint"],
    comfort_temp_bounds=config["comfort_temp_bounds"],
    rbc_setup=True,
    files_dir=files_dir,
    short_test=config["short_episode"] == "True",
    forecast_length=0,
)
if args.algorithm == "rbc":
    pass
else:
    ec = get_envconfig_leiden(
        case_number=config["case"],
        control_vent=config["control_ventilation"],
        comfort_temp_setpoint=config["comfort_temp_setpoint"],
        comfort_temp_bounds=config["comfort_temp_bounds"],
        files_dir=files_dir,
        short_test=config["short_episode"] == "True",
        forecast_length=config["forecast_length"],
    )

if args.map_setpoints_to_comfort_space == "True":
    ec.map_t_setpoints_to_comfort_space = True  # TODO: check if this is necessary
else:
    ec.map_t_setpoints_to_comfort_space = False

ec.emissions_weight = config["emissions_weight"]
ec.air_quality_weight = config["air_quality_weight"]
ec.temperature_weight = config["temperature_weight"]
ec.timesteps_per_hour = config["timesteps_per_hour"]
ec.temperature_margin = config["temperature_margin"]

if config["reward_function_type"] in ["Tolerance", "Linear"]:
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
if args.algorithm == "sac":
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
    )

    replay_buffer = None

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
            history_length=config["history_length"],
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
            eval_rollouts=config["eval_rollouts"],
            seed_steps=config["seed_steps"],
        )

    elif args.algorithm == "rbc":
        no_vent_con = config["case"] in [3, 4, 8, 9, 13, 14]
        ventilation_control = (
            None if no_vent_con else config["ventilation_control_method"]
        )
        print("rbc ventilation control: ", ventilation_control)
        batt_con = config["battery_control_method"] if config["case"] >= 10 else None
        # Tset = (
        #     config["comfort_temp_setpoint"] + 0.3
        #     if no_vent_con
        #     else config["comfort_temp_setpoint"]
        # )
        Tset = config["comfort_temp_setpoint"]
        agent = GeneralRBC(
            action_variable_names=env.variables["action"],
            action_ranges=env.setpoints_space,
            observation_variable_names=env.variables["observation"],
            zone_names=zone_names,
            temp_control_names=t_control_name,
            temperature_names=get_temp_name(bc.use_operative_temperature),
            occupancy_variable_names=occ_name,
            electricity_demand_variable_name=electricity_demand_name,
            electricity_supply_variable_name=produced_electricity_name,
            battery_state_variable_name=battery_charging_state_name,
            battery_charge_variable_name=charge_control_name,
            battery_discharge_variable_name=discharge_control_name,
            utility_demand_target_control_name=utility_demand_target_control_name,
            control_ventilation=ec.control_ventilation,
            control_battery=ec.control_battery_charging,
            temperature_control_method=config["temperature_control_method"],
            ventilation_control_method=ventilation_control,
            battery_control_method=batt_con,
            open_window_co2=config["open_window_co2"],
            close_window_co2=config["close_window_co2"],
            comfort_temp_setpoint=Tset,
            setback_temp_setpoint=config["setback_temp_setpoint"],
            battery_capacity=bc.battery_energy_storage,
            charging_power=bc.battery_power_rating,
        )

        workspace = LeidenWorkspace(
            env=env,
            wandb_logging=args.wandb_logging,
            wandb_entity=args.wandb_entity,
            wandb_project=args.wandb_project,
            wandb_tags=args.wandb_tags,
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
        metrics = workspace.eval(
            agent=agent,
            replay_buffer=replay_buffer,
            checkpoints=False,
            agent_config=config,
            full_logging=True,
        )
        print(metrics)
    else:
        workspace.train(agent, agent_config=config, replay_buffer=replay_buffer)
