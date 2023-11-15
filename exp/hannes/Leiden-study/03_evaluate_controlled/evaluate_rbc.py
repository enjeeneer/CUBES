# pylint: disable=consider-iterating-dictionary

"""script to run the cases with control for the Leiden paper"""
from cubes.construct.buildingconfig import load_building_config
from cubes.construct.building import Building
from cubes.construct.core import materials_evaluator, windows_evaluator
from cubes.package.core import register_environment
from cubes.cubesgym.utils.wrappers import LoggerWrapperCubes
from cubes.rbcs.rbc import GeneralRBC
from cubes.rbcs.constants import (
    zone_names,
    t_set_name,
    occ_name,
    produced_electricity_name,
    electricity_demand_name,
    battery_charging_state_name,
    charge_control_name,
    discharge_control_name,
)
from cubes.constants import BASE_DIR
from cubes.package.utilities import get_envconfig_leiden

from loguru import logger
from tqdm.auto import tqdm
import os
import yaml
import gym
import sys
import numpy as np
import json

if len(sys.argv) != 5:
    logger.error(f"Need 4 input values, but {len(sys.argv)-1} provided")
    sys.exit()


cwd_path = os.getcwd()
print(cwd_path)

materials = materials_evaluator()
windows = windows_evaluator()

i_case = int(sys.argv[1])
year = int(sys.argv[2])
rep = int(sys.argv[3])
rbc_switch = int(sys.argv[4])



if rbc_switch == 0:
    rbc_name = "manual"
    config_name = "config_manual.yaml"
elif rbc_switch == 1:
    rbc_name = "comfort"
    config_name = "config_comfort.yaml"
elif rbc_switch == 2:
    rbc_name = "eco"
    config_name = "config_eco.yaml"
else:
    rbc_name = "constant"
    config_name = "config_constant.yaml"

config_path = BASE_DIR / "cubes" / "rbcs" / config_name


with open(config_path, "rb") as f:
    config = yaml.safe_load(f)

results_path = "results/"
if not os.path.exists(results_path):
    os.makedirs(results_path)


environment = (
    "evaluate_RBC_"
    + rbc_name
    + "_case_"
    + str(i_case)
    + "_year_"
    + str(year)
    + "_rep_"
    + str(rep)
    + "-v1"
)

complete_input_file_path = (
    "../01_evaluate_input/evaluation_new/case_"
    + str(i_case)
    + "/year_"
    + str(year)
    + "/rep_"
    + str(rep)
    + "/input_c.json"
)
files_dir = str(BASE_DIR / "inputs" / environment)
os.makedirs(files_dir, exist_ok=True)

BC = load_building_config(complete_input_file_path)
EC = get_envconfig_leiden(case_number=i_case,
                          files_dir=files_dir,
                          rbc_setup=True)
# EC.control_battery_charging = False
###testing
# EC.observe_battery_charge = False
# EC.observe_battery_charging = False

building = Building(BC, materials, windows)
building.build()
idf = building.get_idf()

if environment in gym.envs.registry.keys():
    del gym.envs.registry.env_specs[environment]

register_environment(environment, idf, BC, EC)

env = gym.make(environment)
env = LoggerWrapperCubes(env)

# pylint: disable=protected-access
n_timesteps_episode = (
    env.simulator._eplus_one_epi_len / env.simulator._eplus_run_stepsize
)

# rbc = TrivialRBC(env.variables["action"],env.action_space_real,
#                  env.variables["observation"],20,1000)
no_vent_con = i_case in [3, 4, 8, 9, 13, 14]
ventilation_control = None if no_vent_con else config["ventilation_control_method"]
#batt_con = "excess_storage" if i_case >= 10 else None
batt_con = None
Tset = (config["comfort_temp_setpoint"]+0.3
        if no_vent_con else config["comfort_temp_setpoint"])
# batt_con = None

rbc = GeneralRBC(
    action_variable_names=env.variables["action"],
    action_ranges=env.setpoints_space,
    observation_variable_names=env.variables["observation"],
    zone_names=zone_names,
    temp_control_names=t_set_name,
    occupancy_variable_names=occ_name,
    electricity_demand_variable_name=electricity_demand_name,
    electricity_supply_variable_name=produced_electricity_name,
    battery_state_variable_name=battery_charging_state_name,
    battery_charge_variable_name=charge_control_name,
    battery_discharge_variable_name=discharge_control_name,
    control_ventilation=EC.control_ventilation,
    control_battery=EC.control_battery_charging,
    temperature_control_method=config["temperature_control_method"],
    ventilation_control_method=config["ventilation_control_method"],
    battery_control_method=config["battery_control_method"],
    open_window_co2=config["open_window_co2"],
    close_window_co2=config["close_window_co2"],
    comfort_temp_setpoint=config["comfort_temp_setpoint"],
    setback_temp_setpoint=config["setback_temp_setpoint"],
    battery_capacity=BC.battery_energy_storage,
    charging_power=BC.battery_power_rating,
)

eval_rewards = []
eval_emissions = []
eval_ndt_t_violations = {}
eval_ndt_aq_violations = {}
eval_heating_dt = {}
eval_heating_beyond_comf_dt = {}
eval_violation_dt = {}
eval_violation_daq = {}
eval_emissions_reward = []
eval_comfort_reward = []
eval_aq_reward = []

done = False
rollout_reward = 0.0
rollout_emissions = 0.0
rollout_ndt_t_violations = {}
rollout_ndt_aq_violations = {}
rollout_heating_dt = {}
rollout_heating_beyond_comf_dt = {}
rollout_violation_dt = {}
rollout_violation_daq = {}
rollout_emissions_reward = 0.0
rollout_comfort_reward = 0.0
rollout_aq_reward = 0.0

obs = env.reset()
# ts = 0
with tqdm(total=n_timesteps_episode) as pbar:
    while not done:
        # ts += 1
        action = rbc.act(obs)
        obs, reward, done, info = env.step(action)

        rollout_reward += reward
        rollout_emissions += info["emissions"]
        if not rollout_ndt_t_violations:
            for k, v in info["t_violation"].items():
                rollout_ndt_t_violations[k] = v
        else:
            for k, v in info["t_violation"].items():
                rollout_ndt_t_violations[k] += v

        if not rollout_ndt_aq_violations:
            for k, v in info["aq_violation"].items():
                rollout_ndt_aq_violations[k] = v
        else:
            for k, v in info["aq_violation"].items():
                rollout_ndt_aq_violations[k] += v

        if not rollout_heating_dt:
            for k, v in info["heating_delta_T"].items():
                rollout_heating_dt[k] = v / 144
        else:
            for k, v in info["heating_delta_T"].items():
                rollout_heating_dt[k] += v / 144

        if not rollout_heating_beyond_comf_dt:
            for k, v in info["heating_beyond_comf_delta_T"].items():
                rollout_heating_beyond_comf_dt[k] = v / 144
        else:
            for k, v in info["heating_beyond_comf_delta_T"].items():
                rollout_heating_beyond_comf_dt[k] += v / 144

        if not rollout_violation_dt:
            for k, v in info["violation_delta_T"].items():
                rollout_violation_dt[k] = v / 144
        else:
            for k, v in info["violation_delta_T"].items():
                rollout_violation_dt[k] += v / 144

        if not rollout_violation_daq:
            for k, v in info["violation_delta_aq"].items():
                rollout_violation_daq[k] = v / 144
        else:
            for k, v in info["violation_delta_aq"].items():
                rollout_violation_daq[k] += v / 144

        rollout_emissions_reward += info["reward_emissions"]
        rollout_comfort_reward += info["reward_comfort"]
        rollout_aq_reward += info["reward_air_quality"]

        pbar.update(1)
        # if ts % 1000 == 0:
        #     print(f"progress: {ts / n_timesteps_episode*100:.2f}%")

env.close()

eval_rewards.append(rollout_reward)
eval_emissions.append(rollout_emissions)

if not eval_ndt_t_violations:
    for k, v in rollout_ndt_t_violations.items():
        eval_ndt_t_violations[k] = [v]
else:
    for k, v in rollout_ndt_t_violations.items():
        eval_ndt_t_violations[k].append(v)

if not eval_ndt_aq_violations:
    for k, v in rollout_ndt_aq_violations.items():
        eval_ndt_aq_violations[k] = [v]
else:
    for k, v in rollout_ndt_aq_violations.items():
        eval_ndt_aq_violations[k].append(v)

if not eval_heating_dt:
    for k, v in rollout_heating_dt.items():
        eval_heating_dt[k] = [v]
else:
    for k, v in rollout_heating_dt.items():
        eval_heating_dt[k].append(v)

if not eval_heating_beyond_comf_dt:
    for k, v in rollout_heating_beyond_comf_dt.items():
        eval_heating_beyond_comf_dt[k] = [v]
else:
    for k, v in rollout_heating_beyond_comf_dt.items():
        eval_heating_beyond_comf_dt[k].append(v)

if not eval_violation_dt:
    for k, v in rollout_violation_dt.items():
        eval_violation_dt[k] = [v]
else:
    for k, v in rollout_violation_dt.items():
        eval_violation_dt[k].append(v)

if not eval_violation_daq:
    for k, v in rollout_violation_daq.items():
        eval_violation_daq[k] = [v]
else:
    for k, v in rollout_violation_daq.items():
        eval_violation_daq[k].append(v)

eval_emissions_reward.append(rollout_emissions_reward)
eval_comfort_reward.append(rollout_comfort_reward)
eval_aq_reward.append(rollout_aq_reward)

eval_t_violations_means = {}
for k, v in eval_ndt_t_violations.items():
    eval_t_violations_means[k] = float(np.mean(v))

eval_aq_violations_means = {}
for k, v in eval_ndt_aq_violations.items():
    eval_aq_violations_means[k] = float(np.mean(v))

eval_heating_dt_means = {}
for k, v in eval_heating_dt.items():
    eval_heating_dt_means[k] = float(np.mean(v))

eval_heating_beyond_comf_dt_means = {}
for k, v in eval_heating_beyond_comf_dt.items():
    eval_heating_beyond_comf_dt_means[k] = float(np.mean(v))

eval_violation_dt_means = {}
for k, v in eval_violation_dt.items():
    eval_violation_dt_means[k] = float(np.mean(v))

eval_violation_daq_means = {}
for k, v in eval_violation_daq.items():
    eval_violation_daq_means[k] = float(np.mean(v))

# print(eval_emissions, float(np.mean(eval_emissions)))

metrics = {
    "eval/mean_episode_reward": float(np.mean(eval_rewards)),
    "eval/mean_episode_emissions_reward": float(np.mean(eval_emissions_reward)),
    "eval/mean_episode_comfort_reward": float(np.mean(eval_comfort_reward)),
    "eval/mean_episode_air_quality_reward": float(np.mean(eval_aq_reward)),
    "eval/mean_episode_emissions": float(np.mean(eval_emissions)),
    "eval/mean_episode_ndt_t_violations": eval_t_violations_means,
    "eval/mean_episode_ndt_aq_violations": eval_aq_violations_means,
    "eval/mean_episode_heating_degree_days": eval_heating_dt_means,
    "eval/mean_episode_heating_beyond_comfort_degree_days": (
        eval_heating_beyond_comf_dt_means
    ),
    "eval/mean_episode_violation_degree_days": eval_violation_dt_means,
    "eval/mean_episode_violation_ppm_days": eval_violation_daq_means,
}

with open(results_path + environment + "_results.json", "w", encoding="utf-8") as fp:
    json.dump(metrics, fp)
