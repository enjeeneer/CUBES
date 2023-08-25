"""script to run the cases with control for the Leiden paper"""
from cubes.construct.buildingconfig import load_building_config
from cubes.construct.building import Building
from cubes.package import envconfig
from cubes.construct.core import materials_evaluator, windows_evaluator
from cubes.package.core import register_environment
from cubes.cubesgym.utils.wrappers import LoggerWrapperCubes
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.vec_env import DummyVecEnv

import os
import gym
import sys

cwd_path = os.getcwd()
print(cwd_path)

materials = materials_evaluator()
windows = windows_evaluator()

print(type(sys.argv))

i_case = int(sys.argv[1])
model_path = sys.argv[2]

reps_per_year = 1
# years = np.arange(2017,2023)
years = [2017]
run_name = "evaulation_controlled"
print(i_case, model_path)

model = SAC.load(model_path)
# sys.exit()

for y in years:
    for r in range(reps_per_year):
        print("year: ", y)
        print("rep: ", r)
        environment = (
            "evaluate_SAC_case_"
            + str(i_case)
            + "_year_"
            + str(y)
            + "_rep_"
            + str(r)
            + "-v1"
        )

        complete_input_file_path = (
            "../01_evaluate_input/evaluation/case_"
            + str(i_case)
            + "/year_"
            + str(y)
            + "/rep_"
            + str(r)
            + "/input_c.json"
        )

        BC = load_building_config(complete_input_file_path)
        control_vent = True
        observe_vent = False
        control_observe_battery = False
        if i_case in [3, 4, 8, 9, 13, 14]:
            control_vent = False
            observe_vent = False
        if i_case >= 10:
            control_observe_battery = True

        EC = envconfig.EnvConfig(
            observe_zone_temperature=True,
            observe_electricity_demand=True,
            observe_outside_temperature=True,
            observe_zone_occupancy=True,
            observe_zone_co2=True,
            observe_grid_carbon_intensity=True,
            observe_zone_thermostat_setpoints=True,
            observe_zone_ventilation=True,
            observe_battery_charge=control_observe_battery,
            observe_batter_charging=control_observe_battery,
            observe_pv_power=control_observe_battery,
            control_battery_charging=control_observe_battery,
            control_ventilation=control_vent,
            control_thermostat_setpoints=True,
            observe_outside_temperature_in_x_hours_forecast=[1],
            observe_grid_carbon_in_x_hours_forecast=[],
            emissions_weight=0.5,
            air_quality_weight=0.15,
            episode_end_date=(15, 1),
            timesteps_per_hour=6,
        )
        building = Building(BC, materials, windows)
        building.build()
        idf = building.get_idf()

        if gym.envs.registry.env_specs[environment]:
            del gym.envs.registry.env_specs[environment]

        register_environment(environment, idf, BC, EC)

        env = gym.make(environment)
        env = LoggerWrapperCubes(env)
        # pylint: disable=cell-var-from-loop
        env_vec = DummyVecEnv([lambda: env])
        env_vec = VecFrameStack(env_vec, n_stack=4)

        # pylint: disable=protected-access
        n_timesteps_episode = (
            env.simulator._eplus_one_epi_len / env.simulator._eplus_run_stepsize
        )

        obs = env_vec.reset()
        done = False
        ts = 0
        while not done:
            ts += 1
            action, _states = model.predict(obs, deterministic=True)
            print(action)
            obs, rewards, done, info = env_vec.step(action)
            if ts % 1000 == 0:
                print("progress:", ts / n_timesteps_episode)

        env.close()
