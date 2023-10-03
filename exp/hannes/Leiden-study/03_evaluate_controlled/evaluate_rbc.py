"""script to run the cases with control for the Leiden paper"""
from cubes.construct.buildingconfig import load_building_config
from cubes.construct.building import Building
from cubes.construct.core import materials_evaluator, windows_evaluator
from cubes.package.core import register_environment
from cubes.cubesgym.utils.wrappers import LoggerWrapperCubes
from cubes.rbcs.rbc import AggressiveRBC
from cubes.package.utilities import get_envconfig_leiden

import os
import gym
import sys


cwd_path = os.getcwd()
print(cwd_path)

materials = materials_evaluator()
windows = windows_evaluator()

i_case = int(sys.argv[1])

reps_per_year = 1
# years = np.arange(2017,2023)
years = [2017]
run_name = "evaulation_controlled"

# sys.exit()
for y in years:
    for r in range(reps_per_year):
        print("year: ", y)
        print("rep: ", r)
        environment = (
            "evaluate_RBC_case_"
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
        EC = get_envconfig_leiden(i_case)
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
        rbc = AggressiveRBC(
            env.variables["action"],
            env.setpoints_space,
            env.variables["observation"],
            BC.heating_setpoint,
            BC.heating_setback,
            EC.air_quality_upper_limit,
        )

        print("action_space_real", env.setpoints_space.low[0])
        obs = env.reset()
        done = False
        ts = 0
        while not done:
            ts += 1
            # print(obs)
            action = rbc.act(obs)
            # print(action)
            obs, rewards, done, info = env.step(action)
            # print(info)
            # break

            if ts % 1000 == 0:
                print("progress:", ts / n_timesteps_episode)

        env.close()
