"""script to run the cases with control for the Leiden paper"""
from cubes.construct.buildingconfig import load_building_config
from cubes.construct.building import Building
from cubes.construct.core import materials_evaluator, windows_evaluator
from cubes.package.core import register_environment
from cubes.cubesgym.utils.wrappers import LoggerWrapperCubes
from cubes.rbcs.rbc import TrivialRBC, AggressiveRBC
from cubes.package.utilities import get_envconfig_leiden

import os
import gym
import sys
import numpy as np
import json


cwd_path = os.getcwd()
print(cwd_path)

materials = materials_evaluator()
windows = windows_evaluator()

i_case = int(sys.argv[1])
rbc_switch = int(sys.argv[2])

if rbc_switch == 0:
    rbc_name = "trivial"
else:
    rbc_name = "aggressive"


reps_per_year = 1
# years = np.arange(2017,2023)
years = [2022]


# sys.exit()
for y in years:
    for r in range(reps_per_year):
        print("year: ", y)
        print("rep: ", r)
        environment = (
            "evaluate_RBC_"
            + rbc_name
            + "_case_"
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
        if rbc_switch == 0:
            rbc = TrivialRBC(
                env.variables["action"],
                env.setpoints_space,
                env.variables["observation"],
                BC.heating_setpoint,
                EC.air_quality_range[1],
            )
        else:
            rbc = AggressiveRBC(
                env.variables["action"],
                env.setpoints_space,
                env.variables["observation"],
                BC.heating_setpoint,
                BC.heating_setback,
                EC.air_quality_range[1],
            )

            # obs = env.reset()
        # done = False
        # ts = 0
        # while not done:
        #     ts += 1
        #     # print(obs)
        #     action = rbc.act(obs)
        #     # print(action)
        #     obs, rewards, done, info = env.step(action)
        #     # print(info)
        #     # break

        #     if ts % 1000 == 0:
        #         print("progress:", ts / n_timesteps_episode)

        # env.close()

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
        ts = 0
        while not done:
            ts += 1
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

            if ts % 1000 == 0:
                print(f"progress: {ts / n_timesteps_episode*100:.2f}%")

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

        with open(environment + "_results.json", "w", encoding="utf-8") as fp:
            json.dump(metrics, fp)
