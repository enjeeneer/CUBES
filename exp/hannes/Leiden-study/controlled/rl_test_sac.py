"""script to run the base cases for the Leiden paper"""
from cubes.construct.buildingconfig import load_building_config
from cubes.construct.building import Building
from cubes.package import envconfig
from cubes.construct.core import materials_evaluator, windows_evaluator
from cubes.package.core import register_environment


import os
import gym


from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import vec_monitor, VecFrameStack

from datetime import datetime

from stable_baselines3.common.vec_env import DummyVecEnv

# from stable_baselines3.common.noise import NormalActionNoise,VectorizedActionNoise

# from input_file_factory import get_input_file_with_schedules_etc

cwd_path = os.getcwd()
print(cwd_path)

materials = materials_evaluator()
windows = windows_evaluator()

cases = [0]
run_name = "sac-v15"
episodes = 50
variations = 6
one_per_year = True
years = [2017, 2018, 2019, 2020, 2021, 2022]


def make_env(env_id_base, idx):
    def _init():
        env_id = env_id_base + str(idx)
        env = gym.make(env_id)
        env.reset()
        return env

    return _init


for i_case in cases:
    environment = "test_env_vecenv_framestack-" + str(i_case) + "-v"

    model_save_path = "saved_models/case" + str(i_case)

    input_file_path = "../base_input/case" + str(i_case) + ".json"
    case_path = cwd_path + "/" + run_name + "/case_" + str(i_case)
    complete_input_file_path = case_path + "/input.json"
    if not os.path.exists(case_path):
        os.makedirs(case_path)

    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    # register environments:
    for v in range(variations):
        sub_environment = environment + str(v)
        # complete_input_file_path = case_path + "/input"+"-v"+str(v)+".json"

        # if one_per_year:
        #     get_input_file_with_schedules_etc(input_file_path,
        #           complete_input_file_path,True,years[v])
        # else:
        #     get_input_file_with_schedules_etc(input_file_path,
        #           complete_input_file_path,True)

        # #test single environment

        r = 0

        complete_input_file_path = (
            "../01_evaluate_input/evaluation/case_"
            + str(i_case)
            + "/year_"
            + str(years[v])
            + "/rep_"
            + str(r)
            + "/input_c.json"
        )
        bc = load_building_config(complete_input_file_path)
        # bc = load_building_config("input_new.json")
        control_vent = True
        observe_vent = False
        control_observe_battery = False
        observe_gcf = [1]
        observe_gci = True
        if i_case in [3, 4, 8, 9, 13, 14]:
            control_vent = False
            observe_vent = False
        if i_case >= 10:
            control_observe_battery = True
        if i_case < 5:
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
        building = Building(bc, materials, windows)
        building.build()
        idf = building.get_idf()

        if gym.envs.registry.env_specs[sub_environment]:
            del gym.envs.registry.env_specs[sub_environment]

        register_environment(sub_environment, idf, bc, ec)

    train_env = DummyVecEnv([make_env(environment, i) for i in range(variations)])
    # Frame-stacking with 4 frames
    train_env = VecFrameStack(train_env, n_stack=4)

    train_env = vec_monitor.VecMonitor(
        train_env, filename="case_" + str(i_case) + "_training.log"
    )
    train_env.reset()
    experiment_date = datetime.today().strftime("%Y-%m-%d %H:%M")

    # register run name
    name = f"SAC-{environment}-episodes_{episodes}({experiment_date})"

    n_timesteps_episode = 52560

    timesteps = episodes * n_timesteps_episode
    # timesteps = 1 * n_timesteps_episode

    # # Add some action noise for exploration
    # n_actions = train_env.get_attr("action_space",0)[0].shape[-1]
    # #action_noise = NormalActionNoise(mean=np.zeros(n_actions),
    # sigma=0.1 * np.ones(n_actions))
    # action_noise = LinearNormalActionNoise(mean=np.zeros(n_actions),
    # sigma=0.5 * np.ones(n_actions),max_steps=timesteps)
    # vec_action_noise = VectorizedActionNoise(action_noise,6)

    # env = gym.make(environment)
    # env = LoggerWrapperCubes(env)

    model = SAC(
        "MlpPolicy",
        train_env,
        verbose=1,
        tensorboard_log="./" + environment + "_tensorboard/",
        # action_noise=vec_action_noise
        # batch_size=256,
        # learning_starts=10000,
        # gradient_steps=64,
        # policy_kwargs=dict(net_arch=[64, 64])
        # learning_rate=lin_7.3e-4
        # buffer_size=100000,
        # train_freq=8,
        # ent_coef=0.05
    )
    # model = SAC.load("/workspaces/elizabeth-homes/exp/hannes/Leiden-study/""
    # "controlled/saved_models/case0/SAC-test_env_vecenv_framestack-0-v-"
    # "episodes_5(2023-08-21 16:05).zip",env=train_env)

    model.learn(total_timesteps=timesteps, log_interval=1)

    model.save(model_save_path + "/" + name)

    train_env.close()
