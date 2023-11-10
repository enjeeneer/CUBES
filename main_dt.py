# pylint: disable=protected-access
# pylint: disable=ungrouped-imports

"""Evaluates the performance of pre-trained agents."""
import yaml
import torch
from os import makedirs
import gym
from argparse import ArgumentParser

from agents.dt.agent import DecisionTransformer
from agents.dt.replay_buffer import DecisionTransformerReplayBuffer
from agents.workspaces import DecisionTransformerWorkspace
from agents.utils import set_seed_everywhere, pull_model_from_wandb

from utils import BASE_DIR

parser = ArgumentParser()
parser.add_argument("--dataset_name", type=str)
parser.add_argument("--wandb_logging", type=str, default="True")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--learning_steps", type=int, default=100000)
parser.add_argument("--evaluate_agent", type=str, default="False")
parser.add_argument("--eval_case", type=int, default=None)
parser.add_argument("--eval_year", type=int, default=None)
parser.add_argument("--eval_rollouts", type=int, default=1)
parser.add_argument("--predict_rewards", type=str, default="False")
parser.add_argument("--save_frequency", type=int, default=10000)
parser.add_argument("--wandb_run_id", type=str)
parser.add_argument("--wandb_model_id", type=str)
args = parser.parse_args()

config_path = BASE_DIR / "agents" / "dt" / "config.yaml"
model_dir = BASE_DIR / "agents" / "dt" / "saved_models"
dataset_path = (
    BASE_DIR / "train" / "processed_datasets" / args.dataset_name / "dataset.npz"
)
if args.wandb_logging == "True":
    args.wandb_logging = True
else:
    args.wandb_logging = False

if args.evaluate_agent == "False":
    evaluate_agent = False
    test_save_path = ""
else:
    evaluate_agent = True
    assert args.eval_case is not None
    assert args.eval_year is not None
    assert args.wandb_run_id is not None
    assert args.wandb_model_id is not None

if args.predict_rewards == "False":
    args.predict_rewards = False
else:
    args.predict_rewards = True

with open(config_path, "rb") as f:
    config = yaml.safe_load(f)

config.update(vars(args))
set_seed_everywhere(config["seed"])
config["device"] = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)

replay_buffer = DecisionTransformerReplayBuffer(
    device=config["device"],
    dataset_path=dataset_path,
    rewards=config["predict_rewards"],
)

if evaluate_agent:
    from cubes.package.core import register_environment
    from cubes.construct.buildingconfig import load_building_config
    from cubes.construct.building import Building
    from cubes.construct.core import materials_evaluator, windows_evaluator
    from cubes.package.utilities import get_envconfig_leiden
    from cubes.cubesgym.utils.wrappers import LoggerWrapperCubes, DatetimeWrapperCubes

    agent = pull_model_from_wandb(
        algorithm="dt",
        wandb_run_id=args.wandb_run_id,
        wandb_model_id=args.wandb_model_id,
        observation_length=None,
        action_length=None,
        config=config,
    )

    # register environments:
    environment = (
        "Leiden-case_"
        + str(config["eval_case"])
        + "-year_"
        + str(config["eval_year"])
        + "-rep_"
        + str(0)
        + "-seed_"
        + str(config["seed"])
    )
    files_dir = str(BASE_DIR / "inputs" / environment)
    makedirs(files_dir, exist_ok=True)

    # register environments:
    eval_config = (
        "eval/configs"
        + "/case_"
        + str(config["eval_case"])
        + "/year_"
        + str(config["eval_year"])
        + "/input_c.json"
    )
    bc = load_building_config(eval_config)
    ec = get_envconfig_leiden(config["eval_case"], files_dir=files_dir)
    ec.map_t_setpoints_to_comfort_space = True

    building = Building(bc, materials_evaluator(), windows_evaluator())
    building.build()
    idf = building.get_idf()

    environment = (
        "Leiden-case_"
        + str(config["eval_case"])
        + "-year_"
        + str(config["eval_year"])
        + "-rep_"
        + str(0)
    )

    register_environment(environment, idf, bc, ec)
    env = gym.make(environment)
    env = LoggerWrapperCubes(env)
    env = DatetimeWrapperCubes(env)

    observation_length = env.observation_space.shape[0]
    action_length = env.action_space.shape[0]

    action_range = [
        env.action_space.low[0],
        env.action_space.high[0],
    ]

    workspace = DecisionTransformerWorkspace(
        learning_steps=config["learning_steps"],
        eval_frequency=config["eval_frequency"],
        eval_rollouts=config["eval_rollouts"],
        wandb_logging=args.wandb_logging,
        device=config["device"],
        model_dir=model_dir,
        eval_env=env,
        observation_dim=observation_length,
        action_dim=action_length,
        context_length=replay_buffer.context_length,
        agent_config=config,
    )

else:
    agent = DecisionTransformer(
        discretisation_bins=config["discretisation_bins"],
        number_of_blocks=config["number_of_blocks"],
        number_of_heads=config["number_of_heads"],
        embedding_dimension=config["embedding_dimension"],
        dropout=config["dropout"],
        feedforward_hidden_dimension=config["feedforward_hidden_dimension"],
        layer_norm_epsilon=float(config["layer_norm_epsilon"]),
        tokenizer_mu=config["tokenizer_mu"],
        tokenizer_M=config["tokenizer_M"],
        positional_encoder_table_dimension=config["positional_encoder_table_dimension"],
        betas=config["betas"],
        learning_rate=float(config["learning_rate"]),
        weight_decay=config["weight_decay"],
        gradient_norm_clip=config["gradient_norm_clip"],
        optimiser_epsilon=float(config["optimiser_epsilon"]),
        device=config["device"],
        batch_size=config["batch_size"],
    )

    workspace = DecisionTransformerWorkspace(
        learning_steps=config["learning_steps"],
        wandb_logging=args.wandb_logging,
        device=config["device"],
        model_dir=model_dir,
        save_frequency=config["save_frequency"],
        agent_config=config,
    )

if __name__ == "__main__":
    if evaluate_agent:
        workspace.eval(agent)
    else:
        workspace.train(agent, replay_buffer)
