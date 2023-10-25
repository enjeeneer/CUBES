# pylint: disable=protected-access
# pylint: disable=ungrouped-imports

"""Evaluates the performance of pre-trained agents."""
import yaml
import torch
import datetime
import gym
from argparse import ArgumentParser

from agents.dt.agent import DecisionTransformer
from agents.dt.replay_buffer import DecisionTransformerReplayBuffer
from agents.workspaces import DecisionTransformerWorkspace
from agents.utils import set_seed_everywhere, pull_model_from_wandb

from utils import BASE_DIR
from cubes.package.core import register_environment
from cubes.construct.buildingconfig import load_building_config
from cubes.construct.building import Building
from cubes.construct.core import materials_evaluator, windows_evaluator
from cubes.package.utilities import get_envconfig_leiden
from cubes.cubesgym.utils.wrappers import LoggerWrapperCubes, DatetimeWrapperCubes

parser = ArgumentParser()
parser.add_argument("--eval_case", type=int)
parser.add_argument("--eval_year", type=int)
parser.add_argument("--dataset_name", type=str)
parser.add_argument("--wandb_logging", type=str, default="True")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--learning_steps", type=int, default=1e6)
parser.add_argument("--eval_frequency", type=int, default=2e4)
parser.add_argument("--eval_rollouts", type=int, default=5)
parser.add_argument("--context_length", type=int, default=128)
parser.add_argument("--load_agent", type=str, default="False")
parser.add_argument("--wandb_run_id", type=str)
parser.add_argument("--wandb_model_id", type=str)
args = parser.parse_args()

config_path = BASE_DIR / "agents" / "dt" / "config.yaml"
model_dir = BASE_DIR / "agents" / "dt" / "saved_models"
dataset_path = (
    BASE_DIR / "train" / "datasets" / "processed" / args.dataset_name / "dataset.npz"
)
time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

with open(config_path, "rb") as f:
    config = yaml.safe_load(f)

config["device"] = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_built() else "cpu")
)
config.update(vars(args))
set_seed_everywhere(config["seed"])

if args.wandb_logging == "True":
    args.wandb_logging = True
else:
    args.wandb_logging = False

if args.load_agent == "False":
    load_agent = False
    test_save_path = ""
else:
    load_agent = True

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
ec = get_envconfig_leiden(config["eval_case"])
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
    + str(config["eval_year"])
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

if load_agent:
    agent = pull_model_from_wandb(
        algorithm="dt",
        wandb_run_id=args.wandb_run_id,
        wandb_model_id=args.wandb_model_id,
        observation_length=observation_length,
        action_length=action_length,
        config=config,
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
        positional_encoder_table_dimension=config["positional_encoder_table_dimension"],
        betas=config["betas"],
        learning_rate=float(config["learning_rate"]),
        weight_decay=config["weight_decay"],
        gradient_norm_clip=config["gradient_norm_clip"],
        optimiser_epsilon=float(config["optimiser_epsilon"]),
        device=config["device"],
    )

replay_buffer = DecisionTransformerReplayBuffer(
    device=config["device"], dataset_path=dataset_path
)

workspace = DecisionTransformerWorkspace(
    learning_steps=config["learning_steps"],
    eval_frequency=config["eval_frequency"],
    eval_rollouts=config["eval_rollouts"],
    wandb_logging=config["wandb_logging"],
    device=config["device"],
    model_dir=model_dir,
    eval_env=env,
    observation_dim=observation_length,
    action_dim=action_length,
    context_length=config["context_length"],
    agent_config=config,
)

if __name__ == "__main__":
    if load_agent:
        workspace.eval(agent)
    else:
        workspace.train(agent, replay_buffer)
