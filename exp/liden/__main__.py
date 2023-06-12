# pylint: disable=protected-access

"""Evaluates the performance of pre-trained agents."""
import yaml
import torch
import datetime
import gym
from cubes.package.core import make_test_env
from cubes.constants import BASE_DIR

from agents.sac.agent import SoftActorCritic
from agents.workspaces import SACWorkspace
from agents.utils import set_seed_everywhere

config_path = BASE_DIR / "agents" / "sac" / "config.yaml"
model_dir = BASE_DIR / "agents" / "sac" / "saved_models"
time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

with open(config_path, "rb") as f:
    config = yaml.safe_load(f)

set_seed_everywhere(config["seed"])
config["device"] = torch.device(
    "cuda" if torch.cuda.is_available() else ("mps" if torch.has_mps else "cpu")
)

env_id = make_test_env()
print(env_id)
env = gym.make(env_id)

observation_length = env.observation_space.shape[0]
action_length = env.action_space.shape[0]

action_range = [
    env.action_space.low[0],
    env.action_space.high[0],
]

agent = SoftActorCritic(
    observation_length=observation_length,
    action_length=action_length,
    device=config["device"],
    name=config["name"],
    buffer_capacity=config["learning_steps"],
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

workspace = SACWorkspace(
    env=env,
    eval_frequency=config["eval_frequency"],
    eval_rollouts=config["eval_rollouts"],
    model_dir=model_dir,
    seed_steps=config["seed_steps"],
    learning_steps=config["learning_steps"],
)

if __name__ == "__main__":
    workspace.train(agent, agent_config=config)
