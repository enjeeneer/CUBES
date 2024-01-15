# pylint: skip-file

"""Module defining utility functions for agents."""

import torch
import random
import math
import wandb
import re
import numpy as np
from typing import Union
from loguru import logger
from os import makedirs

from cubes.constants import BASE_DIR


class TanhTransform(torch.distributions.transforms.Transform):
    """Implementation of the Tanh transformation."""

    domain = torch.distributions.constraints.real
    codomain = torch.distributions.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        # We do not clamp to the boundary here as it may
        # degrade the performance of certain algorithms.
        # one should use `cache_size=1` instead
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):

        return 2.0 * (math.log(2.0) - x - torch.nn.functional.softplus(-2.0 * x))


class SquashedNormal(
    torch.distributions.transformed_distribution.TransformedDistribution
):
    """Implementation of the Squashed Normal distribution."""

    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

        self.base_dist = torch.distributions.Normal(loc, scale)
        transforms = [TanhTransform()]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu


class TruncatedNormal(torch.distributions.Normal):
    """Implementation of the Truncated Normal distribution."""

    def __init__(self, loc, scale, low=-1.0, high=1.0, eps=1e-6) -> None:
        super().__init__(loc, scale, validate_args=False)
        self.low = low
        self.high = high
        self.eps = eps

    def _clamp(self, x) -> torch.Tensor:
        clamped_x = torch.clamp(x, self.low + self.eps, self.high - self.eps)
        x = x - x.detach() + clamped_x.detach()
        return x

    def sample(  # pylint: disable=W0237
        self, clip=None, sample_shape=torch.Size()
    ) -> torch.Tensor:
        shape = self._extended_shape(sample_shape)
        eps = torch.distributions.utils._standard_normal(  # pylint: disable=W0212
            shape, dtype=self.loc.dtype, device=self.loc.device
        )
        eps *= self.scale
        if clip is not None:
            eps = torch.clamp(eps, -clip, clip)
        x = self.loc + eps
        return self._clamp(x)


def schedule(schdl, step) -> float:
    try:
        return float(schdl)
    except ValueError:
        match = re.match(r"linear\((.+),(.+),(.+)\)", schdl)
        if match:
            init, final, duration = [float(g) for g in match.groups()]
            mix = np.clip(step / duration, 0.0, 1.0)
            return (1.0 - mix) * init + mix * final
        match = re.match(r"step_linear\((.+),(.+),(.+),(.+),(.+)\)", schdl)
        if match:
            init, final1, duration1, final2, duration2 = [
                float(g) for g in match.groups()
            ]
            if step <= duration1:
                mix = np.clip(step / duration1, 0.0, 1.0)
                return (1.0 - mix) * init + mix * final1
            else:
                mix = np.clip((step - duration1) / duration2, 0.0, 1.0)
                return (1.0 - mix) * final1 + mix * final2


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def reparameterise(x, clamp=("hard", -5, 2)):
    """
    The reparameterisation trick.
    Construct a Gaussian from x, taken to parameterise
    the mean and log standard deviation.
    """
    mean, log_std = torch.split(x, int(x.shape[-1] / 2), dim=-1)

    if clamp[0] == "hard":  # This is used by default for the SAC policy.
        log_std = torch.clamp(log_std, clamp[1], clamp[2])
    elif clamp[0] == "soft":  # This is used by default for the PETS model.
        log_std = clamp[2] - torch.nn.functional.softplus(clamp[2] - log_std)
        log_std = clamp[1] + torch.nn.functional.softplus(log_std - clamp[1])

    dist = torch.distributions.Normal(mean, torch.exp(log_std))

    return mean, log_std, dist


def squashed_gaussian(x, sample=True):
    """
    For continuous spaces. Interpret pi as the mean
    and log standard deviation of a Gaussian,
    then generate an action by sampling from that
    distribution and applying tanh squashing.
    """
    _, _, gaussian = reparameterise(x)
    if sample:
        action_unsquashed = (
            gaussian.rsample()
        )  # rsample() required to allow differentiation.
    else:
        action_unsquashed = gaussian.mean
    action = torch.tanh(action_unsquashed)
    # Compute log_prob from Gaussian, then apply correction for tanh squashing.
    log_prob = gaussian.log_prob(action_unsquashed).sum(axis=-1)
    log_prob -= (
        2
        * (
            np.log(2)
            - action_unsquashed
            - torch.nn.functional.softplus(-2 * action_unsquashed)
        )
    ).sum(axis=-1)
    return action, log_prob.unsqueeze(-1), gaussian


def pull_model_from_wandb(
    algorithm: str,
    wandb_entity: str,
    wandb_project_id: str,
    wandb_run_id: str,
    wandb_model_id: str,
    observation_length: int,
    action_length: int,
    config: dict,
):
    """
    Downloads a model from a wandb run and hands weights over to newly
    initialized model. This main use case is for loading models onto
    local CPU that were trained on cloud GPU.
    Args:
        algorithm: algo name
        wandb_run_id: wandb run id
        wandb_project_id: wandb project name
        wandb_model_id: name of saved model on wandb run
        observation_length: env obs length
        action_length: env action length
        config: dict for setting up handshake model
    Returns:
        handshaked model: CFB, FB, or CQL agent with weights from wandb run
    """

    # get model from wandb
    logger.info(f"Loading model from wandb run: {wandb_run_id}")
    api = wandb.Api(overrides={"entity": wandb_entity, "project": wandb_project_id})
    save_dir = BASE_DIR / "agents" / f"{algorithm}" / "saved_models" / wandb_run_id
    makedirs(str(save_dir), exist_ok=True)
    save_path = save_dir / f"{wandb_model_id}"

    # check if model already exists
    if save_path.exists():
        logger.info(f"Model already exists at {save_path}.")

    else:
        run = api.from_path(f"{wandb_project_id}/runs/{wandb_run_id}")
        run.file(wandb_model_id).download(root=save_dir.as_posix(), replace=True)

    # load model
    trained_agent = torch.load(save_path, map_location=torch.device("cpu"))

    if algorithm == "sac":

        from agents.sac.agent import SoftActorCritic

        handshake_agent = SoftActorCritic(
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
            action_range=[np.array(-1), np.array(1)],
            history_length=config["history_length"],
            normalisation_samples=config["normalisation_samples"],
        )

        handshake_agent.critic.load_state_dict(trained_agent.critic.state_dict())
        handshake_agent.critic_target.load_state_dict(
            trained_agent.critic_target.state_dict()
        )
        handshake_agent.actor.load_state_dict(trained_agent.actor.state_dict())

        if handshake_agent._normalise == True:
            handshake_agent.running_mean_numpy = trained_agent.running_mean_numpy
            handshake_agent.running_std_numpy = trained_agent.running_std_numpy

    elif algorithm == "dt":

        from agents.dt.agent import DecisionTransformer

        handshake_agent = DecisionTransformer(
            discretisation_bins=config["discretisation_bins"],
            number_of_blocks=config["number_of_blocks"],
            number_of_heads=config["number_of_heads"],
            embedding_dimension=config["embedding_dimension"],
            dropout=config["dropout"],
            feedforward_hidden_dimension=config["feedforward_hidden_dimension"],
            layer_norm_epsilon=config["layer_norm_epsilon"],
            tokenizer_mu=config["tokenizer_mu"],
            positional_encoder_table_dimension=config[
                "positional_encoder_table_dimension"
            ],
            betas=config["betas"],
            learning_rate=config["learning_rate"],
            weight_decay=config["weight_decay"],
            gradient_norm_clip=config["gradient_norm_clip"],
            optimiser_epsilon=config["optimiser_epsilon"],
            device=config["device"],
        )
        handshake_agent.model.load_state_dict(trained_agent.model.state_dict())

    else:
        raise ValueError("Unknown agent name.")

    handshake_agent.eval()

    return handshake_agent
