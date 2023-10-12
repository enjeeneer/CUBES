# pylint: disable=access-member-before-definition
"""Custom Callbacks for stable baselines 3 algorithms."""

import os
from typing import Optional, Union

import gym
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.env_util import is_wrapped
from stable_baselines3.common.vec_env import VecEnv, sync_envs_normalization

from cubes.cubesgym.utils.evaluation import evaluate_policy
from cubes.cubesgym.utils.wrappers import LoggerWrapperCubes
from sinergym.utils.wrappers import NormalizeObservation


class LoggerCallback(BaseCallback):
    """Custom callback for plotting additional values in tensorboard."""

    def __init__(self, sinergym_logger=False, verbose=0):
        """Custom callback for plotting additional values in tensorboard.
        Args:
            sinergym_logger (boolean): Indicate if CSVLogger inner Sinergym
            will be activated or not.
        """
        super().__init__(verbose)

        self.sinergym_logger = sinergym_logger

        self.ep_rewards = []
        self.ep_emissions = []
        self.ep_term_comfort = []
        self.ep_term_emissions = []
        self.ep_term_air_quality = []
        self.num_comfort_violation = 0
        self.ep_timesteps = 0

    def _on_training_start(self):
        # sinergym logger
        if is_wrapped(self.training_env, LoggerWrapperCubes):
            if self.sinergym_logger:
                self.training_env.env_method("activate_logger")
            else:
                self.training_env.env_method("deactivate_logger")

        # record method depending on the type of algorithm

        if "OnPolicyAlgorithm" in self.globals.keys():
            self.record = self.logger.record
        elif "OffPolicyAlgorithm" in self.globals.keys():
            self.record = self.logger.record_mean
        else:
            raise KeyError

    def _on_step(self) -> bool:
        info = self.locals["infos"][-1]

        # OBSERVATION
        variables = self.training_env.get_attr("variables")[0]["observation"]
        # log normalized and original values
        if self.training_env.env_is_wrapped(wrapper_class=NormalizeObservation)[0]:
            obs_normalized = self.locals["new_obs"][-1]
            obs = self.training_env.env_method("get_unwrapped_obs")[-1]
            for i, variable in enumerate(variables):
                self.record("normalized_observation/" + variable, obs_normalized[i])
                self.record("observation/" + variable, obs[i])
        # Only original values
        else:
            obs = self.locals["new_obs"][-1]
            for i, variable in enumerate(variables):
                self.record("observation/" + variable, obs[i])

        # ACTION
        variables = self.training_env.get_attr("variables")[0]["action"]
        action = None
        # sinergym action received inner its own setpoints range
        action_ = info["action_"]
        try:
            # network output clipped with gym action space
            action = self.locals["clipped_actions"][-1]
        except KeyError:
            try:
                action = self.locals["action"][-1]
            except KeyError:
                try:
                    action = self.locals["actions"][-1]
                except KeyError as exc:
                    raise KeyError(
                        "Algorithm action key in locals dict unknown."
                    ) from exc

        if self.training_env.get_attr("flag_discrete")[0]:
            action = self.training_env.get_attr("action_mapping")[0][action]
        for i, variable in enumerate(variables):
            if action is not None:
                self.record("action/" + variable, action[i])

            self.record("action_simulation/" + variable, action_[i])

        # Store episode data
        try:
            self.ep_rewards.append(self.locals["rewards"][-1])
        except KeyError:
            try:
                self.ep_rewards.append(self.locals["reward"][-1])
            except KeyError:
                print("Algorithm reward key in locals dict unknown")

        self.ep_emissions.append(info["emissions"])
        self.ep_term_comfort.append(info["reward_comfort"])
        self.ep_term_emissions.append(info["reward_emissions"])
        self.ep_term_air_quality.append(info["reward_air_quality"])
        if info["reward_comfort"] != 0:
            self.num_comfort_violation += 1
        self.ep_timesteps += 1

        # If episode ends, store summary of episode and reset
        try:
            done = self.locals["dones"][-1]
        except KeyError:
            try:
                done = self.locals["done"][-1]
            except KeyError:
                print("Algorithm done key in locals dict unknown")
        if done:
            # store last episode metrics
            self.episode_metrics = {}
            self.episode_metrics["ep_length"] = self.ep_timesteps
            self.episode_metrics["cumulative_reward"] = np.sum(self.ep_rewards)
            self.episode_metrics["mean_reward"] = np.mean(self.ep_rewards)
            self.episode_metrics["mean_emissions"] = np.mean(self.ep_emissions)
            self.episode_metrics["cumulative_power"] = np.sum(self.ep_emissions)
            self.episode_metrics["mean_comfort_penalty"] = np.mean(self.ep_term_comfort)
            self.episode_metrics["cumulative_comfort_penalty"] = np.sum(
                self.ep_term_comfort
            )
            self.episode_metrics["mean_emissions_penalty"] = np.mean(
                self.ep_term_emissions
            )
            self.episode_metrics["cumulative_power_penalty"] = np.sum(
                self.ep_term_emissions
            )
            self.episode_metrics["mean_air_quality_penalty"] = np.mean(
                self.ep_term_air_quality
            )
            self.episode_metrics["cumulative_air_quality_penalty"] = np.sum(
                self.ep_term_air_quality
            )
            try:
                self.episode_metrics["comfort_violation_time(%)"] = (
                    self.num_comfort_violation / self.ep_timesteps * 100
                )
            except ZeroDivisionError:
                self.episode_metrics["comfort_violation_time(%)"] = np.nan

            # reset episode info
            self.ep_rewards = []
            self.ep_emissions = []
            self.ep_term_comfort = []
            self.ep_term_emissions = []
            self.ep_term_air_quality = []
            self.ep_timesteps = 0
            self.num_comfort_violation = 0

        # During first episode, as it not finished, it shouldn't be recording
        if hasattr(self, "episode_metrics"):
            for key, metric in self.episode_metrics.items():
                self.logger.record("episode/" + key, metric)

        return True

    def on_training_end(self):
        if is_wrapped(self.training_env, LoggerWrapperCubes):
            self.training_env.env_method("activate_logger")


class LoggerEvalCallback(EvalCallback):
    """Callback for evaluating an agent.
    :param eval_env: The environment used for initialization
    :param callback_on_new_best:
        Callback to trigger when there is a new best model according to the mean reward
    :param n_eval_episodes: The number of episodes to test the agent
    :param eval_freq: Evaluate the agent every eval freq call of the callback.
    :param log_path:
        Path to a folder where the evaluations (evaluations.npz) will be saved.
        It will be updated at each evaluation.
    :param best_model_save_path: Path to a folder where the best model
        according to performance on the eval env will be saved.
    :param deterministic: Whether the evaluation should use a stochastic
        or deterministic actions.
    :param render: Whether to render or not the environment during evaluation
    :param verbose:
    :param warn: Passed to evaluate policy
        (warns if eval env has not been wrapped with a Monitor wrapper)
    """

    def __init__(
        self,
        eval_env: Union[gym.Env, VecEnv],
        callback_on_new_best: Optional[BaseCallback] = None,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        log_path: Optional[str] = None,
        best_model_save_path: Optional[str] = None,
        deterministic: bool = True,
        render: bool = False,
        verbose: int = 1,
        warn: bool = True,
    ):
        super().__init__(
            eval_env=eval_env,
            callback_on_new_best=callback_on_new_best,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            log_path=log_path,
            best_model_save_path=best_model_save_path,
            deterministic=deterministic,
            render=render,
            verbose=verbose,
            warn=warn,
        )

        self.evaluations_emissions = []
        self.evaluations_comfort_violation = []
        self.evaluations_comfort_penalty = []
        self.evaluations_emissions_penalty = []
        self.evaluations_air_quality_penalty = []
        self.evaluation_metrics = {}

    def _on_step(self) -> bool:

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Sync training and eval env if there is VecNormalize
            sync_envs_normalization(self.training_env, self.eval_env)

            # Reset success rate buffer
            self._is_success_buffer = []

            episodes_data = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                callback=None,
            )

            if self.log_path is not None:
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episodes_data["episodes_rewards"])
                self.evaluations_length.append(episodes_data["episodes_lengths"])
                self.evaluations_emissions.append(episodes_data["episodes_emissions"])
                self.evaluations_comfort_violation.append(
                    episodes_data["episodes_comfort_violations"]
                )
                self.evaluations_comfort_penalty.append(
                    episodes_data["episodes_comfort_penalties"]
                )
                self.evaluations_emissions_penalty.append(
                    episodes_data["episodes_emissions_penalties"]
                )
                self.evaluations_air_quality_penalty.append(
                    episodes_data["episodes_air_quality_penalties"]
                )

                kwargs = {}
                # Save success log if present
                if len(self._is_success_buffer) > 0:
                    self.evaluations_successes.append(self._is_success_buffer)
                    kwargs = {"successes": self.evaluations_successes}

                np.savez(
                    self.log_path,
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                    ep_lengths=self.evaluations_length,
                    ep_emissions=self.evaluations_emissions,
                    ep_comfort_violations=self.evaluations_comfort_violation,
                    episodes_comfort_penalties=self.evaluations_comfort_penalty,
                    episodes_emissions_penalties=self.evaluations_emissions_penalty,
                    episodes_air_quality_penalties=self.evaluations_air_quality_penalty,
                    **kwargs,
                )

            mean_reward, std_reward = np.mean(
                episodes_data["episodes_rewards"]
            ), np.std(episodes_data["episodes_rewards"])
            mean_ep_length, std_ep_length = np.mean(
                episodes_data["episodes_lengths"]
            ), np.std(episodes_data["episodes_lengths"])

            self.evaluation_metrics["mean_rewards"] = mean_reward
            self.evaluation_metrics["std_rewards"] = std_reward
            self.evaluation_metrics["mean_ep_length"] = mean_ep_length
            self.evaluation_metrics["mean_emissions"] = np.mean(
                episodes_data["episodes_emissions"]
            )
            self.evaluation_metrics["comfort_violation(%)"] = np.mean(
                episodes_data["episodes_comfort_violations"]
            )
            self.evaluation_metrics["comfort_penalty"] = np.mean(
                episodes_data["episodes_comfort_penalties"]
            )
            self.evaluation_metrics["emissions_penalty"] = np.mean(
                episodes_data["episodes_emissions_penalties"]
            )
            self.evaluation_metrics["air_quality_penalty"] = np.mean(
                episodes_data["episodes_air_quality_penalties"]
            )

            if self.verbose > 0:
                print(
                    f"Eval num_timesteps={self.num_timesteps}, "
                    f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}"
                )
                print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
            # Add to current Logger
            for key, metric in self.evaluation_metrics.items():
                self.logger.record("eval/" + key, metric)

            if len(self._is_success_buffer) > 0:
                success_rate = np.mean(self._is_success_buffer)
                if self.verbose > 0:
                    print(f"Success rate: {100 * success_rate:.2f}%")
                self.logger.record("eval/success_rate", success_rate)

            if mean_reward > self.best_mean_reward:
                if self.verbose > 0:
                    print("New best mean reward!")
                if self.best_model_save_path is not None:
                    self.model.save(
                        os.path.join(self.best_model_save_path, "model.zip")
                    )
                self.best_mean_reward = mean_reward
                # Trigger callback if needed
                if self.callback is not None:
                    return self._on_event()

        return True
