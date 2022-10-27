import bauwerk
import numpy as np
from tqdm import tqdm
from utils.utils import ObsWrapper


class Helper:
    def __init__(self, cfg, tokenizer, env, obs_dim):
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.env = env
        self.obs_dim = obs_dim
        self.act_dim = int(env.action_space.shape[0])
        self.prompt_steps = int(
            np.ceil(self.cfg.transformer.context_length / (self.obs_dim + self.act_dim)))  # +1 for 1-d action

    def evaluate_actions(self, actions, env):
        cum_reward = 0
        env
        obs = env.reset()
        obses = []
        for i, action in enumerate(actions):
            obs, reward, done, info = env.step(np.array(action, dtype=np.float32))
            obses.append(obs)
            if i >= self.prompt_steps:
                cum_reward += reward

        return cum_reward / len(actions)

    def get_bauwerk_prompt(self, prompt_steps):
        """
        Creates task-specifc tokenized prompt for DT.
        :param prompt_steps: no. of steps that this method must create
        :return tokenized prompt: array of state-action tokens, shape [context_length,]
        :return obs_mask: array of state-action tokens, shape [context_length,]
        :return action_mask: array of state-action tokens, shape [context_length,]
        """
        print('...creating prompt...')
        optimal_actions = bauwerk.solve(self.env)[0]
        state_actions = []

        # create masks
        obs_mask = np.zeros(shape=(prompt_steps + 1, self.obs_dim + self.act_dim))  # +1 because we include final additional obs
        act_mask = np.zeros(shape=(prompt_steps, self.obs_dim + self.act_dim))
        obs_mask[:, :self.obs_dim] = np.arange(start=1, stop=self.obs_dim+1)
        act_mask[:, self.obs_dim: self.obs_dim + self.act_dim] = 1
        obs_mask = obs_mask.flatten()[-self.cfg.transformer.context_length:]
        act_mask = act_mask.flatten()[-self.cfg.transformer.context_length:]

        obs = self.env.reset()
        for step in range(prompt_steps):
            state_actions.append(obs)
            action = optimal_actions[step]
            obs, _, _, _ = self.env.step(action)
            state_actions.append(action)

        state_actions.append(obs)

        # correct masks for last obs
        obs_mask[:-self.obs_dim] = obs_mask[self.obs_dim:]
        obs_mask[-self.obs_dim:] = np.arange(start=1, stop=self.obs_dim+1)
        act_mask[:-self.obs_dim] = act_mask[self.obs_dim:]
        act_mask[-self.obs_dim:] = 0

        prompt = np.concatenate(np.array(state_actions, dtype=object))[-self.cfg.transformer.context_length:]  # flattened array sliced to context length
        tokenised_prompt = self.tokenizer.tokenize(prompt)

        return tokenised_prompt, obs_mask, act_mask

    def rollout_with_prompt(self, model, eval_steps):
        """
        Performs rollout with prompt
        :param self.env:
        :param eval_steps:
        :return:
        """
        model_actions = []
        optimal_actions = bauwerk.solve(self.env)[0]
        rollout_reward = 0


        ### we have to create two ugly loops to give DT the self.env after prompt steps ###

        # loop 1: setting up self.env
        print('...preparing self.env...')
        obs = self.env.reset()
        for i in range(self.prompt_steps):
            action = optimal_actions[i]
            obs, _, _, _ = self.env.step(action)

        # create prompt sequence
        tokens, obs_mask, act_mask = self.get_bauwerk_prompt(self.prompt_steps)

        # loop 2: eval rollout
        print('...collecting rollout...')
        for _ in tqdm(range(eval_steps)):
            action_dims = []
            # we predict action dimensions one-by-one
            for _ in range(self.act_dim):
                out_seq = model.predict_sequence(
                    input_sequence=np.expand_dims(tokens, axis=0),
                    obs_mask=np.expand_dims(obs_mask, axis=0),
                    act_mask=np.expand_dims(act_mask, axis=0)
                )
                action_dims.append(out_seq[:, -1].detach().numpy())  # action dim is final dim of predicted sequence
                tokens, obs_mask, act_mask = self.tokenizer.update_sequences(tokens, obs_mask, act_mask, out_seq[:, -1],
                                                                        action=True)

            action = np.array(action_dims, dtype=np.float32).flatten()
            obs, reward, _, _ = self.env.step(action)

            obs_tokens = self.tokenizer.tokenize(obs)
            tokens, obs_mask, act_mask = self.tokenizer.update_sequences(tokens, obs_mask, act_mask, obs_tokens, obs=True)

            model_actions.append(action)
            rollout_reward += reward

        mean_reward = rollout_reward / len(model_actions)

        return mean_reward, model_actions

    def test_across_battery_sizes(self, model):
        """takes an agent at tests performance across the range of bauwerk tasks."""
        # setup
        build_dist_b = bauwerk.benchmarks.BuildDistB(seed=0)
        tasks = build_dist_b.train_tasks

        # logging
        optimal_rewards = []
        random_rewards = []
        no_charge_rewards = []
        dt_rewards = []

        battery_sizes = np.arange(1, 21, 1)
        for size in battery_sizes:
            # set task
            env = build_dist_b.make_env()
            task = bauwerk.benchmarks.Task(
                env_name=str(size),
                cfg=bauwerk.envs.solar_battery_house.EnvConfig(
                    battery_size=size,
                    episode_len=24 * 30 + self.prompt_steps,
                )
            )
            env.set_task(task)
            env = ObsWrapper(env)

            # DT rollout
            dt_reward, model_actions = self.rollout_with_prompt(model=model, env=env, eval_steps=int(24 * 30))
            dt_rewards.append(dt_reward)

            # optimal rollout
            optimal_actions = bauwerk.solve(env)
            optimal_reward = self.evaluate_actions(optimal_actions[0][:24 * 30 + self.prompt_steps], env)
            optimal_rewards.append(optimal_reward)

            # random rollout
            random_actions = [env.action_space.sample() for _ in range(24 * 30)]
            random_reward = self.evaluate_actions(random_actions[:24 * 30 + self.prompt_steps], env)
            random_rewards.append(random_reward)

            # no charging
            nocharge_reward = self.evaluate_actions(np.zeros((24 * 30 + self.prompt_steps, 1)), env)
            no_charge_rewards.append(nocharge_reward)

            # performance
            dt_p = (dt_reward - random_reward) / (optimal_reward - random_reward)

            # print
            print('Battery size: {:d}kWh | DT Performance: {:.3f}'.format(size, dt_p))