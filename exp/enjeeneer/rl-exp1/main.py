import gym
import wandb
import bauwerk
import argparse
import numpy as np
from tqdm import tqdm
from cfgs.parser import parse_cfg
from cfgs.wrappers import ObsWrapper, dict_to_array
from agent.sac import Agent as SAC

env = gym.make("bauwerk/SolarBatteryHouse-v0")
env = ObsWrapper(env)
episodes = 10
eval_steps = 24 * 30
eval_interval = 1
sweep = True

wandb.init(project="bauwerk",
           entity="enjeeneer",
           tags=['sac-sweep'],
           reinit=True)

if sweep:
    parser = argparse.ArgumentParser()
    parser.add_argument('--hist_length', type=int)
    parser.add_argument('--alpha', type=float)
    parser.add_argument('--tau', type=float)
    parser.add_argument('--scale', type=int)
    parser.add_argument('--gamma', type=float)
    parser.add_argument('--layer_dims', type=int)
    parser.add_argument('--batch_size', type=int)
    args = parser.parse_args()
    wandb.config.update(args)
    cfg = wandb.config
else:
    cfg = parse_cfg()
    wandb.config.update(cfg)

agent = SAC(cfg=cfg, env=env, models_dir='tmp/')

# function for evaluating mean performance of actions
def evaluate_actions(actions, env):
    cum_reward = 0
    obs = env.reset()
    for action in actions:
        obs, reward, _, _, _, _ = env.step(np.array(action, dtype=np.float32))
        cum_reward += reward

    return cum_reward / len(actions)


# evaluate random actions
random_trials = [evaluate_actions([env.action_space.sample() for _ in range(eval_steps)], env) for _ in range(100)]
p_rand = np.mean(random_trials)
print('p_rand: {:.4f}'.format(p_rand))

# evaluate optimal actions
optimal_actions, _ = bauwerk.solve(env)
p_opt = evaluate_actions(optimal_actions.reshape((-1, 1))[:eval_steps], env)
print('p_opt: {:.4f}'.format(p_opt))

# run SAC
for i in tqdm(range(episodes)):
    ep_reward = 0
    evals = 0
    done = False
    obs = dict_to_array(env.reset()[0])

    while not done:
        action, state = agent.act(obs, evaluate=False)
        obs_, reward, done, obs_dict, _, _ = env.step(action)

        if agent.n_steps > cfg.hist_length:
            history = agent.memory.get_history()
            state_ = np.concatenate((obs_, history), axis=0)
            agent.memory.store_transition(state, state_, action, reward, done)

        merged_data = {**obs_dict}
        merged_data['reward'] = reward
        merged_data['action'] = action
        wandb.log(merged_data)

        obs = obs_
        agent.n_steps += 1

        if agent.n_steps > cfg.batch_size:
            value_loss, actor_loss, critic_loss = agent.learn()

    if i % eval_interval == 0:
        print('...performing evaluation...')
        # perform evaluation at end of each episode
        eval_rewards = 0
        obs = dict_to_array(env.reset()[0])
        for j in range(eval_steps + cfg.hist_length):
            action, state = agent.act(obs, evaluate=True)
            obs_, reward, done, _, _, _ = env.step(action)
            obs = obs_

            # only log rewards for steps beyond hist length
            if j > cfg.hist_length:
                eval_rewards += reward

        mean_eval_reward = eval_rewards / eval_steps
        print('EVAL Number: {}, mean reward: {}'.format(i, mean_eval_reward))

        wandb.log({
            'p_sac_eval': mean_eval_reward,
            'p_random': p_rand,
            'p_optimal': p_opt
        })

wandb.log({
    'p_sac': mean_eval_reward
})

env.close()




