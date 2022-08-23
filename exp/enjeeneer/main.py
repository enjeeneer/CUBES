import os
import gym
import wandb
from configs.parser import parse_cfg
from agent import Agent as SAC

cfg = parse_cfg()
wandb.init(project="sac-testing", entity="enjeeneer", tags=['hopper'])
env = gym.make("Walker2d-v3")
agent = SAC(cfg=cfg, env=env, obs_dim=env.observation_space.shape[0], act_dim=env.action_space.shape[0], models_dir='tmp/')

episodes = 4000
eval_interal = 10
eval_episodes = 10

for i in range(episodes):
    ep_reward = 0
    episode_steps = 0
    evals = 0
    done = False
    state = env.reset()
    while not done:
        action = agent.act(state, evaluate=False)
        state_, reward, done, _ = env.step(action)
        agent.memory.store_transition(state, state_, action, reward, done)
        state = state_
        agent.n_steps += 1
        episode_steps += 1
        ep_reward += reward

        if agent.n_steps > agent.batch_size:
            value_loss, actor_loss, critic_loss = agent.learn()
            # wandb.log({
            #     'value_loss': value_loss,
            #     'actor_loss': actor_loss,
            #     'critic_loss': critic_loss
            # })

    print('Episode: {}, episode timesteps: {}, episode reward: {}'.format(i, episode_steps, ep_reward))

    if i % eval_interal == 0:
        eval_rewards = 0
        for i in range(eval_episodes):
            evals += 1
            done = False
            state = env.reset()
            while not done:
                action = agent.act(state, evaluate=True)
                state_, reward, done, _ = env.step(action)
                state = state_
                eval_rewards += reward

        eval_rewards = eval_rewards / eval_episodes # mean


        print('EVAL Episode: {}, episode reward: {}'.format(evals, eval_rewards))

    wandb.log({
        'train_reward': ep_reward,
        'test_reward': eval_rewards,
    })
