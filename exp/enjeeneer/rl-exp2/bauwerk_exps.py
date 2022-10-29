import bauwerk
import bauwerk.benchmarks

import wandb
import omegaconf
import numpy as np
import matplotlib.pyplot as plt

from utils.utils import Cfg
from utils.utils import ObsWrapper
from data.tokenizer import Tokenizer
from exp.helper import Helper
from exp.trainer import Trainer
from exp.dataset import Collector

# load config and helper
Cfg = Cfg()
cfg = Cfg.parse(model='dt')

# import tokenizer
tokenizer = Tokenizer(cfg.tokenizer)

# unroll cfg for wandb
wandb_cfg = {}
for key1, value1 in cfg.items():
    if type(value1) == omegaconf.dictconfig.DictConfig:
        for key2, value2 in value1.items():
            wandb_cfg[key2] = value2
    else:
        wandb_cfg[key1] = value1

# setup wandb
run = wandb.init(
    project='cubes',
    entity="beobench",
    config=wandb_cfg,
    tags=['18kWh-training', 'eval'],
)
wandb.config.update(dict(cfg))

# create dataset
collector = Collector(cfg)
dataset = collector.dataset()

# train
trainer = Trainer(cfg, dataset)
model, losses = trainer.train()

# test
build_dist_b = bauwerk.benchmarks.BuildDistB(seed=0)
env = build_dist_b.make_env()
env = ObsWrapper(env)
battery_sizes = np.arange(1, 21, 1)
helper = Helper(cfg=cfg, tokenizer=tokenizer, obs_dim=5)
dt_rewards, optimal_rewards, nocharge_rewards = helper.test_across_battery_sizes(model, battery_sizes)

# plot
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
plt.plot(battery_sizes, optimal_rewards, label='optimal')
plt.plot(battery_sizes, dt_rewards, label='DT')
plt.plot(battery_sizes, nocharge_rewards, label='no charging', linestyle='--', color='lightblue')
plt.ylabel('average $ per step')
plt.xlabel('battery size (kWh)')
plt.legend()
plt.tight_layout()

# login wandb
wandb.log({
    'chart': wandb.Image(plt),
    'loss': losses,
           })
run.finish()
