import bauwerk.benchmarks

import pickle
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

# setup thresholds
contexts = [
    18,
    36,
    6
]

Cfg = Cfg()
threshold_rewards = {}

for con in contexts:
    # load config and helper
    cfg = Cfg.parse(model='dt')
    cfg.transformer.context_length = con

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
        tags=['18kWh-training', 'context-testing'],
    )
    wandb.config.update(dict(cfg), allow_val_change=True)

    # create dataset
    collector = Collector(cfg)
    dataset = collector.dataset()

    # train
    trainer = Trainer(cfg, dataset)
    model, losses = trainer.train()
    wandb.log(
        {f"loss-{con}": loss for loss in losses}
    )

    # test
    build_dist_b = bauwerk.benchmarks.BuildDistB(seed=0)
    env = build_dist_b.make_env()
    env = ObsWrapper(env)
    battery_sizes = np.arange(1, 21, 1)
    helper = Helper(cfg=cfg, tokenizer=tokenizer, obs_dim=5)
    dt_rewards, optimal_rewards, nocharge_rewards = helper.test_across_battery_sizes(model, battery_sizes)
    threshold_rewards[str(con)] = dt_rewards


with open('threshold-rewards.pickle', 'wb') as f:
    pickle.dump(threshold_rewards, f)

# plot
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
plt.plot(battery_sizes, optimal_rewards, label='optimal')
plt.plot(battery_sizes, nocharge_rewards, label='no charging', linestyle='--', color='lightblue')
for key, value in threshold_rewards.items():
    plt.plot(battery_sizes, value, label=key)
plt.ylabel('average $ per step')
plt.xlabel('battery size (kWh)')
plt.legend()
plt.tight_layout()

# login wandb
wandb.log({
    'chart': wandb.Image(plt)
})
run.finish()
