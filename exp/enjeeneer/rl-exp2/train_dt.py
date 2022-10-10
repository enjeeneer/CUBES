import os
import torch
import pickle
from tqdm import tqdm
from torch.optim import AdamW

from agent.DT.agent import Agent
from utils.utils import Cfg
from data.collector import batch

# load config
Cfg = Cfg()
cfg = Cfg.parse(model='dt')

model = Agent(cfg)
optimizer = AdamW(params=model.parameters(),
                  lr=cfg.optim.lr,
                  eps=cfg.optim.epsilon,
                  betas=cfg.optim.betas,
                  weight_decay=cfg.optim.weight_decay)

# load datasets
with open('data/sequenced_dataset.pickle', 'rb') as f:
    dataset = pickle.load(f)

# batch data
input_sequences, target_sequences, obs_masks, action_masks, _ = batch(dataset, cfg.dataset)

losses = []
for i in tqdm(range(cfg.agent.learning_steps)):
    input_batch, target_batch, obs_mask, action_mask = torch.tensor(input_sequences[i, :, :], dtype=torch.int64).to(cfg.device), \
                                                       torch.tensor(target_sequences[i, :, :], dtype=torch.int64).to(cfg.device), \
                                                       torch.tensor(obs_masks[i, :, :], dtype=torch.int64).to(cfg.device), \
                                                       torch.tensor(action_masks[i, :, :], dtype=torch.int64).to(cfg.device)

    # predict
    with torch.set_grad_enabled(True):
        loss = model.predict_sequence(input_sequence=input_batch,
                                  obs_mask=obs_mask,
                                  act_mask=action_mask,
                                  targets=target_batch)
        losses.append(loss)

    # update params
    model.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.grad_norm_clip)
    optimizer.step()
    print('...learning step {:d} | model loss: {:.2f} ...'.format(i, loss))

    # decay the learning rate based on our progress
    # if config.lr_decay:
    #     y = batch[-2]
    #     self.n_tokens += (y != vocab_size).sum()  # number of tokens processed this step
    #     if self.n_tokens < config.warmup_tokens:
    #         # linear warmup
    #         lr_mult = float(self.n_tokens) / float(max(1, config.warmup_tokens))
    #     else:
    #         # cosine learning rate decay
    #         progress = float(self.n_tokens - config.warmup_tokens) / float(
    #             max(1, config.final_tokens - config.warmup_tokens))
    #         lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
    #     lr = config.learning_rate * lr_mult
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] = lr
    # else:
    #     lr = config.learning_rate


