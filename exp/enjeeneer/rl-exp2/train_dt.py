import os
import torch
import pickle
from tqdm import tqdm
from torch.optim import AdamW

from agent.DT.agent import Agent
from cfgs.parser import parse_cfg
from data.scripts import batch

# load config
cfg = parse_cfg(model='dt')

model = Agent(cfg)
optimizer = AdamW(params=model.parameters(),
                  lr=cfg.optim.lr,
                  eps=cfg.optim.eps,
                  betas=cfg.optim.betas,
                  weight_decay=cfg.optim.weight_decay)

# load datasets
data_path = os.path.join(cfg.dataset.save_dir, cfg.dataset.seq_name)
with open(data_path, 'rb') as f:
    dataset = pickle.load(f)

# batch data
input_sequences, target_sequences, obs_masks, action_masks, _ = batch(dataset, cfg.dataset)

losses = []
for i in tqdm(range(cfg.learning_steps)):
    input_batch, target_batch, obs_mask, action_mask = torch.tensor(input_sequences[i, :, :], dtype=torch.float32).to(cfg.device), \
                                                       torch.tensor(target_sequences[i, :, :], dtype=torch.float32).to(cfg.device), \
                                                       torch.tensor(obs_masks[i, :, :], dtype=torch.float32).to(cfg.device), \
                                                       torch.tensor(action_masks[i, :, :], dtype=torch.float32).(cfg.device)

    # one hot encode target sequence
    target_one_hot = torch.nn.functional.one_hot(target_batch, num_classes=cfg.bins)  # [batch, context, bins]

    # predict
    with torch.set_grad_enabled(True):
        loss = model.predict_sequence(input_sequence=input_batch,
                                  obs_mask=obs_mask,
                                  act_mask=action_mask,
                                  targets=target_one_hot)
        losses.append(loss)

    # update params
    model.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.grad_norm_clip)
    optimizer.step()

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


