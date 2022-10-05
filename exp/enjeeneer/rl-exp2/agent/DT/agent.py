from actor import Actor
from tokenizers import Tokenizer
from transformer import TransformerBlock, OutputPooler
from embeddings import DiscreteEmbedding, PositionEncoding

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional


class Agent(nn.Module):
    def __init__(self, cfg: Dict):
        super(Agent, self).__init__()

        self.cfg = cfg
        self.block = TransformerBlock(cfg=cfg.transformer)
        self.output_pooler = OutputPooler(cfg=cfg.pooler)
        self.tokenizer = Tokenizer(cfg=cfg.tokenizer)
        self.discrete_embedder = DiscreteEmbedding(cfg=cfg.embedding)
        self.positional_encoder = PositionEncoding(cfg=cfg.pos_encoder)

        if cfg.stochastic_policy:
            self.actor = Actor(cfg.actor)

    def predict_action(self, obs: np.array, act: np.array, rewards: Optional[np.array]):
        """
        Takes sequence, tokenizes, embeds, passes through transformer blocks and pools/
        :param inputs: tensor of shape (?)
        :return outputs: tensor of shape (?)
        """
        # TODO: In main.py, we'll probabably want a loop that makes action token predictions depending on the size of
        #  act_dim, to confine the model to the correct output size.

        # embed sequence
        sequence_embedding = self.tokenize_and_embed(obs, act)
        x = sequence_embedding

        # pass through transformer heads
        for i in range(self.cfg.transformer.blocks):
            x = self.block(x)

        action_bin = self.output_pooler(x)
        action = self.tokenizer.detokenize(action_bin)

        # TODO: stochastic actor
        # if self.actor:
        #     action = self.actor.act(action)

        return action

    def tokenize_and_embed(self, obs: np.array, actions: np.array, rewards: Optional[np.array] = None):
        """
        Takes arrays of states, actions and/or rewards of arbitrary length
        slices to context length and tokenizes and embeds them ready for transformer.
        :param states: array of shape (*, obs_dim)
        :param actions: array of shape (*, act_dim)
        :param rewards: array of shape (*, rew_dim)
        :return sequence embedding tensor of shape (batch_dim, context_length, embed_dim):
        """
        # TODO: provide reward prediction extensibility; do rewards require their own tokenisation procedure?
        # tokenize
        obs_tokens = self.tokenizer.tokenize(torch.tensor(obs, dtype=torch.float, device=self.device))
        act_tokens = self.tokenizer.tokenize(torch.tensor(actions, dtype=torch.float, device=self.device))

        # embed
        obs_embedding = self.discrete_embedder(obs_tokens)
        act_embedding = self.discrete_embedder(act_tokens)

        # add positional encoding
        obs_embedding = obs_embedding + self.positional_encoder(obs_embedding)
        act_embedding = act_embedding + self.positional_encoder(act_embedding)

        # concat
        sequence_embedding = torch.cat([obs_embedding, act_embedding], dim=-1)

        if rewards:
            rew_tokens = self.tokenizer.tokenize(torch.tensor(rewards, dtype=torch.float, device=self.device))
            rew_embedding = self.discrete_embedder(rew_tokens)
            sequence_embedding = torch.cat([sequence_embedding, rew_embedding], dim=-1)

        # TODO: ensure that this slicing fits with array preprocessing
        # slice to context length
        sequence_embedding = sequence_embedding[..., -self.cfg.transformer.context_length:]

        return sequence_embedding

    def train(self):
        pass



