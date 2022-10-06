from actor import Actor
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
        self.discrete_embedder = DiscreteEmbedding(cfg=cfg.embedding)
        self.positional_encoder = PositionEncoding(cfg=cfg.pos_encoder)

        if cfg.stochastic_policy:
            self.actor = Actor(cfg.actor)

    def predict_sequence(self, obs: torch.tensor,
                               act: torch.tensor,
                               rewards: Optional[torch.tensor],
                               targets: Optional[torch.tensor],
                               masks: Optional[torch.tensor]
                         ):
        """
        Takes sequence, tokenizes, embeds, passes through transformer blocks and pools/
        :param obs: tensor of shape (sequence_length, batch_size, obs_dim)
        :param act: tensor of shape (sequence_length, batch_size, act_dim)
        :param rewards: tensor of shape (sequence_length, batch_size, 1)
        :param targets: tensor of target variables, shape [sequence_length, batch_size, embed_dim]
        :param masks: tensor of masked variables for loss function, shape [sequence_length, batch_size, embed_dim]
        :return output: tensor of shape (sequence_length, batch_size, 1) i.e. real-valued output
        """
        # TODO: In main.py, we'll probabably want a loop that makes action token predictions depending on the size of
        #  act_dim, to confine the model to the correct output size.

        # embed sequence
        sequence_embedding = self.tokenize_and_embed(obs, act)
        x = sequence_embedding

        # pass through transformer heads
        for i in range(self.cfg.transformer.blocks):
            x = self.block(x)

        # training
        if targets:
            loss = self.output_pooler(x, targets, masks)

            return loss

        # deployment
        else:
            output_bins = self.output_pooler(x)
            output = self.tokenizer.detokenize(output_bins)

            return output

        # TODO: stochastic actor
        # if self.actor:
        #     action = self.actor.act(action)



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
        obs_embedding = obs_embedding + self.positional_encoder(obs_embedding)  # [seq_length, batch, obs_dim, embed]
        act_embedding = act_embedding + self.positional_encoder(act_embedding, actions=True)

        # concat
        sequence_embedding = torch.cat([obs_embedding, act_embedding], dim=-1)   # [seq, batch, ]

        if rewards:
            rew_tokens = self.tokenizer.tokenize(torch.tensor(rewards, dtype=torch.float, device=self.device))
            rew_embedding = self.discrete_embedder(rew_tokens)
            sequence_embedding = torch.cat([sequence_embedding, rew_embedding], dim=-1)

        # TODO: ensure that this slicing fits with array preprocessing
        # slice to context length
        sequence_embedding = sequence_embedding[..., -self.cfg.transformer.context_length:]

        return sequence_embedding

    def train(self, obs_array: np.array, act_array:np.array, target_array: np.array):
        """

        :param obs_array:
        :param act_array:
        :param target_array: tensor of cont-valued target actions of shape (*, act_dim)
        :return:
        """
        obs_array

        target_tokens = self.tokenizer.tokenize(target_array)
        target_tokens = torch.nn.functional.one_hot(target_tokens, num_classes=self.cfg.tokeniser.bins)

    def masking(self):






