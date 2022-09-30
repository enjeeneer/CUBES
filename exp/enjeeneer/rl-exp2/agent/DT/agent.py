from actor import Actor
from tokenizers import ContinuousValueTokenizer
from transformer import TransformerBlock, OutputPooler
from embeddings import DiscreteEmbedding, PositionEncoding

import torch
import torch.nn as nn
from typing import Dict, Any, Union


class Agent(nn.Module):
    def __init__(self, cfg: Dict):
        super(Agent, self).__init__()

        self.cfg = cfg
        self.block = TransformerBlock(cfg=cfg.transformer)
        self.output_pooler = OutputPooler(cfg=cfg.pooler)
        self.tokenizer = ContinuousValueTokenizer(cfg=cfg.tokenizer)
        self.discrete_embedder = DiscreteEmbedding(cfg=cfg.embedding)
        self.positional_encoder = PositionEncoding(cfg=cfg.pos_encoder)

        if cfg.stochastic_policy:
            self.actor = Actor(cfg.actor)

    def predict(self, inputs):
        """
        Takes sequence, tokenizes, embeds, passes through transformer blocks and pools/
        :param inputs: tensor of shape (?)
        :return outputs: tensor of shape (?)
        """
        tokens = self.tokenizer(inputs)
        embedding = self.discrete_embedder(tokens)
        embedding = embedding + self.positional_encoder(embedding)
        x = embedding

        # pass through transformer heads
        for i in range(self.cfg.num_trans_blocks):
            x = self.block(x)

        action = self.output_pooler(x)
        if self.actor:
            action = self.actor.act(action)

        return action

    def update(self):
        pass



