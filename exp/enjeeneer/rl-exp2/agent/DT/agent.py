from agent.DT.transformer import TransformerBlock, OutputPooler
from agent.DT.embeddings import DiscreteEmbedding, PositionEncoding

import torch
import torch.nn as nn
from typing import Dict, Optional


class Agent(nn.Module):
    def __init__(self, cfg: Dict):
        super(Agent, self).__init__()

        self.cfg = cfg
        self.device = cfg.device
        self.block_0 = TransformerBlock(cfg=cfg.transformer, block=str(0))
        self.block_1 = TransformerBlock(cfg=cfg.transformer, block=str(1))
        self.block_2 = TransformerBlock(cfg=cfg.transformer, block=str(2))
        self.block_3 = TransformerBlock(cfg=cfg.transformer, block=str(3))
        self.blocks = [self.block_0, self.block_1, self.block_2, self.block_3]
        self.output_pooler = OutputPooler(cfg=cfg.pooler)
        self.discrete_embedder = DiscreteEmbedding(cfg=cfg.embedding)
        self.positional_encoder = PositionEncoding(cfg=cfg.pos_encoder)

        # if cfg.stochastic_policy:
        #     self.actor = Actor(cfg.actor)

    def predict_sequence(self, input_sequence: torch.tensor,
                         obs_mask: torch.tensor,
                         act_mask: torch.tensor,
                         rew_mask: Optional[torch.tensor] = None,
                         targets: Optional[torch.tensor] = None):
        """
        Takes sequence, embeds, passes through transformer blocks and pools/
        :param input_sequence: tensor of inputs of shape [batch_dim, context_length]
        :param obs_mask: tensor of inputs of shape [batch_dim, context_length]
        :param act_mask: tensor of inputs of shape [batch_dim, context_length]
        :param rew_mask: tensor of inputs of shape [batch_dim, context_length]
        :param targets: tensor of target variables, shape [sequence_length, batch_size, embed_dim]
        :return loss: loss tensor of shape [batch_size,]
        :return output: output tensor of shape [sequence_length, batch_size] i.e. real-valued output
        """
        # TODO: In main.py, we'll probabably want a loop that makes action token predictions depending on the size of
        #  act_dim, to confine the model to the correct output size.

        # embed sequence
        if self.cfg.dataset.rewards:
            sequence_embedding = self.embed(input_sequence=input_sequence, obs_mask=obs_mask,
                                            action_mask=act_mask, reward_mask=rew_mask)
        else:
            sequence_embedding = self.embed(input_sequence=input_sequence, obs_mask=obs_mask, action_mask=act_mask)

        x = sequence_embedding

        # pass through transformer blocks
        for i in range(self.cfg.transformer.blocks):
            x = self.blocks[i].forward(x)

        # training
        if targets is not None:
            loss = self.output_pooler(x=x, targets=targets, action_mask=act_mask)

            return loss

        # deployment
        else:
            output_bins = self.output_pooler(x=x)
            output = self.tokenizer.detokenize(output_bins)

            return output

        # TODO: stochastic actor

    def embed(self, input_sequence: torch.tensor,
              obs_mask: torch.tensor,
              action_mask: torch.tensor,
              reward_mask: Optional[torch.tensor] = False):
        """
        Takes arrays of states, actions and/or rewards of arbitrary length
        slices to context length and tokenizes and embeds them ready for transformer.
        :param input_sequence: tensor of shape [batch_size, context_length]
        :param obs_mask: tensor of shape [batch_size, context_length]
        :param action_mask: tensor of shape [batch_size, context_length]
        :param reward_mask: tensor of shape [batch_size, context_length]
        :return embedded_sequence: sequence embedding tensor of shape [batch_size, context_length, embed_dim]
        """
        # embed
        embedded_sequence = self.discrete_embedder(input_sequence)  # [batch, context, embed]

        # add positional encoding
        if self.cfg.dataset.rewards:
            embedded_sequence = self.positional_encoder.embed(embedded_input_sequence=embedded_sequence,
                                                              obs_mask=obs_mask,
                                                              act_mask=action_mask,
                                                              rew_mask=reward_mask)

        else:
            embedded_sequence = self.positional_encoder.embed(embedded_input_sequence=embedded_sequence,
                                                              obs_mask=obs_mask,
                                                              act_mask=action_mask)

        return embedded_sequence









