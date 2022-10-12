import torch
import torch.nn as nn
from typing import Dict, Optional


class DiscreteEmbedding(nn.Module):
    """
    Class for embedding a discrete observation/action token.
    See Figure 13 in Gato paper.
    """

    def __init__(self, cfg: Dict):
        super(DiscreteEmbedding, self).__init__()

        self.cfg = cfg
        self.embedding = nn.Embedding(
            num_embeddings=self.cfg.embed_num,  # number of bins in my discretisation (i.e. one-hot encoding)
            embedding_dim=self.cfg.embed_dim,  # number of dimensions of embedded vector
            device=self.cfg.device
        )

    def forward(self, inputs):
        """
        Find embedding vector for a discretised obs/action vector.
        Takes tensor of batches discretised tokens of shape (*, 1), but the value in the last dimenision
        must be an int in the range [0, num_embeddings).

        :param inputs: tensor of shape (?)
        :return: embedding tensor of shape (*, embedding_dim)
        """
        return self.embedding(inputs)


class PositionEncoding(nn.Module):
    """
    Class for encoding the local-position of each observation / action dim. E.g. if an observation
    has dimension 5, we find create a local postional embedding for each of the 5 dimensions, to signpost
    where they sit in the observation. NB actions are always given the same positional encoding.
    See Figure 16 in Gato paper.
    """

    def __init__(self, cfg: Dict):
        super(PositionEncoding, self).__init__()

        self.cfg = cfg
        self.embedding = nn.Embedding(
            num_embeddings=self.cfg.position_table_dim,  # larger value than any obs_dim
            embedding_dim=self.cfg.embed_dim,
            device=self.cfg.device
        )

    def embed(self, embedded_input_sequence: torch.tensor,
                      obs_mask: torch.tensor,
                      act_mask: torch.tensor,
                      rew_mask: Optional[torch.tensor] = False) -> torch.tensor:
        """
        Finds positional encoding input sequence. Figure 16 in Gato paper.
        Takes tensor of observations or actions and finds their local positional embedding. NB: actions
        are always given the same positional embedding which we take, arbitrarily, as the last index in
        the table look-up
        :param embedded_input_sequence: embedding tensor of shape [batch_dim, context_length, embed_dim]
        :param obs_mask: tensor of local obs_dim position and zeros elsewhere, shape [[batch_dim, context_length]
        :param act_mask: one-hot encoding of act_dim position, shape [[batch_dim, context_length]
        :param rew_mask: one-hot encoding of reward position, shape [[batch_dim, context_length]
        :return embedded_input_sequence: embedding tensor with positional encoding added
        """
        if self.cfg.rewards:
            assert rew_mask == True, "Reward mask must be passed as argument if we rewards are being predicted."

        # observations
        obs_pos_embed = self.embedding(obs_mask)  # [batch, context, embed]
        obs_mask_bool = obs_mask.type(torch.bool)
        embedded_input_sequence[obs_mask_bool] = embedded_input_sequence[obs_mask_bool]\
                                                                + obs_pos_embed[obs_mask_bool]  # only add to pos to obs values

        # actions (every action gets same embedding -- last input in table)
        act_pos = torch.ones(size=(embedded_input_sequence.shape[0], embedded_input_sequence.shape[1]), dtype=torch.int64)  #
        act_pos = act_pos * int(self.cfg.position_table_dim - 1)
        act_pos_embed = self.embedding(act_pos)
        act_mask_bool = act_mask.type(torch.bool)
        embedded_input_sequence[act_mask_bool] = embedded_input_sequence[act_mask_bool] + act_pos_embed[act_mask_bool]

        # rewards (every reward gets same embedding -- middle input in table
        if self.cfg.rewards:
            rew_pos = torch.ones(size=(embedded_input_sequence.shape[0], embedded_input_sequence.shape[1]), dtype=torch.int64)
            rew_pos = rew_pos * int((self.cfg.position_table_dim - 1) / 2)
            rew_pos_embed = self.embedding(rew_pos)
            rew_mask_bool = rew_mask.type(torch.bool)
            embedded_input_sequence[rew_mask_bool] = embedded_input_sequence[rew_mask_bool] + rew_pos_embed[rew_mask_bool]

        return embedded_input_sequence
