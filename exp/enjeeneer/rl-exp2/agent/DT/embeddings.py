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
            num_embeddings=1024,  # number of bins in my discretisation (i.e. one-hot encoding)
            embedding_dim=512,  # number of dimensions of embedded vector
            device='cpu'
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
            embedding_dim=self.cfg.embedding_dim,
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
        :param obs_mask: tensor of local obs_dim position and zeros elsewhere, shape [[batch_dim, context_length, 1]
        :param act_mask: one-hot encoding of act_dim position, shape [[batch_dim, context_length, 1]
        :param rew_mask: one-hot encoding of reward position, shape [[batch_dim, context_length, 1]
        :return embedded_input_sequence: embedding tensor with positional encoding added
        """
        if self.cfg.rewards:
            assert rew_mask == True, "Reward mask must be passed as argument if we rewards are being predicted."

        # observations
        obs_pos = embedded_input_sequence[:, obs_mask.astype(bool), :]
        obs_pos_embed = self.embedding(obs_pos)
        embedded_input_sequence[:, obs_mask.astype(bool), :] = embedded_input_sequence[:, obs_mask.astype(bool), :] + obs_pos_embed

        # actions
        act_pos = torch.tensor([self.cfg.position_table_dim - 1], dtype=torch.int)  # every action gets same (last) index in table
        act_pos_embed = self.embedding(act_pos)
        embedded_input_sequence[:, act_mask.astype(bool), :] = embedded_input_sequence[:, act_mask.astype(bool), :] + act_pos_embed

        # rewards
        if self.cfg.rewards:
            rew_pos = torch.tensor([int(self.cfg.position_table_dim - 1 / 2)], dtype=torch.int)  # every rew gets same (middle) index in table
            rew_pos_embed = self.embedding(rew_pos)
            embedded_input_sequence[:, rew_mask.astype(bool), :] = embedded_input_sequence[:, rew_mask.astype(bool), :] + act_pos_embed

        return embedded_input_sequence
