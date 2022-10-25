import torch
import numpy as np
from typing import Dict, Optional, Union


class Tokenizer:
    def __init__(self, cfg):
        super(Tokenizer, self).__init__()

        self.cfg = cfg

    def mu_law(self, x: np.array):
        """
        Mu-law normalisation of continuous features. Note if our obs/action space is already
        normalised in the range [-1, 1] this is not required.
        From Appendix B of Gato paper: https://arxiv.org/pdf/2205.06175.pdf
        :param x: array of shape (*, obs/act/rew dim)
        :return output: tensor of shape (*, obs/act/rew dim)
        """
        x = torch.tensor(x, dtype=torch.float32)
        mu = torch.tensor([self.cfg.mu], dtype=torch.float32)

        sign = torch.sign(x)
        numer = torch.log((torch.absolute(x) * mu) + 1)
        denom = torch.log(mu + 1)

        output = sign * (numer / denom)

        # clip to ensure values are in range [-1, 1]
        output = torch.clip(output, min=-1, max=1)

        return output

    def inverse_mu_law(self, y):
        """
        Inverse mu-law encoding (i.e. expansion) for continuous features.
        :param y: tensor of shape (*, obs/act/rew dim)
        :return output: tensor of shape (*, obs/act/rew dim)
        """
        mu = torch.tensor([self.cfg.mu], dtype=torch.float32)

        sign = torch.sign(y)
        numer = (1 + mu)**(torch.absolute(y)) - 1
        denom = mu

        output = sign * (numer / denom)

        return output

    @torch.no_grad()
    def tokenize(self, x: Union[torch.tensor, np.array], shift=None):
        """
        Tokenization of continuous features using a combination of mu-law encoding and
        binning in discrete range [-1, 1]. From Appendix B of Gato paper: https://arxiv.org/pdf/2205.06175.pdf
        :param x: tensor/array of any shape
        :param shift: number of idxs to shift by to avoid text tokens in gato paper
        :return: tokenized tensor of same shape as input
        """
        if type(x) == np.ndarray:
            x = torch.tensor(x, dtype=torch.float).to(self.cfg.device)

        norm = self.mu_law(x)
        bin = torch.bucketize(input=norm, boundaries=torch.arange(start=-1, end=1, step=(2 / (self.cfg.bins - 1))))
        bin = bin.type(torch.LongTensor)  # convert to int64

        if shift is not None:
            bin += shift

        return bin.numpy()

    @torch.no_grad()
    def detokenize(self, bin):
        """
        Takes predicted token(s) from transformer and inverts tokenisation procedure to produce real-valued action
        :param bin: int bin representing quantized token
        :return y:
        """
        norm = bin / (self.cfg.bins / 2) - 1
        norm = norm.type(torch.float32)

        y = self.inverse_mu_law(norm)

        return y

    def update_sequences(self, sequence: torch.tensor,
                               obs_mask: torch.tensor,
                               act_mask: torch.tensor,
                               tokens: torch.tensor,
                               obs: Optional[bool] = False,
                               action: Optional[bool] =False):
        """
        Add news tokens to sequence and updates masks.
        :param sequence: tensor, shape [batch_size, context_length]
        :param obs_mask: tensor, shape [batch_size, context_length]
        :param act_mask: tensor, shape [batch_size, context_length]
        :param tokens: tensor, shape Union[[batch_size, obs_dim,], [batch_size, act_dim]]
        :param obs: bool flag to indicate whether tokens are from observation
        :param action: bool flag to indicate whether tokens are from action
        :return sequence: tensor, shape [batch_size, context_length]
        :return obs_mask: tensor, shape [batch_size, context_length]
        :return act_mask: tensor, shape [batch_size, context_length]
        """

        n_tokens = tokens.shape[0]

        # sequence
        sequence[:-n_tokens] = sequence[n_tokens:]
        sequence[-n_tokens:] = tokens

        # masks
        obs_mask[:-n_tokens] = obs_mask[n_tokens:]
        act_mask[:-n_tokens] = act_mask[n_tokens:]

        if obs:
            obs_mask[-n_tokens:] = np.arange(start=1, stop=n_tokens + 1)
            act_mask[-n_tokens:] = 0

        if action:
            obs_mask[-n_tokens:] = 0
            act_mask[-n_tokens:] = 1

        return sequence, obs_mask, act_mask






