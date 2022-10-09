import torch
import numpy as np
from typing import Dict


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

        return output

    def inverse_mu_law(self, y):
        """
        Inverse mu-law encoding (i.e. expansion) for continuous features. Note if our obs/action space is already
        normalised in the range [-1, 1] this is not required.
        :param y: tensor of shape (*, obs/act/rew dim)
        :return output: array of shape (*, obs/act/rew dim)
        """
        mu = torch.tensor([self.cfg.mu], dtype=torch.float32)

        sign = torch.sign(y)
        numer = (1 + mu)**(torch.absolute(y)) - 1
        denom = mu

        output = sign * (numer / denom)

        output = output.numpy().detach()

        return output

    def tokenize(self, x, shift=None):
        """
        Tokenization of continuous features using a combination of mu-law encoding and
        binning in discrete range [-1, 1]. From Appendix B of Gato paper: https://arxiv.org/pdf/2205.06175.pdf
        :param x: tensor of any shape
        :param shift: number of idxs to shift by to avoid text tokens in gato paper
        :return: tokenized tensor of same shape as input
        """

        norm = self.mu_law(x)
        bin = (norm + 1) * (self.cfg.bins / 2)  # get discrete bin index
        bin = bin.type(torch.LongTensor)  # convert to int64

        if shift is not None:
            bin += shift

        return bin

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



