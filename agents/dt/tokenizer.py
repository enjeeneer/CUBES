# pylint: disable=[superfluous-parens, invalid-name]
"""DT's tokenizer"""
import torch
import numpy as np
from typing import Union


class Tokenizer:
    """Tokenizes real-valued inputs"""

    def __init__(
        self,
        mu: int,
        M: int,
        bins: int,
        device: torch.device,
    ):

        self.mu = mu
        self.M = M
        self.bins = bins
        self.device = device

    def mu_law_encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Mu-law normalisation of continuous features. Note of our
        obs/action space is already
        normalised in the range [-1, 1] this is not required.
        From Appendix B of Gato paper: https://arxiv.org/pdf/2205.06175.pdf
        Args:
            x: array of shape (*, obs/act dim)
        Returns:
            output: tensor of shape (*, obs/act/ dim)
        """

        mu = torch.tensor([self.mu], dtype=torch.float32, device=self.device)
        M = torch.tensor([self.M], dtype=torch.float32, device=self.device)

        sign = torch.sign(x)
        numer = torch.log((torch.absolute(x) * mu) + 1)
        denom = torch.log((M * mu) + 1)

        output = sign * (numer / denom)

        # clip to ensure values are in range [-1, 1]
        output = torch.clip(output, min=-1, max=1)

        return output

    def mu_law_decode(self, y: torch.Tensor) -> torch.Tensor:
        """
        Inverse mu-law encoding (i.e. expansion) for continuous features.
        :param y: tensor of shape (*, obs/act/rew dim)
        :return output: tensor of shape (*, obs/act/rew dim)
        """
        mu = torch.tensor([self.mu], dtype=torch.float32).to(self.device)
        M = torch.tensor([self.M], dtype=torch.float32).to(self.device)

        sign = torch.sign(y)
        numer = (1 + mu * M) ** (torch.absolute(y)) - 1
        denom = mu

        output = sign * (numer / denom)

        return output

    @torch.no_grad()
    def tokenize(self, x: Union[torch.tensor, np.array], shift=None) -> torch.Tensor:
        """
        Tokenization of continuous features using a combination of mu-law encoding and
        binning in discrete range [-1, 1].
        From Appendix B of Gato paper: https://arxiv.org/pdf/2205.06175.pdf
        Args:
            x: tensor/array of any shape
            shift: number of idxs to shift by to avoid text tokens in gato paper
        Returns:
            output: tokenized tensor of same shape as input
        """

        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32, device=self.device)

        norm = self.mu_law_encode(x)

        # create bins
        boundaries = torch.arange(
            start=-1, end=1, step=(2 / (self.bins - 1)), device=self.device
        )
        bins = torch.sum(norm.unsqueeze(-1) > boundaries, dim=-1, dtype=torch.long)

        if shift is not None:
            bins += shift

        return bins

    @torch.no_grad()
    def detokenize(self, bins: torch.tensor) -> torch.tensor:
        """
        Takes predicted token(s) from transformer and inverts tokenisation
        procedure to produce real-valued action dimension.
        Args:
            bins: int bin representing quantized token
        Returns
            y: real-valued action dimension
        """
        norm = bins / (self.bins / 2) - 1
        norm = norm.type(torch.float32)

        y = self.mu_law_decode(norm)

        return y
