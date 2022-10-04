import torch
from typing import Dict


@torch.no_grad()
def mu_law_encode(x, mu, M):
    """
    Mu-law normalisation of continuous features.
    From Appendix B of Gato paper: https://arxiv.org/pdf/2205.06175.pdf 
    """
    mu = torch.tensor([mu], dtype=torch.float32)
    M = torch.tensor([M], dtype=torch.float32)

    sign = torch.sign(x)
    numer = torch.log((torch.absolute(x) * mu) + 1)
    denom = torch.log((M * mu) + 1)

    output = sign * (numer / denom)

    return output


@torch.no_grad()
def tokenize_cont_values(x, mu=100, M=256, bins=1024, shift=None):
    """
    Tokenization of continuous features using a combination of mu-law encoding and
    binning in discrete range [-1, 1]. From Appendix B of Gato paper: https://arxiv.org/pdf/2205.06175.pdf
    :param x: tensor of shape (?)
    :param mu: mu-law encding param
    :param M: mu-law encding param
    :param bins: number of bins for discretisation
    :param shift: number of idxs to shift by to avoid text tokens in gato paper
    :return: tokenized tensor of shape (?)
    """

    norm = mu_law_encode(x, mu, M)
    bin = (norm + 1) * (bins / 2)  # get discrete bin index
    bin = bin.type(torch.LongTensor)  # convert to int64

    if shift is not None:
        bin += shift

    return bin


class ContinuousValueTokenizer:
    def __init__(self):
        super(ContinuousValueTokenizer, self).__init__()

    @staticmethod
    def call(inputs):
        outputs = tokenize_cont_values(inputs)

        return outputs
