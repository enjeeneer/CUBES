"""Module for embedding functions for DT."""
import torch


class DiscreteEmbedding(torch.nn.Module):
    """
    Class for embedding a discrete observation/action token.
    See Figure 13 in Gato paper.
    """

    def __init__(
        self,
        embedding_number: int,
        embedding_dimension: int,
        device: torch.device,
    ):
        super().__init__()

        self.embedding_number = embedding_number

        self.embedding = torch.nn.Embedding(
            # number of bins in my discretisation (i.e. one-hot encoding)
            num_embeddings=embedding_number,
            # number of dimensions of embedded vector
            embedding_dim=embedding_dimension,
            device=device,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Find embedding vector for a discretised obs/action vector.
        Takes a batch of discretised tokens of shape (*, 1), where the token dimension
        must be an int in the range [0, num_embeddings).

        :param inputs: tensor of shape (?)
        :return: embedding tensor of shape (*, embedding_dim)
        """
        assert torch.all(
            inputs[..., :] < self.embedding_number
        ), f"Discretised token must take value < {self.embedding_number}."

        return self.embedding(inputs)


class PositionEncoding(torch.nn.Module):
    """
    Class for encoding the local-position of each
    observation/action dim. E.g. if an observation
    has dimension 5, we create a local positional
    embedding for each of the 5 dimensions, to signpost
    where they sit in the observation. NB actions are
    always given the same positional encoding. See Figure 16 in Gato paper.
    """

    def __init__(
        self,
        table_dimension: int,
        embedding_dimension: int,
        device: torch.device,
    ):
        super().__init__()
        self.table_dimension = table_dimension

        self.embedding = torch.nn.Embedding(
            # larger value than largest possible obs_dim
            num_embeddings=table_dimension,
            embedding_dim=embedding_dimension,
            device=device,
        )

    def embed(
        self,
        embedded_input_sequence: torch.tensor,
        obs_mask: torch.tensor,
        act_mask: torch.tensor,
    ) -> torch.tensor:
        """
        Finds positional encoding input sequence. Figure 16 in Gato paper.
        Takes tensor of observations or actions and finds
        their local positional embedding. NB: actions
        are always given the same positional embedding which we
        take, arbitrarily, as the last index in
        the table look-up.
        Args:
            embedded_input_sequence: embedding tensor
                                        of shape [batch_dim, context_length, embed_dim]
            obs_mask: tensor of local obs_dim position
                            and zeros elsewhere, shape [batch_dim, context_length]
            act_mask: one-hot encoding of
                                    act_dim position, shape [batch_dim, context_length]
        Returns:
            embedded_input_sequence: embedding tensor with positional encoding added
        """

        # observations
        obs_pos_embed = self.embedding(obs_mask)  # [batch, context, embed]
        obs_mask_bool = obs_mask.type(torch.bool)
        embedded_input_sequence[obs_mask_bool] = (
            embedded_input_sequence[obs_mask_bool] + obs_pos_embed[obs_mask_bool]
        )  # only add to pos to obs values

        # actions (every action gets same embedding -- last input in table)
        act_pos = torch.ones(
            size=(embedded_input_sequence.shape[0], embedded_input_sequence.shape[1]),
            dtype=torch.int,
        )
        act_pos = act_pos * int(self.table_dimension - 1)
        act_pos_embed = self.embedding(act_pos)
        act_mask_bool = act_mask.type(torch.bool)
        embedded_input_sequence[act_mask_bool] = (
            embedded_input_sequence[act_mask_bool] + act_pos_embed[act_mask_bool]
        )

        return embedded_input_sequence
