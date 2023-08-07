import random
from itertools import cycle
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from src.models.modules.pooling_modules.pooling_base import GroupPoolingBase


class ConnectedSearch2(GroupPoolingBase):
    """
    For RBP with a head region, where the search vector from the head region is different from non-head region to
    non-head region
    """
    supports_precomputing = True
    forward_args = "precomputed", "x", "ch_names", "channel_name_to_index"

    def __init__(self, *, latent_search_features: int, in_features: int, share_edge_embeddings: bool, num_regions: int,
                 head_region_idx: Optional[int] = None, bias: bool = False):
        """
        Initialise
        Args:
            latent_search_features: Number of features of the latent search vector
            in_features: Number of input features (e.g. num_kernels*2 if using ROCKET features)
            share_edge_embeddings: If the mapping from head region should share some if its parameters with the non-head
                regions (see Eqs. 4 and 7 in the paper. If this argument is set to false, then the parameters in Eqs. 4
                and 7 will not be the same)
            num_regions: Number of regions of the channel split
            head_region_idx: Index of which region is the head region
            bias: To use bias (True) or not (False)
        Examples:
            >>> ConnectedSearch2(latent_search_features=65, in_features=77, share_edge_embeddings=True, num_regions=35,
            ...                  head_region_idx=8, bias=True).hyperparameters  # doctest: +NORMALIZE_WHITESPACE
            {'latent_search_features': 65, 'in_features': 77, 'share_edge_embeddings': True, 'num_regions': 35,
             'head_region_idx': 8, 'bias': True}
            >>> random.seed(3)  # random.seed is required to reproduce if head_region_idx is not specified
            >>> ConnectedSearch2(latent_search_features=65, in_features=77, share_edge_embeddings=True, num_regions=35,
            ...                  bias=True).hyperparameters  # doctest: +NORMALIZE_WHITESPACE
            {'latent_search_features': 65, 'in_features': 77, 'share_edge_embeddings': True, 'num_regions': 35,
             'bias': True, 'head_region_idx': 15}
            >>> ConnectedSearch2(latent_search_features=65, in_features=7, share_edge_embeddings=True, num_regions=35,
            ...                  head_region_idx=35, bias=True).hyperparameters
            Traceback (most recent call last):
            ...
            ValueError: The index of the head region cannot exceed the number of head regions, but were 35 and 35
        """
        super().__init__()

        # Maybe select head region at random
        head_region_idx = random.randint(0, num_regions - 1) if head_region_idx is None else head_region_idx
        self.update_input_dict(key="head_region_idx", value=head_region_idx)

        # Input check
        if head_region_idx >= num_regions:
            raise ValueError(f"The index of the head region cannot exceed the number of head regions, but were "
                             f"{head_region_idx} and {num_regions}")

        # ------------------
        # Non-head regions
        # ------------------
        self._search_mappings = nn.ModuleList([nn.Linear(in_features=in_features, out_features=latent_search_features,
                                                         bias=bias) for _ in range(num_regions-1)])

        # ------------------
        # Head region
        # ------------------
        if share_edge_embeddings:
            self._head_search_linear = cycle((None,))

        else:
            self._head_search_linear = nn.ModuleList(
                [nn.Linear(in_features=in_features, out_features=latent_search_features,
                           bias=bias) for _ in range(num_regions-1)])
        self._head_search_gates = nn.ModuleList(
            [nn.Linear(in_features=in_features, out_features=latent_search_features,
                       bias=bias) for _ in range(num_regions-1)])
        self._head_node_mapping = nn.Linear(in_features=in_features, out_features=1, bias=bias)

        self._head_region_idx = head_region_idx

        # Store settings
        self._share_edge_embeddings = share_edge_embeddings

    def forward(self, x: torch.Tensor, precomputed: torch.Tensor, ch_names: Tuple[Tuple[str, ...], ...],
                channel_name_to_index: Dict[str, int]) -> torch.Tensor:
        """
        Forward method
        Args:
            x: Full EEG data
            precomputed: Precomputed features (one feature vector per channel)
            ch_names: Channel names of the regions. The i-th element is the channels contained in the i-th region
            channel_name_to_index: Mapping from channel name to index

        Returns: A concatenation of the region representations

        Examples:
            >>> my_cs_names = (("Cz", "POO10h", "FFT7h"), ("C3", "C1"), ("PPO10h", "POO10h", "FTT7h", "FTT7h"),
            ...                ("C3", "C1"), ("PPO10h", "POO10h", "Cz"), ("Cz",))
            >>> my_channel_name_to_index = {"Cz": 0, "C1": 1, "C3": 2, "PPO10h": 3, "POO10h": 4, "FTT7h": 5, "FFT7h": 6,
            ...                             "Fp1": 7}
            >>> my_x = torch.rand(size=(10, 13, 3000))
            >>> my_precomputed = torch.rand(size=(10, 13, 202))  # A vector of 202 features per channel
            >>> my_model = ConnectedSearch2(latent_search_features=17, in_features=my_precomputed.size()[-1],
            ...                             share_edge_embeddings=False, num_regions=len(my_cs_names),
            ...                             head_region_idx=5)
            >>> my_model(x=my_x, precomputed=my_precomputed, ch_names=my_cs_names,
            ...          channel_name_to_index=my_channel_name_to_index).size()
            torch.Size([10, 6, 3000])
            >>> my_model = ConnectedSearch2(latent_search_features=17, in_features=my_precomputed.size()[-1],
            ...                             share_edge_embeddings=True, num_regions=len(my_cs_names), head_region_idx=2)
            >>> my_model(x=my_x, precomputed=my_precomputed, ch_names=my_cs_names,
            ...          channel_name_to_index=my_channel_name_to_index).size()
            torch.Size([10, 6, 3000])
        """
        # Initialise list containing region representations
        region_representations: List[torch.Tensor] = list()

        # Compute channel indices of head region
        head_region_indices = channel_names_to_indices(channel_names=ch_names[self._head_region_idx],
                                                       channel_name_to_index=channel_name_to_index)

        # -------------------
        # Loop through all non-head regions
        # -------------------
        non_head_ch_names = list(ch_names)
        non_head_ch_names.pop(self._head_region_idx)
        for legal_ch_names, region_receptor_mapping, head_search_mapping, head_gate_mapping in \
                zip(non_head_ch_names, self._search_mappings, self._head_search_linear, self._head_search_gates):

            #  Extract indices of the nodes in the group
            allowed_node_indices = channel_names_to_indices(channel_names=legal_ch_names,
                                                            channel_name_to_index=channel_name_to_index)

            # Compute vector in search space. shape=(batch, channel, latent). If the vectors are similar to the search
            # vector (by cosine similarity), then they are more 'received', in a way. Which is the reasoning for the
            # name 'receptor vectors'
            receptor_vectors = region_receptor_mapping(precomputed[:, allowed_node_indices])

            # Compute search vector
            linear_mapping = region_receptor_mapping if self._share_edge_embeddings else head_search_mapping

            search_linear = linear_mapping(precomputed[:, head_region_indices])
            search_gate = head_gate_mapping(precomputed[:, head_region_indices])

            search_vector = torch.sum(search_linear * torch.softmax(search_gate, dim=1), dim=1, keepdim=True)

            # Compute similarities with search vector, and normalise by softmax
            normalised_similarities = torch.softmax(torch.cosine_similarity(receptor_vectors, search_vector, dim=-1),
                                                    dim=-1)

            # Compute region representation
            region_representations.append(torch.matmul(torch.unsqueeze(normalised_similarities, dim=1),
                                                       x[:, allowed_node_indices]))

        # -------------------
        # Compute region representation for head region
        # -------------------
        head_node_coefficients = self._head_node_mapping(precomputed[:, head_region_indices])
        head_node_coefficients = torch.softmax(head_node_coefficients, dim=1)

        # Insert head region representation, and return
        region_representations.insert(self._head_region_idx,
                                      torch.matmul(torch.transpose(head_node_coefficients, dim0=1, dim1=2),
                                                   x[:, head_region_indices]))

        return torch.cat(region_representations, dim=1)


# ------------------------
# Functions
# ------------------------
def channel_names_to_indices(channel_names: Tuple[str, ...], channel_name_to_index: Dict[str, int]) -> Tuple[int, ...]:
    """
    Same as channel_name_to_index, but now you can pass in a tuple of channel names
    Args:
        channel_names: Channel names to be mapped to indices
        channel_name_to_index: Object calculated from channel_name_to_index.
    Returns: The indices of the input channel names
    Examples:
        >>> my_relevant_channel_names = ("Cz", "POO10h", "FFT7h")
        >>> my_channel_name_to_index = {"Cz": 0, "C1": 1, "C3": 2, "PPO10h": 3, "POO10h": 4, "FTT7h": 5, "FFT7h": 6}
        >>> channel_names_to_indices(channel_names=my_relevant_channel_names,
        ...                          channel_name_to_index=my_channel_name_to_index)
        (0, 4, 6)
        >>> channel_names_to_indices(channel_names=tuple(my_relevant_channel_names),
        ...                          channel_name_to_index=my_channel_name_to_index)
        (0, 4, 6)
    """
    return tuple(channel_name_to_index[channel_name] for channel_name in channel_names)
