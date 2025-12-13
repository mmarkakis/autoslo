from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.nn.utils.rnn import (
    pack_padded_sequence,
    pad_packed_sequence,
    pad_sequence,
)


def xavier_init(m: nn.Module) -> None:
    if isinstance(m, nn.Module):
        for name, param in m.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param.data)
            elif "bias" in name:
                nn.init.constant_(param.data, 0.0)


class IconqRuntimeNet(nn.Module):
    def __init__(
        self,
        input_size: int,
        embedding_dim: int,
        hidden_size: int,
        output_size: int,
        num_layers: int,
        dropout: float = 0.1,
    ) -> None:
        super(IconqRuntimeNet, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_dim = embedding_dim
        self.bn = nn.BatchNorm1d(input_size)
        print(f"BN: {self.bn}")
        self.embedding = nn.Sequential(
            nn.Linear(input_size, embedding_dim),
            nn.Linear(embedding_dim, embedding_dim),
        )
        xavier_init(self.embedding)
        self.num_layers = num_layers
        self.model = nn.LSTM(
            embedding_dim,
            hidden_size,
            num_layers,
            dropout=dropout,
            batch_first=True,
        )
        xavier_init(self.model)
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Linear(hidden_size // 2, output_size),
        )

        xavier_init(self.output_layer)

    def model_forward(
        self,
        x: torch.Tensor,
        x_len: torch.Tensor,
        h0: Optional[torch.Tensor] = None,
        c0: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

        if x.shape[1] > 1:
            x = torch.transpose(x, 1, 2)
            x = self.bn(x)
            x = torch.transpose(x, 1, 2)
        x = self.embedding(x)
        packed_input = pack_padded_sequence(
            x, x_len, batch_first=True, enforce_sorted=False
        )
        if h0 is None:
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(
                x.device
            )
        if c0 is None:
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(
                x.device
            )
        output, (hn, cn) = self.model(packed_input, (h0, c0))
        output, _ = pad_packed_sequence(output, batch_first=True)
        output = output[torch.arange(len(x_len)), x_len - 1]
        output = self.output_layer(output)
        return output, (hn, cn)

    def forward(
        self, x: torch.Tensor, x_len: torch.Tensor, pinch_points: torch.Tensor
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:

        pre_info_x = []
        avg_rt_list = []
        post_info_x = []
        post_info_len = []
        zero_idx = []
        non_zero_idx = []
        for i in range(len(x)):
            pre_info_l = pinch_points[i] + 1
            pre_info_x.append(x[i, :pre_info_l, :])
            avg_rt_list.append(float(x[i][0][0]))
            post_info_l = int(x_len[i]) - pre_info_l
            if post_info_l <= 0:
                zero_idx.append(i)
            else:
                non_zero_idx.append(i)
                post_info_x.append(x[i, pre_info_l:, :])
                post_info_len.append(post_info_l)
        # pad pre and post info
        avg_rt = torch.tensor(avg_rt_list, requires_grad=False).reshape(-1, 1)
        pre_seq_lengths = torch.tensor(
            [len(x) for x in pre_info_x], dtype=torch.long
        )
        padded_pre_info_x = pad_sequence(
            pre_info_x, batch_first=True, padding_value=0
        )
        y_prime, (hn, cn) = self.model_forward(
            padded_pre_info_x, pre_seq_lengths
        )
        y_prime = y_prime * avg_rt / 3
        if len(non_zero_idx) == 0:
            # very unlikely that a whole batch has no post info
            return y_prime, None, None

        hn = hn[:, non_zero_idx, :]
        cn = cn[:, non_zero_idx, :]

        new_post_info_x = []
        for i in range(len(non_zero_idx)):
            curr_post_info_x = post_info_x[i].clone()
            for j in range(len(curr_post_info_x)):
                curr_post_info_x[j, 0] = y_prime[non_zero_idx[i]]
            new_post_info_x.append(curr_post_info_x)

        post_seq_lengths = torch.tensor(post_info_len, dtype=torch.long)
        padded_post_info_x = pad_sequence(
            new_post_info_x, batch_first=True, padding_value=0
        )
        y, _ = self.model_forward(padded_post_info_x, post_seq_lengths, hn, cn)
        y = y * y_prime[non_zero_idx]
        output = torch.zeros((len(y_prime), 1), requires_grad=False)
        if len(zero_idx) != 0:
            output[zero_idx] = y_prime[zero_idx]

        output[non_zero_idx] = y

        return output, None, None
