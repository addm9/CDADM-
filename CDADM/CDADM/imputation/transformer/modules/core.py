"""
The implementation of Transformer for the partially-observed time-series imputation task.

Refer to the paper "Du, W., Cote, D., & Liu, Y. (2023). SAITS: Self-Attention-based Imputation for Time Series.
Expert systems with applications."

Notes
-----
Partial implementation uses code from https://github.com/WenjieDu/SAITS.

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: GPL-v3

from typing import Tuple, Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from ....modules.self_attention import EncoderLayer, PositionalEncoding
from ....utils.metrics import cal_mae


class _TransformerEncoder(nn.Module):
    def __init__(
        self,
        n_layers: int,
        n_steps:int,
        d_feature: int,
        d_model: int,
        d_inner: int,
        n_heads: int,
        d_k: int,
        d_v: int,
        epochs: int,
        dropout: float,
        attn_dropout: float,
        ORT_weight: float = 1,
        MIT_weight: float = 1,
        threshold_value: float = 0.7,
        threshold_diff: float = 0.4,
        mask_percentage: float = 0.3,
        starttime: int = 10
    ):
        super().__init__()
        self.n_layers = n_layers
        actual_d_feature = d_feature * 2
        self.ORT_weight = ORT_weight
        self.MIT_weight = MIT_weight
        self.epochs =epochs
        self.n_steps = n_steps

        self.layer_stack = nn.ModuleList(
            [
                EncoderLayer(
                    n_steps,
                    d_model,
                    d_inner,
                    n_heads,
                    d_k,
                    d_v,
                    dropout,
                    attn_dropout,
                    threshold_value,
                    threshold_diff,
                    mask_percentage,
                    starttime
                )
                for _ in range(n_layers)
            ]
        )

        self.embedding = nn.Linear(actual_d_feature, d_model)
        self.position_enc = PositionalEncoding(d_model, n_position=n_steps)
        self.dropout = nn.Dropout(p=dropout)
        self.reduce_dim = nn.Linear(d_model, d_feature)

    def _process(self,inputs: dict,epoch: int,attention_mask: Optional[torch.Tensor] = None,) -> Tuple[torch.Tensor,list,torch.Tensor,torch.Tensor]:
        X, masks = inputs["X"], inputs["missing_mask"]
        input_X = torch.cat([X, masks], dim=2)
        input_X = self.embedding(input_X)
        enc_output = self.dropout(self.position_enc(input_X))

        for encoder_layer in self.layer_stack:
            enc_output,Weight_time,Weight_variable= encoder_layer(enc_output,epoch,attention_mask)

        learned_presentation = self.reduce_dim(enc_output)
        imputed_data = (
            masks * X + (1 - masks) * learned_presentation
        )  # replace non-missing part with original data
        return imputed_data, learned_presentation,Weight_time,Weight_variable

    def forward(self, inputs: dict,epoch: int, training: bool = True) -> dict:
        X, masks = inputs["X"], inputs["missing_mask"]
        #Mask = (1 - torch.eye(self.n_steps)).to(X.device)
        # then broadcast on the batch axis
        #Mask = Mask.unsqueeze(0)

        imputed_data, learned_presentation, Weight_time, Weight_variable= self._process(inputs,epoch)

        if not training:
            # if not in training mode, return the classification result only
            return {
                "imputed_data": imputed_data,
            }

        ORT_loss = cal_mae(learned_presentation, X, masks)
        MIT_loss = cal_mae(
            learned_presentation, inputs["X_intact"], inputs["indicating_mask"]
        )

        # `loss` is always the item for backward propagating to update the model
        loss = self.ORT_weight * ORT_loss + self.MIT_weight * MIT_loss

        results = {
            "imputed_data": imputed_data,
            "ORT_loss": ORT_loss,
            "MIT_loss": MIT_loss,
            "loss": loss,
        }
        return results,Weight_time, Weight_variable
