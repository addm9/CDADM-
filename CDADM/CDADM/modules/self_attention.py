
from typing import Tuple, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import combinations
from itertools import product


class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention.

    Parameters
    ----------
    temperature:
        The temperature for scaling.

    attn_dropout:
        The dropout rate for the attention map.

    """

    def __init__(self, temperature: float, attn_dropout: float = 0.1,threshold_value: float = 0.7,threshold_diff: float = 0.4,mask_percentage: float = 0.3,starttime =10):
        super().__init__()
        assert temperature > 0, "temperature should be positive"
        assert attn_dropout >= 0, "dropout rate should be non-negative"
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout) if attn_dropout > 0 else None
        self.threshold_value = threshold_value
        self.threshold_diff = threshold_diff
        self.mask_percentage = mask_percentage
        self.starttime = starttime

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        epoch:int,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor,torch.Tensor]:
        """Forward processing of the scaled dot-product attention.

        Parameters
        ----------
        q:
            Query tensor.
        k:
            Key tensor.
        v:
            Value tensor.

        attn_mask:
            Masking tensor for the attention map. The shape should be [batch_size, n_heads, n_steps, n_steps].
            0 in attn_mask means values at the according position in the attention map will be masked out.

        Returns
        -------
        output:
            The result of Value multiplied with the scaled dot-product attention map.

        attn:
            The scaled dot-product attention map.

        """
        # q, k, v all have 4 dimensions [batch_size, n_heads, n_steps, d_tensor]
        # d_tensor could be d_q, d_k, d_v

        def jaccard_similarity(tensor1, tensor2):
            intersection = torch.logical_and(tensor1, tensor2)
            union = torch.logical_or(tensor1, tensor2)
            similarity = torch.sum(intersection).float() / torch.sum(union).float()
            return similarity.item()

        def mask_matrix(matrix, num_elements_to_zero):
            indices_to_zero = (matrix == 1).nonzero()
            ones_tensor = torch.ones_like(matrix)
            if indices_to_zero.size(0) > 0:
                random_indices = torch.randperm(indices_to_zero.size(0))[:num_elements_to_zero]
                selected_indices = indices_to_zero[random_indices]
                ones_tensor[selected_indices[:, 0], selected_indices[:, 1],selected_indices[:, 2]] = 0
            return ones_tensor

        def process_matrices(sub_matrices, threshold, mask_percentage):
            sub_matrices = list(sub_matrices)
            ifmask = 0
            masked_indices = set()  # 用于记录进行过mask_matrix的j的值

            # 使用 product 生成所有可能的组合
            combinations_list = list(product(range(len(sub_matrices)), repeat=2))

            for i, j in combinations_list:
                if i < j:  # 确保不对相同的矩阵进行比较和处理
                    matrix1 = sub_matrices[i]
                    matrix2 = sub_matrices[j]

                    diff_score = jaccard_similarity(matrix1, matrix2)

                    if diff_score > threshold:
                        num_elements_to_zero = int(mask_percentage * (matrix2 == 1).sum())
                        matrix2 = mask_matrix(matrix2, num_elements_to_zero)
                        sub_matrices[j] = matrix2  # 更新 sub_matrices 中的矩阵
                        ifmask = 1
                        masked_indices.add(j)  # 记录进行过 mask_matrix 的 j 的值

            # 将 sub_matrices 中没有 mask_matrix 过的都全变为1
            for idx, matrix in enumerate(sub_matrices):
                if idx not in masked_indices:
                    sub_matrices[idx] = torch.ones_like(matrix)

            sub_matrices = tuple(sub_matrices)
            return sub_matrices, ifmask

        # dot product q with k.T to obtain similarity
        attn_time = torch.matmul(q / self.temperature, k.transpose(2, 3))
        attn_variable = torch.matmul(k.transpose(2, 3) / self.temperature, v)
        # compute attention score [0, 1], then apply dropout
        attn_time = F.softmax(attn_time, dim=-1)
        #attn_weights_time = attn.squeeze(0).squeeze(0)
        attn_variable = F.softmax(attn_variable, dim=-1)

        if self.dropout is not None:
            attn_time = self.dropout(attn_time)
        if self.dropout is not None:
            attn_variable = self.dropout(attn_variable)

        if self.starttime < epoch <= 170:
            min_values_time, _ = torch.min(attn_time, dim=-1, keepdim=True)
            max_values_time, _ = torch.max(attn_time, dim=-1, keepdim=True)
            attn_mask_d_time = (attn_time - min_values_time) / (max_values_time - min_values_time)
            # 设定阈值
            min_values_variable, _ = torch.min(attn_variable, dim=-1, keepdim=True)
            max_values_variable, _ = torch.max(attn_variable, dim=-1, keepdim=True)
            attn_mask_d_variable = (attn_variable - min_values_variable) / (max_values_variable - min_values_variable)

            # 将大于阈值的元素置为1，小于等于阈值的元素置为0
            attn_mask_d_time = torch.where(attn_mask_d_time > self.threshold_value, torch.tensor(1.0), torch.tensor(0.0))
            sub_matrices_time = attn_mask_d_time.unbind(dim=1)

            attn_mask_d_variable = torch.where(attn_mask_d_variable >self.threshold_value, torch.tensor(1.0), torch.tensor(0.0))
            sub_matrices_variable = attn_mask_d_variable.unbind(dim=1)

            new_sub_matrices_time,ifmask_time = process_matrices(sub_matrices_time,self.threshold_diff, mask_percentage=self.mask_percentage)
            attn_mask_New_time= torch.stack(new_sub_matrices_time, dim=1)
            # apply masking on the attention map, this is optional
            if ifmask_time !=0 :
                attn_time = attn_time.masked_fill(attn_mask_New_time == 0, -1e9)
            new_sub_matrices_variable,ifmask_variable= process_matrices(sub_matrices_variable,self.threshold_diff, mask_percentage=self.mask_percentage)
            attn_mask_New_variable = torch.stack(new_sub_matrices_variable, dim=0)
            # apply masking on the attention map, this is optional
            if ifmask_variable !=0 :
                attn_variable = attn_variable.masked_fill(attn_mask_New_variable == 0, -1e9)

        # multiply the score with v
        output_time = torch.matmul(attn_time, v)
        output_variable = torch.matmul(q, attn_variable)
        #output = torch.cat((output_time, output_variable), dim=1)
        return output_time,output_variable,attn_time,attn_variable



class MultiHeadAttention(nn.Module):
    """Transformer multi-head attention module.

    Parameters
    ----------
    n_heads:
        The number of heads in multi-head attention.

    d_model:
        The dimension of the input tensor.

    d_k:
        The dimension of the key and query tensor.

    d_v:
        The dimension of the value tensor.

    dropout:
        The dropout rate.

    attn_dropout:
        The dropout rate for the attention map.

    """

    def __init__(
        self,
        n_steps: int,
        n_heads: int,
        d_model: int,
        d_k: int,
        d_v: int,
        dropout: float,
        attn_dropout: float,
        threshold_value: float = 0.7,
        threshold_diff: float = 0.4,
        mask_percentage: float = 0.3,
        starttime:int=10,
    ):
        super().__init__()

        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.n_steps = n_steps

        self.w_qs = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_heads * d_v, bias=False)

        self.attention = ScaledDotProductAttention(d_k**0.5, attn_dropout,threshold_value,threshold_diff,mask_percentage,starttime)
        self.fc_time = nn.Linear(n_heads * d_v, d_model, bias=False)
        self.fc_varible = nn.Linear(n_heads * n_steps, n_steps, bias=False)
        self.fc = nn.Linear(2 * d_v, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.starttime = starttime

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        epoch: int,
        attn_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor,torch.Tensor]:
        """Forward processing of the multi-head attention module.

        Parameters
        ----------
        q:
            Query tensor.

        k:
            Key tensor.

        v:
            Value tensor.

        attn_mask:
            Masking tensor for the attention map. The shape should be [batch_size, n_heads, n_steps, n_steps].
            0 in attn_mask means values at the according position in the attention map will be masked out.

        Returns
        -------
        v:
            The output of the multi-head attention layer.

        attn_weights:
            The attention map.

        """
        # the input q, k, v currently have 3 dimensions [batch_size, n_steps, d_tensor]
        # d_tensor could be n_heads*d_k, n_heads*d_v

        # keep useful variables
        batch_size, n_steps,n_features = q.size(0), q.size(1),q.size(2)
        residual = q

        # now separate the last dimension of q, k, v into different heads -> [batch_size, n_steps, n_heads, d_k or d_v]
        q = self.w_qs(q).view(batch_size, n_steps, self.n_heads, self.d_k)
        k = self.w_ks(k).view(batch_size, n_steps, self.n_heads, self.d_k)
        v = self.w_vs(v).view(batch_size, n_steps, self.n_heads, self.d_v)

        # transpose for self-attention calculation -> [batch_size, n_steps, d_k or d_v, n_heads]
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        v_time,v_variable,attn_weights_time,attn_weights_variable= self.attention(q, k, v,epoch)

        # transpose back -> [batch_size, n_steps, n_heads, d_v]
        # then merge the last two dimensions to combine all the heads -> [batch_size, n_steps, n_heads*d_v]
        v_time = v_time.transpose(1, 2).contiguous().view(batch_size, n_steps, -1)
        v_variable = v_variable.transpose(1, 2).contiguous().view(batch_size,-1,n_features)
        v_variable = v_variable.transpose(1, 2)
        v_time = self.fc_time(v_time)
        v_variable = self.fc_varible(v_variable)
        v_variable = v_variable.transpose(1, 2)
        v = torch.cat((v_time, v_variable), dim=2)
        v = self.fc(v)

        # apply dropout and residual connection
        v = self.dropout(v)
        v += residual

        # apply layer-norm
        v = self.layer_norm(v)

        return v, attn_weights_time,attn_weights_variable

class PositionWiseFeedForward(nn.Module):
    """Position-wise feed forward network (FFN) in Transformer.

    Parameters
    ----------
    d_in:
        The dimension of the input tensor.

    d_hid:
        The dimension of the hidden layer.

    dropout:
        The dropout rate.

    """

    def __init__(self, d_in: int, d_hid: int, dropout: float = 0.1):
        super().__init__()
        self.linear_1 = nn.Linear(d_in, d_hid)
        self.linear_2 = nn.Linear(d_hid, d_in)
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward processing of the position-wise feed forward network.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        x:
            Output tensor.
        """
        # save the original input for the later residual connection
        residual = x
        # the 1st linear processing and ReLU non-linear projection
        x = F.relu(self.linear_1(x))
        # the 2nd linear processing
        x = self.linear_2(x)
        # apply dropout
        x = self.dropout(x)
        # apply residual connection
        x += residual
        # apply layer-norm
        x = self.layer_norm(x)
        return x


class EncoderLayer(nn.Module):
    """Transformer encoder layer.

    Parameters
    ----------
    d_model:
        The dimension of the input tensor.

    d_inner:
        The dimension of the hidden layer.

    n_heads:
        The number of heads in multi-head attention.

    d_k:
        The dimension of the key and query tensor.

    d_v:
        The dimension of the value tensor.

    dropout:
        The dropout rate.

    attn_dropout:
        The dropout rate for the attention map.
    """

    def __init__(
        self,
        n_steps:int,
        d_model: int,
        d_inner: int,
        n_heads: int,
        d_k: int,
        d_v: int,
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
        threshold_value: float = 0.7,
        threshold_diff:float = 0.4,
        mask_percentage:float = 0.3,
        starttime:int=10,
    ):
        super().__init__()
        self.slf_attn = MultiHeadAttention(
            n_steps,n_heads, d_model, d_k, d_v, dropout, attn_dropout,threshold_value,threshold_diff,mask_percentage,starttime
        )
        self.pos_ffn = PositionWiseFeedForward(d_model, d_inner, dropout)

    def forward(
        self,
        enc_input: torch.Tensor,
        epoch: int,
        src_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor,torch.Tensor]:
        """Forward processing of the encoder layer.

        Parameters
        ----------
        enc_input:
            Input tensor.

        src_mask:
            Masking tensor for the attention map. The shape should be [batch_size, n_heads, n_steps, n_steps].

        Returns
        -------
        enc_output:
            Output tensor.

        attn_weights:
            The attention map.

        """
        enc_output, attn_weights_time,attn_weights_variable = self.slf_attn(
            enc_input,
            enc_input,
            enc_input,
            epoch,
            attn_mask=src_mask,
        )
        enc_output = self.pos_ffn(enc_output)
        return enc_output,attn_weights_time,attn_weights_variable



class PositionalEncoding(nn.Module):
    """Positional-encoding module for Transformer.

    Parameters
    ----------
    d_hid:
        The dimension of the hidden layer.

    n_position:
        The number of positions.

    """

    def __init__(self, d_hid: int, n_position: int = 200):
        super().__init__()
        # Not a parameter
        self.register_buffer(
            "pos_table", self._get_sinusoid_encoding_table(n_position, d_hid)
        )

    @staticmethod
    def _get_sinusoid_encoding_table(n_position: int, d_hid: int) -> torch.Tensor:
        """Sinusoid position encoding table"""

        def get_position_angle_vec(position):
            return [
                position / np.power(10000, 2 * (hid_j // 2) / d_hid)
                for hid_j in range(d_hid)
            ]

        sinusoid_table = np.array(
            [get_position_angle_vec(pos_i) for pos_i in range(n_position)]
        )
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward processing of the positional encoding module.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        x:
            Output tensor, the input tensor with the positional encoding added.

        """
        return x + self.pos_table[:, : x.size(1)].clone().detach()
