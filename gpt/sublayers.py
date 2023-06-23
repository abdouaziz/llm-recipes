import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from typing import Tuple, Optional

import math

from utils import compute_mask_indices
from dataclasses import dataclass


@dataclass
class Config:
    def __init__(self):
        self.vocab_size = 3020
        self.model_dim = 512
        self.num_heads = 8
        self.num_layers = 6
        self.ff_dim = 2048
        self.dropout = 0.1
        self.batch_size = 32
        self.epochs = 10
        self.learning_rate = 0.001
        self.num_workers = 4
        self.output_dim = 35
        self.seed = 1
        self.log_interval = 10
        self.save_model = True
        self.save_model_path = "model.pt"
        self.cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.cuda else "cpu")

    def __str__(self):
        return str(self.__dict__)


class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention"""

    def __init__(self, model_dim, attn_dropout=0.1) -> None:
        super().__init__()
        self.sqrt_d_model = model_dim
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        """
        Args:
            q: [batch_size, len_q, d_k]
            k: [batch_size, len_k, d_k]
            v: [batch_size, len_k, d_v]
            mask: [batch_size, len_q, len_k]
        Returns:
            context: [batch_size, len_q, d_v]
            attn: [batch_size, len_q, len_k]
        """
        attn = torch.bmm(q, k.transpose(1, 2)) / self.sqrt_d_model

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))

        context = torch.bmm(attn, v)

        return context, attn


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention module"""

    def __init__(self, model_dim, n_head, dropout=0.1) -> None:
        super().__init__()
        self.n_head = n_head
        self.d_k = model_dim // n_head
        self.d_v = model_dim // n_head
        self.w_qs = nn.Linear(model_dim, n_head * self.d_k)
        self.w_ks = nn.Linear(model_dim, n_head * self.d_k)
        self.w_vs = nn.Linear(model_dim, n_head * self.d_v)
        self.fc = nn.Linear(n_head * self.d_v, model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        """
        Args:
            q: [batch_size, len_q, d_model]
            k: [batch_size, len_k, d_model]
            v: [batch_size, len_v, d_model]
            mask: [batch_size, len_q, len_k]

        Returns:
            context: [batch_size, len_q, d_model]
            attn: [batch_size, n_head, len_q, len_k]
        """
        d_k, d_v = self.d_k, self.d_v
        n_head = self.n_head
        # residual = q

        batch_size, len_q, d_model = q.size()
        batch_size, len_k, d_model = k.size()
        batch_size, len_v, d_model = v.size()

        q = self.w_qs(q).view(batch_size, len_q, n_head, d_k)
        k = self.w_ks(k).view(batch_size, len_k, n_head, d_k)
        v = self.w_vs(v).view(batch_size, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv

        if mask is not None:
            mask = mask.repeat(n_head, 1, 1)

        output, attn = ScaledDotProductAttention(model_dim=d_model)(q, k, v, mask=mask)
        output = output.view(n_head, batch_size, len_q, d_v)
        output = (
            output.permute(1, 2, 0, 3).contiguous().view(batch_size, len_q, -1)
        )  # b x lq x (n*dv)
        output = self.dropout(self.fc(output))
        # output = output + residual

        return output, attn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int = 512, max_len: int = 5000) -> None:
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model, requires_grad=False)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, length: int) -> Tensor:
        return self.pe[:, :length, :]


class PositionWiseFeedForward(nn.Module):
    """
    Implement position-wise feed forward layer.
    FFN(x) = max(0, xW1 + b1)W2 + b2
    """

    def __init__(
        self,
        model_dim: int = 512,
        ff_dim: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        super(PositionWiseFeedForward, self).__init__()
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(ff_dim, model_dim),
            nn.Dropout(p=dropout),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.feed_forward(inputs)


class LayerNorm(nn.Module):
    def __init__(
        self,
        model_dim: int,
        eps: float = 1e-6,
    ) -> None:
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(model_dim))
        self.b_2 = nn.Parameter(torch.zeros(model_dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class DecoderLayer(nn.Module):
    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout: float,
    ) -> None:
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(model_dim, num_heads)
        self.layer_norm = LayerNorm(model_dim)
        self.feed_forward = PositionWiseFeedForward(model_dim, ff_dim, dropout)

    def forward(
        self,
        encodings: Tensor,
        mask: Optional[Tensor] = None,
    ):
        context, _ = self.self_attn(encodings, encodings, encodings, mask)

        mask_multihead = compute_mask_indices(context, mask_prob=0.02, mask_length=10)

        context_masked = context.masked_fill(mask_multihead.unsqueeze(-1), 0)

        output_norm = self.layer_norm(encodings + context_masked)

        output_ff = self.feed_forward(output_norm)

        output = self.layer_norm(output_norm + output_ff)

        return output


class Decoder(nn.Module):
    def __init__(
        self,
        model_dim: int,
        num_layers: int,
        num_heads: int,
        ff_dim: int,
        dropout: float,
    ) -> None:
        super(Decoder, self).__init__()
        self.num_layers = num_layers
        self.pos_encoding = PositionalEncoding(model_dim)
        self.layers = nn.ModuleList(
            [
                DecoderLayer(model_dim, num_heads, ff_dim, dropout)
                for _ in range(self.num_layers)
            ]
        )

    def forward(
        self,
        encodings: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        length = encodings.size(1)
        out = encodings + self.pos_encoding(length)
        for i in range(self.num_layers):
            out = self.layers[i](out, mask)
        return out


class GPT(nn.Module):
    def __init__(self,config=Config() , n_classes=10) :
        super(GPT, self).__init__()

        self.decoder = Decoder(model_dim=config.model_dim,
                               num_layers=config.num_layers,
                               num_heads=config.num_heads,
                               ff_dim=config.ff_dim,
                               dropout=config.dropout)
        
        self.generation = nn.Linear(config.model_dim, config.vocab_size)

        self.classification = nn.Linear(config.model_dim, n_classes)

    def forward(self, input):
        output = self.decoder(input)

        logits = self.generation(output)
        prediction = self.classification(output)

        return logits, prediction




     
if __name__ == "__main__":

    config = Config()

    print(config , "\n")

    x = torch.rand(4, 512, 512)

    multihead = GPT(
        config=config,
        n_classes=10
    )

    logits, prediction = multihead(x)

    print("logits shape ", logits.shape)
    print("class prediction shape ", prediction)
