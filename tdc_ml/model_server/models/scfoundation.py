
import math
import os
from dataclasses import dataclass
from functools import partial
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from einops import rearrange, repeat
except ImportError:
    raise ImportError("einops is required for scFoundation. Install with: pip install einops")



@dataclass
class scFoundationConfig:
    """Configuration for scFoundation model."""
    # Encoder
    encoder_hidden_dim: int = 768
    encoder_depth: int = 12
    encoder_heads: int = 12
    encoder_dim_head: int = 64

    # Decoder
    decoder_hidden_dim: int = 512
    decoder_depth: int = 6
    decoder_heads: int = 8
    decoder_dim_head: int = 64

    # Embedding
    num_genes: int = 19264
    max_seq_len: int = 19266
    bin_num: int = 100
    bin_alpha: float = 1.0

    # Special tokens
    pad_token_id: int = 19264
    mask_token_id: int = 19265

    # Dropout
    ff_dropout: float = 0.0
    attn_dropout: float = 0.0




def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def cast_tuple(val):
    return (val,) if not isinstance(val, tuple) else val


def get_module_device(module):
    try:
        return next(module.parameters()).device
    except StopIteration:
        def find_tensor_attributes(module):
            tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)]
            return tuples
        gen = module._named_members(get_members_fn=find_tensor_attributes)
        first_tuple = next(gen)
        return first_tuple[1].device


def find_modules(nn_module, type):
    return [module for module in nn_module.modules() if isinstance(module, type)]





def softmax_kernel(data, *, projection_matrix, is_query, normalize_data=True, eps=1e-4, device=None):
    """Compute softmax kernel for Performer attention."""
    b, h, *_ = data.shape

    data_normalizer = (data.shape[-1] ** -0.25) if normalize_data else 1.

    ratio = (projection_matrix.shape[0] ** -0.5)

    projection = repeat(projection_matrix, 'j d -> b h j d', b=b, h=h)
    projection = projection.type_as(data)

    data_dash = torch.einsum('...id,...jd->...ij', (data_normalizer * data), projection)

    diag_data = data ** 2
    diag_data = torch.sum(diag_data, dim=-1)
    diag_data = (diag_data / 2.0) * (data_normalizer ** 2)
    diag_data = diag_data.unsqueeze(dim=-1)

    if is_query:
        data_dash = ratio * (
            torch.exp(data_dash - diag_data -
                      torch.max(data_dash, dim=-1, keepdim=True).values) + eps)
    else:
        data_dash = ratio * (
            torch.exp(data_dash - diag_data - torch.max(data_dash)) + eps)

    return data_dash.type_as(data)


def generalized_kernel(data, *, projection_matrix, kernel_fn=nn.ReLU(),
                       kernel_epsilon=0.001, normalize_data=True, device=None):
    """Compute generalized kernel for Performer attention."""
    b, h, *_ = data.shape

    data_normalizer = (data.shape[-1] ** -0.25) if normalize_data else 1.

    if projection_matrix is None:
        return kernel_fn(data_normalizer * data) + kernel_epsilon

    projection = repeat(projection_matrix, 'j d -> b h j d', b=b, h=h)
    projection = projection.type_as(data)

    data_dash = torch.einsum('...id,...jd->...ij', (data_normalizer * data), projection)

    data_prime = kernel_fn(data_dash) + kernel_epsilon
    return data_prime.type_as(data)


def orthogonal_matrix_chunk(cols, device=None):
    """Generate orthogonal matrix chunk using QR decomposition."""
    unstructured_block = torch.randn((cols, cols), device=device)
    q, r = torch.linalg.qr(unstructured_block.cpu(), mode='reduced')
    q, r = map(lambda t: t.to(device), (q, r))
    return q.t()


def gaussian_orthogonal_random_matrix(nb_rows, nb_columns, scaling=0, device=None):
    """Generate Gaussian orthogonal random matrix for random features."""
    nb_full_blocks = int(nb_rows / nb_columns)

    block_list = []

    for _ in range(nb_full_blocks):
        q = orthogonal_matrix_chunk(nb_columns, device=device)
        block_list.append(q)

    remaining_rows = nb_rows - nb_full_blocks * nb_columns
    if remaining_rows > 0:
        q = orthogonal_matrix_chunk(nb_columns, device=device)
        block_list.append(q[:remaining_rows])

    final_matrix = torch.cat(block_list)

    if scaling == 0:
        multiplier = torch.randn((nb_rows, nb_columns), device=device).norm(dim=1)
    elif scaling == 1:
        multiplier = math.sqrt((float(nb_columns))) * torch.ones((nb_rows,), device=device)
    else:
        raise ValueError(f'Invalid scaling {scaling}')

    return torch.diag(multiplier) @ final_matrix


def linear_attention(q, k, v):
    """Compute linear attention (non-causal)."""
    k_cumsum = k.sum(dim=-2)
    D_inv = 1. / torch.einsum('...nd,...d->...n', q, k_cumsum.type_as(q))
    context = torch.einsum('...nd,...ne->...de', k, v)
    out = torch.einsum('...de,...nd,...n->...ne', context, q, D_inv)
    return out





class FastAttention(nn.Module):
    """
    Performer's linear attention using random feature projections.

    Achieves O(n) complexity instead of O(n^2) for standard attention.
    """

    def __init__(
        self,
        dim_heads: int,
        nb_features: Optional[int] = None,
        ortho_scaling: int = 0,
        causal: bool = False,
        generalized_attention: bool = False,
        kernel_fn: nn.Module = nn.ReLU(),
        no_projection: bool = False
    ):
        super().__init__()
        nb_features = default(nb_features, int(dim_heads * math.log(dim_heads)))

        self.dim_heads = dim_heads
        self.nb_features = nb_features
        self.ortho_scaling = ortho_scaling

        self.create_projection = partial(
            gaussian_orthogonal_random_matrix,
            nb_rows=self.nb_features,
            nb_columns=dim_heads,
            scaling=ortho_scaling
        )
        projection_matrix = self.create_projection()
        self.register_buffer('projection_matrix', projection_matrix)

        self.generalized_attention = generalized_attention
        self.kernel_fn = kernel_fn
        self.no_projection = no_projection
        self.causal = causal

    @torch.no_grad()
    def redraw_projection_matrix(self, device):
        """Redraw random projection matrix."""
        projections = self.create_projection(device=device)
        self.projection_matrix.copy_(projections)
        del projections

    def forward(self, q, k, v, output_attentions=False):
        device = q.device

        if self.no_projection:
            q = q.softmax(dim=-1)
            k = torch.exp(k) if self.causal else k.softmax(dim=-2)
        elif self.generalized_attention:
            create_kernel = partial(
                generalized_kernel,
                kernel_fn=self.kernel_fn,
                projection_matrix=self.projection_matrix,
                device=device
            )
            q, k = map(create_kernel, (q, k))
        else:
            create_kernel = partial(
                softmax_kernel,
                projection_matrix=self.projection_matrix,
                device=device
            )
            q = create_kernel(q, is_query=True)
            k = create_kernel(k, is_query=False)

        out = linear_attention(q, k, v)

        if output_attentions:
            # Compute attention weights for visualization (expensive)
            v_diag = torch.eye(v.shape[-2]).to(device)
            v_diag = v_diag.unsqueeze(0).unsqueeze(0).repeat(v.shape[0], v.shape[1], 1, 1)
            attn_weights = torch.zeros(1, q.shape[1], q.shape[2], q.shape[2]).to('cpu').to(torch.float16)
            for head_dim in range(q.shape[1]):
                attn_weights[0, head_dim] = linear_attention(
                    q[:, head_dim].to(torch.float16),
                    k[:, head_dim].to(torch.float16),
                    v_diag[:, head_dim].to(torch.float16)
                ).detach().cpu()
            attn_weights /= q.shape[1]
            return out, attn_weights
        return out


class PreLayerNorm(nn.Module):
    """Pre-LayerNorm wrapper for transformer blocks."""

    def __init__(self, dim: int, fn: nn.Module):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    """Feed-forward network with GELU activation."""

    def __init__(
        self,
        dim: int,
        mult: int = 4,
        dropout: float = 0.,
        activation: Optional[nn.Module] = None,
        glu: bool = False
    ):
        super().__init__()
        activation = default(activation, nn.GELU)

        self.glu = glu
        self.w1 = nn.Linear(dim, dim * mult * (2 if glu else 1))
        self.act = activation()
        self.dropout = nn.Dropout(dropout)
        self.w2 = nn.Linear(dim * mult, dim)

    def forward(self, x, **kwargs):
        if not self.glu:
            x = self.w1(x)
            x = self.act(x)
        else:
            x, v = self.w1(x).chunk(2, dim=-1)
            x = self.act(x) * v

        x = self.dropout(x)
        x = self.w2(x)
        return x


class Chunk(nn.Module):
    """Chunk wrapper for processing large tensors in chunks."""

    def __init__(self, chunks: int, fn: nn.Module, along_dim: int = -1):
        super().__init__()
        self.dim = along_dim
        self.chunks = chunks
        self.fn = fn

    def forward(self, x, **kwargs):
        if self.chunks == 1:
            return self.fn(x, **kwargs)
        chunks = x.chunk(self.chunks, dim=self.dim)
        return torch.cat([self.fn(c, **kwargs) for c in chunks], dim=self.dim)


class SelfAttention(nn.Module):
    """Multi-head self-attention with Performer's fast attention."""

    def __init__(
        self,
        dim: int,
        causal: bool = False,
        heads: int = 8,
        dim_head: int = 64,
        local_heads: int = 0,
        local_window_size: int = 256,
        nb_features: Optional[int] = None,
        feature_redraw_interval: int = 1000,
        generalized_attention: bool = False,
        kernel_fn: nn.Module = nn.ReLU(),
        dropout: float = 0.,
        no_projection: bool = False,
        qkv_bias: bool = False
    ):
        super().__init__()
        assert dim % heads == 0, 'dimension must be divisible by number of heads'
        dim_head = default(dim_head, dim // heads)
        inner_dim = dim_head * heads

        self.fast_attention = FastAttention(
            dim_head,
            nb_features,
            causal=causal,
            generalized_attention=generalized_attention,
            kernel_fn=kernel_fn,
            no_projection=no_projection
        )

        self.heads = heads
        self.global_heads = heads - local_heads
        self.local_attn = None  # Local attention not used in default config

        self.to_q = nn.Linear(dim, inner_dim, bias=qkv_bias)
        self.to_k = nn.Linear(dim, inner_dim, bias=qkv_bias)
        self.to_v = nn.Linear(dim, inner_dim, bias=qkv_bias)
        self.to_out = nn.Linear(inner_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, pos_emb=None, context=None, mask=None,
                context_mask=None, output_attentions=False, **kwargs):
        b, n, _, h, gh = *x.shape, self.heads, self.global_heads

        cross_attend = exists(context)
        context = default(context, x)
        context_mask = default(context_mask, mask) if not cross_attend else context_mask

        q, k, v = self.to_q(x), self.to_k(context), self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))
        (q, lq), (k, lk), (v, lv) = map(lambda t: (t[:, :gh], t[:, gh:]), (q, k, v))

        attn_outs = []

        if q.numel() > 0:
            if exists(context_mask):
                global_mask = context_mask[:, None, :, None]
                v.masked_fill_(~global_mask, 0.)

            if output_attentions:
                out, attn_weights = self.fast_attention(q, k, v, output_attentions)
            else:
                out = self.fast_attention(q, k, v)
            attn_outs.append(out)

        if lq.numel() > 0 and self.local_attn is not None:
            assert not cross_attend, 'local attention is not compatible with cross attention'
            out = self.local_attn(lq, lk, lv, input_mask=mask)
            attn_outs.append(out)

        out = torch.cat(attn_outs, dim=1)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)

        if output_attentions:
            return self.dropout(out), attn_weights
        return self.dropout(out)





class StandardTransformerEncoder(nn.Module):
    """
    Standard transformer encoder matching the scFoundation checkpoint.

    The official checkpoint uses nn.TransformerEncoderLayer stored in a
    ModuleList called 'transformer_encoder', not Performer attention.
    """

    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        dim_feedforward: int = 3072,
        ff_dropout: float = 0.,
        attn_dropout: float = 0.
    ):
        super().__init__()
        self.transformer_encoder = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=heads,
                dim_feedforward=dim_feedforward,
                dropout=ff_dropout,
                activation='gelu',
                batch_first=True,
                norm_first=False
            )
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, padding_mask=None, **kwargs):
        for layer in self.transformer_encoder:
            x = layer(x, src_key_padding_mask=padding_mask)
        return self.norm(x)


def route_args(router, args, depth):
    """Route arguments to appropriate layers."""
    routed_args = [(dict(), dict()) for _ in range(depth)]
    matched_keys = [key for key in args.keys() if key in router]

    for key in matched_keys:
        val = args[key]
        for depth_idx, ((f_args, g_args), routes) in enumerate(zip(routed_args, router[key])):
            new_f_args, new_g_args = map(lambda route: ({key: val} if route else {}), routes)
            routed_args[depth_idx] = ({**f_args, **new_f_args}, {**g_args, **new_g_args})
    return routed_args


class SequentialSequence(nn.Module):
    """Sequential execution of transformer layers."""

    def __init__(self, layers: nn.ModuleList, args_route: Dict = {}):
        super().__init__()
        assert all(len(route) == len(layers) for route in args_route.values())
        self.layers = layers
        self.args_route = args_route

    def forward(self, x, output_attentions=False, **kwargs):
        args = route_args(self.args_route, kwargs, len(self.layers))
        layers_and_args = list(zip(self.layers, args))

        if output_attentions:
            attn_weights = []

        for (f, g), (f_args, g_args) in layers_and_args:
            if output_attentions:
                out, weights = f(x, output_attentions=output_attentions, **f_args)
                x = x + out
                attn_weights.append(weights.unsqueeze(0))
            else:
                x = x + f(x, **f_args)
            x = x + g(x, **g_args)

        if output_attentions:
            attn_weights = torch.transpose(torch.cat(attn_weights, dim=0), 0, 1)
            return x, attn_weights
        return x




class Performer(nn.Module):
    """Performer transformer with linear attention."""

    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        local_attn_heads: int = 0,
        local_window_size: int = 256,
        causal: bool = False,
        ff_mult: int = 4,
        nb_features: Optional[int] = None,
        feature_redraw_interval: int = 1000,
        reversible: bool = False,
        ff_chunks: int = 1,
        generalized_attention: bool = False,
        kernel_fn: nn.Module = nn.ReLU(),
        use_scalenorm: bool = False,
        use_rezero: bool = False,
        ff_glu: bool = False,
        ff_dropout: float = 0.,
        attn_dropout: float = 0.,
        cross_attend: bool = False,
        no_projection: bool = False,
        auto_check_redraw: bool = True,
        qkv_bias: bool = True
    ):
        super().__init__()
        layers = nn.ModuleList([])
        local_attn_heads = cast_tuple(local_attn_heads)
        local_attn_heads = local_attn_heads * depth if len(local_attn_heads) == 1 else local_attn_heads

        wrapper_fn = partial(PreLayerNorm, dim)

        for _, local_heads in zip(range(depth), local_attn_heads):
            layers.append(nn.ModuleList([
                wrapper_fn(SelfAttention(
                    dim,
                    causal=causal,
                    heads=heads,
                    dim_head=dim_head,
                    local_heads=local_heads,
                    local_window_size=local_window_size,
                    nb_features=nb_features,
                    generalized_attention=generalized_attention,
                    kernel_fn=kernel_fn,
                    dropout=attn_dropout,
                    no_projection=no_projection,
                    qkv_bias=qkv_bias
                )),
                wrapper_fn(Chunk(ff_chunks, FeedForward(dim, mult=ff_mult, dropout=ff_dropout, glu=ff_glu), along_dim=1))
            ]))

        route_attn = ((True, False),) * depth
        attn_route_map = {'mask': route_attn, 'pos_emb': route_attn}
        self.net = SequentialSequence(layers, args_route=attn_route_map)

        self.auto_check_redraw = auto_check_redraw
        self.feature_redraw_interval = feature_redraw_interval
        self.register_buffer('calls_since_last_redraw', torch.tensor(0))

    def fix_projection_matrices_(self):
        self.feature_redraw_interval = None

    def check_redraw_projections(self):
        if not self.training:
            return

        if exists(self.feature_redraw_interval) and self.calls_since_last_redraw >= self.feature_redraw_interval:
            device = get_module_device(self)
            fast_attentions = find_modules(self, FastAttention)
            for fast_attention in fast_attentions:
                fast_attention.redraw_projection_matrix(device)
            self.calls_since_last_redraw.zero_()
            return

        self.calls_since_last_redraw += 1

    def forward(self, x, output_attentions=False, **kwargs):
        if self.auto_check_redraw:
            self.check_redraw_projections()
        return self.net(x, output_attentions=output_attentions, **kwargs)


class PerformerModule(nn.Module):
    """Performer module with normalization."""

    def __init__(
        self,
        max_seq_len: int,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int = 64,
        local_attn_heads: int = 0,
        local_window_size: int = 256,
        causal: bool = False,
        ff_mult: int = 4,
        nb_features: Optional[int] = None,
        feature_redraw_interval: int = 1000,
        reversible: bool = False,
        ff_chunks: int = 1,
        ff_glu: bool = False,
        ff_dropout: float = 0.,
        attn_dropout: float = 0.,
        generalized_attention: bool = False,
        kernel_fn: nn.Module = nn.ReLU(),
        use_scalenorm: bool = False,
        use_rezero: bool = False,
        cross_attend: bool = False,
        no_projection: bool = False,
        auto_check_redraw: bool = True,
        qkv_bias: bool = True
    ):
        super().__init__()
        self.max_seq_len = max_seq_len

        self.performer = Performer(
            dim, depth, heads, dim_head,
            local_attn_heads, local_window_size, causal, ff_mult,
            nb_features, feature_redraw_interval, reversible, ff_chunks,
            generalized_attention, kernel_fn, use_scalenorm, use_rezero,
            ff_glu, ff_dropout, attn_dropout, cross_attend,
            no_projection, auto_check_redraw, qkv_bias
        )
        self.norm = nn.LayerNorm(dim)

    def check_redraw_projections(self):
        self.performer.check_redraw_projections()

    def fix_projection_matrices_(self):
        self.performer.fix_projection_matrices_()

    def forward(self, x, output_attentions=False, **kwargs):
        b, n, _, device = *x.shape, x.device
        assert n <= self.max_seq_len, f'sequence length {n} must be less than the max sequence length {self.max_seq_len}'

        if output_attentions:
            x, attn_weights = self.performer(x, output_attentions=output_attentions, **kwargs)
            x = self.norm(x)
            return x, attn_weights
        else:
            x = self.performer(x, output_attentions=output_attentions, **kwargs)
            x = self.norm(x)
            return x





class AutoDiscretizationEmbedding(nn.Module):
    """
    Soft binning embedding for continuous expression values.

    Converts continuous values to embeddings via learned soft discretization.
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int,
        bin_num: int = 100,
        bin_alpha: float = 1.0,
        mask_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None
    ):
        super().__init__()

        self.dim = dim
        self.max_seq_len = max_seq_len
        self.bin_num = bin_num
        self.bin_alpha = bin_alpha

        # MLP for soft binning
        self.mlp = nn.Linear(1, self.bin_num)
        self.mlp2 = nn.Linear(self.bin_num, self.bin_num)
        self.LeakyReLU = nn.LeakyReLU(0.1)
        self.Softmax = nn.Softmax(dim=-1)

        # Bin embeddings
        self.emb = nn.Embedding(self.bin_num, self.dim)

        # Special token embeddings
        self.emb_mask = nn.Embedding(1, self.dim)
        self.emb_pad = nn.Embedding(1, self.dim)

        self.register_buffer('bin_num_idx', torch.arange(self.bin_num))
        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id

    def forward(self, x, output_weight=False):
        """
        Args:
            x: Input tensor of shape (batch, seq_len, 1).
            output_weight: If True, also return soft binning weights.

        Returns:
            Embeddings of shape (batch, seq_len, dim).
        """
        # Find special token positions
        x_squeezed = x.squeeze(-1) if x.dim() == 3 else x
        x_mask_idx = (x_squeezed == self.mask_token_id).nonzero()
        x_pad_idx = (x_squeezed == self.pad_token_id).nonzero()

        # Ensure x is 3D for MLP
        if x.dim() == 2:
            x = x.unsqueeze(-1)

        # Soft binning MLP
        x = self.mlp(x)  # [B, N, 1] -> [B, N, bin_num]
        x = self.LeakyReLU(x)
        x_crosslayer = self.mlp2(x)
        x = self.bin_alpha * x + x_crosslayer
        weight = self.Softmax(x)  # [B, N, bin_num]

        # Get bin embeddings
        bin_num_idx = self.bin_num_idx.to(x.device)
        token_emb = self.emb(bin_num_idx)  # [bin_num, dim]

        # Weighted combination
        x = torch.matmul(weight, token_emb)  # [B, N, dim]

        # Replace mask token embeddings
        if x_mask_idx.numel() > 0:
            tensor0 = torch.tensor(0, dtype=torch.long, device=x.device)
            mask_token_emb = self.emb_mask(tensor0).type(x.dtype)
            x[x_mask_idx[:, 0], x_mask_idx[:, 1], :] = mask_token_emb.repeat(x_mask_idx.shape[0], 1)

        # Replace pad token embeddings
        if x_pad_idx.numel() > 0:
            tensor0 = torch.tensor(0, dtype=torch.long, device=x.device)
            pad_token_emb = self.emb_pad(tensor0).type(x.dtype)
            x[x_pad_idx[:, 0], x_pad_idx[:, 1], :] = pad_token_emb.repeat(x_pad_idx.shape[0], 1)

        if output_weight:
            return x, weight
        return x





class scFoundationModel(nn.Module):
    """
    scFoundation: 100M parameter single-cell foundation model.

    This model uses a MAE-style architecture with:
    - Performer encoder (12 layers, 768 dim, 12 heads)
    - Performer decoder (6 layers, 512 dim, 8 heads)
    - AutoDiscretization embedding for continuous expression values

    Reference: https://github.com/biomap-research/scFoundation
    """

    def __init__(self, config: Optional[scFoundationConfig] = None):
        super().__init__()

        self.config = config or scFoundationConfig()
        cfg = self.config

        self.max_seq_len = cfg.max_seq_len
        self.num_tokens = cfg.num_genes
        self.pad_token_id = cfg.pad_token_id
        self.mask_token_id = cfg.mask_token_id

        # Token embedding (soft binning)
        self.token_emb = AutoDiscretizationEmbedding(
            dim=cfg.encoder_hidden_dim,
            max_seq_len=cfg.max_seq_len,
            bin_num=cfg.bin_num,
            bin_alpha=cfg.bin_alpha,
            pad_token_id=cfg.pad_token_id,
            mask_token_id=cfg.mask_token_id
        )

        # Positional embedding
        self.pos_emb = nn.Embedding(cfg.max_seq_len + 1, cfg.encoder_hidden_dim)

        # Encoder — uses standard transformer attention to match the checkpoint
        self.encoder = StandardTransformerEncoder(
            dim=cfg.encoder_hidden_dim,
            depth=cfg.encoder_depth,
            heads=cfg.encoder_heads,
            dim_feedforward=cfg.encoder_hidden_dim * 4,
            ff_dropout=cfg.ff_dropout,
            attn_dropout=cfg.attn_dropout
        )

        # Decoder
        self.decoder = PerformerModule(
            max_seq_len=cfg.max_seq_len,
            dim=cfg.decoder_hidden_dim,
            depth=cfg.decoder_depth,
            heads=cfg.decoder_heads,
            dim_head=cfg.decoder_dim_head,
            ff_dropout=cfg.ff_dropout,
            attn_dropout=cfg.attn_dropout
        )

        # Decoder projection
        self.decoder_embed = nn.Linear(cfg.encoder_hidden_dim, cfg.decoder_hidden_dim, bias=True)
        self.norm = nn.LayerNorm(cfg.decoder_hidden_dim)
        self.to_final = nn.Linear(cfg.decoder_hidden_dim, 1)

    def load(self, checkpoint_path: Optional[str] = None, key: str = 'cell'):
        """
        Load model weights from checkpoint.

        Weights are stored at ./scfoundation_model/models.ckpt inside the PyTDC
        directory (same pattern as scVI's ./scvi_model/). Download the checkpoint
        manually from SharePoint and place it there before calling this method.

        Args:
            checkpoint_path: Explicit path to models.ckpt. If None, uses the loader
                             to locate ./scfoundation_model/models.ckpt.
            key: Key in checkpoint dict ('cell', 'gene', 'rde').
        """
        from tdc_ml.model_server.model_loaders.scfoundation_loader import scFoundationLoader
        loader = scFoundationLoader()
        checkpoint_path = loader.load(checkpoint_path=checkpoint_path)

        print(f"Loading scFoundation weights from {checkpoint_path}")
        model_data = torch.load(checkpoint_path, map_location='cpu')

        if key is not None and key in model_data:
            model_data = model_data[key]

        # Convert config format if needed
        if 'config' in model_data:
            config = self._convert_config(model_data['config'])
        else:
            config = {}

        # Load state dict
        if 'model_state_dict' in model_data:
            state_dict = model_data['model_state_dict']
        elif 'state_dict' in model_data:
            # Convert from training checkpoint format
            state_dict = {}
            for k, v in model_data['state_dict'].items():
                if k.startswith('model.'):
                    state_dict[k[6:]] = v
                else:
                    state_dict[k] = v
        else:
            state_dict = model_data

        # Load weights
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"Missing keys: {missing[:5]}..." if len(missing) > 5 else f"Missing keys: {missing}")
        if unexpected:
            print(f"Unexpected keys: {unexpected[:5]}..." if len(unexpected) > 5 else f"Unexpected keys: {unexpected}")

        self.eval()
        return self

    def _convert_config(self, ckpt_config: Dict) -> Dict:
        """Convert checkpoint config format."""
        config = {}

        if 'model_config' in ckpt_config:
            model_type = ckpt_config.get('model', 'mae_autobin')
            if model_type in ckpt_config['model_config']:
                config.update(ckpt_config['model_config'][model_type])

        if 'dataset_config' in ckpt_config:
            if 'rnaseq' in ckpt_config['dataset_config']:
                config.update(ckpt_config['dataset_config']['rnaseq'])

        return config

    def forward(
        self,
        x: torch.Tensor,
        padding_label: torch.Tensor,
        encoder_position_gene_ids: torch.Tensor,
        encoder_labels: torch.Tensor,
        decoder_data: torch.Tensor,
        decoder_position_gene_ids: torch.Tensor,
        decoder_data_padding_labels: torch.Tensor,
        mask_gene_name: bool = False,
        mask_labels: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **kwargs
    ) -> torch.Tensor:
        """
        Forward pass for gene expression prediction.

        Args:
            x: Encoder input data (gathered expressed genes).
            padding_label: Encoder padding mask.
            encoder_position_gene_ids: Gene position IDs for encoder.
            encoder_labels: Which positions have valid genes.
            decoder_data: Decoder input data (full sequence).
            decoder_position_gene_ids: Gene position IDs for decoder.
            decoder_data_padding_labels: Decoder padding mask.
            mask_gene_name: Whether to mask gene names.
            mask_labels: Labels for masked positions.
            output_attentions: Whether to output attention weights.

        Returns:
            Predicted expression values.
        """
        b, n, device = *x.shape, x.device
        assert n <= self.max_seq_len, f'sequence length {n} must be less than the max sequence length {self.max_seq_len}'

        # Token and positional embedding for encoder
        x = self.token_emb(torch.unsqueeze(x, 2))
        if output_attentions:
            x.requires_grad_()

        position_emb = self.pos_emb(encoder_position_gene_ids)
        x = x + position_emb
        x = self.encoder(x, padding_mask=padding_label)

        # Prepare decoder input
        decoder_data = self.token_emb(torch.unsqueeze(decoder_data, 2))
        position_emb = self.pos_emb(decoder_position_gene_ids)

        # Transfer encoder outputs to decoder
        batch_idx, gen_idx = (encoder_labels == True).nonzero(as_tuple=True)
        decoder_data[batch_idx, gen_idx] = x[~padding_label].to(decoder_data.dtype)

        decoder_data = decoder_data + position_emb
        decoder_data = self.decoder_embed(decoder_data)
        x = self.decoder(decoder_data, padding_mask=decoder_data_padding_labels)

        x = self.norm(x)

        if exists(self.to_final):
            x = self.to_final(x)
            return x.squeeze(2)

        return x

    def get_cell_embedding(
        self,
        encoder_data: torch.Tensor,
        encoder_position_gene_ids: torch.Tensor,
        encoder_padding_mask: torch.Tensor,
        pool_type: str = 'all'
    ) -> torch.Tensor:
        """
        Extract cell-level embeddings from encoder.

        Args:
            encoder_data: Gathered expression values (batch, max_expressed).
            encoder_position_gene_ids: Gene position IDs (batch, max_expressed).
            encoder_padding_mask: Padding mask (batch, max_expressed).
            pool_type: Pooling strategy:
                - 'all': Concatenate 4 pooling strategies -> (batch, 3072)
                - 'max': Max pooling -> (batch, 768)
                - 'mean': Mean pooling -> (batch, 768)

        Returns:
            Cell embeddings tensor.
        """
        self.eval()
        with torch.no_grad():
            # Token and positional embedding
            x = self.token_emb(torch.unsqueeze(encoder_data, 2).float())
            position_emb = self.pos_emb(encoder_position_gene_ids)
            x = x + position_emb

            # Encode
            gene_emb = self.encoder(x, padding_mask=encoder_padding_mask)

            if pool_type == 'all':
                # 4 pooling strategies concatenated
                emb1 = gene_emb[:, -1, :]  # Last token (target resolution)
                emb2 = gene_emb[:, -2, :]  # Second-to-last (current resolution)
                emb3, _ = torch.max(gene_emb[:, :-2, :], dim=1)  # Max over genes
                emb4 = torch.mean(gene_emb[:, :-2, :], dim=1)  # Mean over genes
                return torch.cat([emb1, emb2, emb3, emb4], dim=1)
            elif pool_type == 'max':
                return torch.max(gene_emb, dim=1)[0]
            elif pool_type == 'mean':
                # Masked mean
                mask = ~encoder_padding_mask
                mask_expanded = mask.unsqueeze(-1).float()
                return (gene_emb * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
            else:
                raise ValueError(f"Unknown pool_type: {pool_type}")

    def get_gene_embeddings(
        self,
        encoder_data: torch.Tensor,
        encoder_position_gene_ids: torch.Tensor,
        encoder_padding_mask: torch.Tensor,
        encoder_labels: torch.Tensor,
        decoder_data: torch.Tensor,
        decoder_position_gene_ids: torch.Tensor,
        decoder_padding_mask: torch.Tensor,
        num_genes: int = 19264
    ) -> torch.Tensor:
        """
        Extract gene context embeddings from decoder.

        Args:
            encoder_data: Gathered expression values.
            encoder_position_gene_ids: Encoder gene position IDs.
            encoder_padding_mask: Encoder padding mask.
            encoder_labels: Which positions have valid genes.
            decoder_data: Full expression values.
            decoder_position_gene_ids: Decoder gene position IDs.
            decoder_padding_mask: Decoder padding mask.
            num_genes: Number of genes to return embeddings for.

        Returns:
            Gene embeddings of shape (batch, num_genes, decoder_hidden_dim).
        """
        self.eval()
        # Temporarily disable final projection
        to_final_backup = self.to_final
        self.to_final = None

        with torch.no_grad():
            out = self.forward(
                x=encoder_data,
                padding_label=encoder_padding_mask,
                encoder_position_gene_ids=encoder_position_gene_ids,
                encoder_labels=encoder_labels,
                decoder_data=decoder_data,
                decoder_position_gene_ids=decoder_position_gene_ids,
                decoder_data_padding_labels=decoder_padding_mask,
                mask_gene_name=False,
                mask_labels=None
            )

        # Restore final projection
        self.to_final = to_final_backup

        return out[:, :num_genes, :].contiguous()
