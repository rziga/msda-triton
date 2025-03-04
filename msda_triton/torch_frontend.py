import warnings

import torch
from torch import nn
from torch.nn import functional as F

try:
    from .autograd_function import triton_multiscale_deformable_attention
except ModuleNotFoundError:
    warnings.warn("Could not import custom kernel. Please make sure triton is installed.")


def native_multiscale_deformable_attention(
    img: torch.Tensor,
    img_shapes: torch.Tensor,
    sampling_points: torch.Tensor,
    attention_weights: torch.Tensor,
) -> torch.Tensor:
    B, I, H, C = img.shape
    B, N, H, L, P, _ = sampling_points.shape

    # split the image into levels
    img_levels = img.split_with_sizes(img_shapes.prod(-1).tolist(), dim=1)

    # normalize points to from [0, 1] to [-1, 1]
    sampling_points = 2 * sampling_points - 1

    samples = []
    for img_level, points_level, (h, w) in zip(
        img_levels, sampling_points.unbind(-3), img_shapes
    ):
        # reshape for sampling
        img_level = (
            img_level # [B, I, H, C]
            .permute(0, 2, 3, 1) # [B, H, C, I]
            .reshape(B*H, C, h, w) # [B*H, C, H, W]
        )
        points_level = (
            points_level # [B, N, H, P, 2]
            .permute(0, 2, 1, 3, 4) # [B, H, N, P, 2]
            .reshape(B*H, N, P, 2) # [B*H, N, P, 2]
        )

        # sample
        samples_level = F.grid_sample(
            img_level, points_level,
            mode="bilinear", align_corners=True, padding_mode="border"
        )
        samples_level = (
            samples_level # [B*H, C, N, P]
            .reshape(B, H, C, N, P) # [B, H, C, N, P]
            .permute(0, 3, 1, 4, 2) # [B, N, H, P, C]
        )
        samples.append(samples_level)
    # [B, N, H, L, P, C]
    samples = torch.stack(samples, dim=3)
    # [B, N, H, C]
    out = torch.sum(attention_weights[..., None] * samples, dim=(3, 4))

    return out


def multiscale_deformable_attention(
    img: torch.Tensor,
    img_shapes: torch.Tensor,
    sampling_points: torch.Tensor,
    attention_weights: torch.Tensor,
) -> torch.Tensor:
    """
    Differentiable multiscale deformable attention function.

    Args:
        img (torch.Tensor): Flattened image pyramid tensor of shape `[batch_size, num_image, num_head, num_channel]`, where `num_image=sum(h[i]*w[i] for i in range(levels))`
        img_shapes (torch.Tensor): Shapes of each pyramid level tensor of shape `[num_levels, 2]`, in (height, width) order.
        sampling_points (torch.Tensor): Sampling points tensor of shape `[batch_size, num_queries, num_heads, num_levels, num_points, 2]`, in (x, y) order, where x and y \in [0-1]. 
        attention_weights (torch.Tensor): Attention weights tensor of shape `[batch_size, num_queries, num_heads, num_levels, num_points]`.
    
    Returns:
        output (torch.Tensor): Output tensor of shape `[batch_size, num_queries, num_heads, num_channels]`.
    """
    try:
        return triton_multiscale_deformable_attention(img, img_shapes, sampling_points, attention_weights)
    except Exception:
        return native_multiscale_deformable_attention(img, img_shapes, sampling_points, attention_weights)


class MultiscaleDeformableAttention(nn.Module):

    def __init__(self, emb_dim, hidden_dim, num_levels, num_heads, num_points):
        super().__init__()

        if hidden_dim % num_heads != 0:
            raise ValueError(f"Hidden dimension ({hidden_dim=}) should be divisible by number of heads ({num_heads=}).")

        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.img_input_proj = nn.Linear(emb_dim, hidden_dim)
        self.query_input_proj = nn.Linear(emb_dim, num_heads*num_levels*num_points*3)
        self.query_output_proj = nn.Linear(hidden_dim, emb_dim)

    def forward(
        self,
        img: torch.Tensor, 
        shapes: torch.Tensor,
        queries: torch.Tensor,
        reference_points: torch.Tensor,
    ) -> torch.Tensor:
        """Multiscale deformable attention forward pass.
        
        Project the queries to get offsets and attention weights.
        Add offsets to reference points to obtain sampling points.
        Perform multiscale deformable attention like described in https://arxiv.org/abs/2010.04159.

        Args:
            img (torch.Tensor): Flattened image pyramid tensor of shape `[batch_size, num_image, num_channels]`, where `num_image` is the total pixel count for all levels.
            shapes (torch.Tensor): 2D shapes of the feature pyramid levels of shape `[num_levels, 2]`.
            queries (torch.Tensor): Latent queries of shape `[batch_size, num_queries, num_channels]` which are projected for sampling offsets and attention weights.
            reference_points (torch.Tensor): Positions of queries of shape `[batch_size, num_queries, 2]` or `[batch_size, num_queries, 4]`.

        Returns:
            output (torch.Tensor): Samples aggregated based on attention weights of shape `[batch_size, num_queries, num_channels]`.
        """
        B, I, C = img.shape
        B, N, C = queries.shape
        L, H, P = self.num_levels, self.num_heads, self.num_points

        # project image
        img = (
            self.img_input_proj(img)
            .reshape(B, I, H, C//H)
        )

        # project queries to get offsets and weights
        offsets, attention_weights = (
            self.query_input_proj(queries)
            .reshape(B, N, H, L, P, 3)
            .split_with_sizes((2, 1), dim=-1)
        )
        attention_weights = torch.nn.functional.softmax(
            attention_weights.reshape(B, N, H, L*P), dim=-1).reshape(B, N, H, L, P)

        # calculate sampling points
        # [B, N, 1, 1, 1, 2] + [B, N, H, L, P, 2] -> [B, N, H, L, P, 2]
        last_dim = reference_points.shape[-1]
        if last_dim == 2:
            sampling_points = reference_points[:, :, None, None, None, :] + offsets
        elif last_dim == 4:
            pass
        else:
            raise ValueError(f"`reference_points` should have the last dim either 2 or 4, but got {last_dim}.")

        # deformable attention
        # [B, N, H, C//H] -> [B, N, C]
        out = multiscale_deformable_attention(img, shapes, sampling_points, attention_weights)
        out = out.reshape(B, N, C)
        out = self.query_output_proj(out)
        return out
