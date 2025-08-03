from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd.function import Function, once_differentiable
from torch.amp import custom_fwd, custom_bwd

from .kernels import (
    triton_multi_scale_deformable_attention_fwd,
    triton_multi_scale_deformable_attention_bwd,
)


def native_multiscale_deformable_attention(
    img: torch.Tensor,
    img_shapes: torch.Tensor,
    sampling_points: torch.Tensor,
    attention_weights: torch.Tensor,
    padding_mode: Literal["border", "zeros"],
    align_corners: bool,
) -> torch.Tensor:
    """
    Fallback torch implementation.
    """

    B, I, H, C = img.shape  # noqa: E741
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
            mode="bilinear", align_corners=align_corners, padding_mode=padding_mode
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


def triton_multiscale_deformable_attention(
    img: torch.Tensor,
    img_shapes: torch.Tensor,
    sampling_points: torch.Tensor,
    attention_weights: torch.Tensor,
    padding_mode: Literal["border", "zeros"],
    align_corners: bool,
) -> torch.Tensor:
    """
    Triton implementation.
    """

    # check input dtypes
    valid_dtypes = [torch.float16, torch.float32, torch.float64]
    if img.dtype not in valid_dtypes:
        raise ValueError(f"Dtype of `img` should be in {valid_dtypes}, but got {img.dtype}.")
    if sampling_points.dtype not in valid_dtypes:
        raise ValueError(f"Dtype of `sampling_points` should be in {valid_dtypes}, but got {sampling_points.dtype}.")
    if attention_weights.dtype not in valid_dtypes:
        raise ValueError(f"Dtype of `attention_weights` should be in {valid_dtypes}, but got {attention_weights.dtype}.")

    # check input devices
    devices = [inpt.device for inpt in [img, img_shapes, sampling_points, attention_weights]]
    if any(device.type != "cuda" for device in devices):
        raise ValueError(f"Expected all inputs to be on gpu, but got {devices}.")

    # run it
    return _TritonMultiscaleDeformableAttentionFunction.apply(
        img,
        img_shapes,
        sampling_points,
        attention_weights,
        padding_mode,
        align_corners,
    )


class _TritonMultiscaleDeformableAttentionFunction(Function):

    @staticmethod
    @custom_fwd(device_type="cuda", cast_inputs=torch.float32)
    def forward(
        ctx, 
        img: torch.Tensor,
        img_shapes: torch.Tensor,
        sampling_points: torch.Tensor,
        attention_weights: torch.Tensor,
        padding_mode: Literal["border", "zeros"],
        align_corners: bool,
    ) -> torch.Tensor:
        ctx.save_for_backward(img, img_shapes, sampling_points, attention_weights)
        ctx.padding_mode = padding_mode
        ctx.align_corners = align_corners
        out = triton_multi_scale_deformable_attention_fwd(
            img, img_shapes, sampling_points, attention_weights, padding_mode, align_corners
        )
        return out

    @staticmethod
    @once_differentiable
    @custom_bwd(device_type="cuda")
    def backward(
        ctx, 
        out_grad: torch.Tensor,
    ) -> tuple[torch.Tensor, None, torch.Tensor, torch.Tensor]:
        img, img_shapes, sampling_points, attention_weights = ctx.saved_tensors
        padding_mode = ctx.padding_mode
        align_corners = ctx.align_corners
        img_grad, sampling_points_grad, attention_weights_grad = triton_multi_scale_deformable_attention_bwd(
            out_grad, img, img_shapes, sampling_points, attention_weights, padding_mode, align_corners
        )
        return img_grad, None, sampling_points_grad, attention_weights_grad, None, None


def multiscale_deformable_attention(
    img: torch.Tensor,
    img_shapes: torch.Tensor,
    sampling_points: torch.Tensor,
    attention_weights: torch.Tensor,
    padding_mode: Literal["border", "zeros"],
    align_corners: bool,
) -> torch.Tensor:
    """
    Differentiable multiscale deformable attention function.

    Args:
        img (torch.Tensor): Flattened image pyramid tensor of shape `[batch_size, num_image, num_head, num_channel]`, where `num_image=sum(h[i]*w[i] for i in range(levels))`
        img_shapes (torch.Tensor): Shapes of each pyramid level tensor of shape `[num_levels, 2]`, in (height, width) order.
        sampling_points (torch.Tensor): Sampling points tensor of shape `[batch_size, num_queries, num_heads, num_levels, num_points, 2]`, in (x, y) order, where x and y are normalized to [0, 1] and (0, 0) is top-left corner and (1, 1) is bottom right corner. 
        attention_weights (torch.Tensor): Attention weights tensor of shape `[batch_size, num_queries, num_heads, num_levels, num_points]`.
        padding_mode (Literal["border", "zeros"]): Determines how to handle out-of-bounds (OOB) samples. `border` sets the OOB samples to closest image pixel; `zeros` sets the OOB samples to 0.
        align_corners (bool): Determines the grid alignment of the image. See: https://discuss.pytorch.org/t/what-we-should-use-align-corners-false/22663/9 for illustration.

    Returns:
        output (torch.Tensor): Output tensor of shape `[batch_size, num_queries, num_heads, num_channels]`.
    """
    try:
        return triton_multiscale_deformable_attention(
            img, img_shapes, sampling_points, attention_weights, padding_mode, align_corners)
    except Exception:
        return native_multiscale_deformable_attention(
            img, img_shapes, sampling_points, attention_weights, padding_mode, align_corners)


class MultiscaleDeformableAttention(nn.Module):
    """
    Multiscale deformable attention module.

    It handles input and output projections along with performing the actual multiscale deformable attention operation.
    See: Figure 2. in https://arxiv.org/pdf/2010.04159 for illustration.

    Args:
        emb_dim (int): Feature dimension of inputs.
        hidden_dim (int): Feature dimension to which to project. Must be divisible by `num_heads`.
        num_levels (int): Number of feature levels of input images.
        num_heads (int): Number of attention heads. Must divide `hidden_dim`.
        num_points (int): Number of points.
        padding_mode (Literal["border", "zeros"]): Determines how to handle out-of-bounds (OOB) samples. `border` sets the OOB samples to closest image pixel; `zeros` sets the OOB samples to 0.
        align_corners (bool): Determines the grid alignment of the image. See: https://discuss.pytorch.org/t/what-we-should-use-align-corners-false/22663/9 for illustration.

    Returns:
        output (torch.Tensor): Output tensor of shape `[batch_size, num_queries, num_heads, num_channels]`.

    Raises:
        ValueError: If `hidden_dim` is not divisible by `num_heads`.
    """


    def __init__(
        self,
        emb_dim: int,
        hidden_dim: int,
        num_levels: int,
        num_heads: int,
        num_points: int,
        padding_mode: Literal["border", "zeros"],
        align_corners: bool,
    ):
        super().__init__()

        if hidden_dim % num_heads != 0:
            raise ValueError(f"Hidden dimension ({hidden_dim=}) should be divisible by number of heads ({num_heads=}).")

        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.hidden_dim = hidden_dim
        self.img_input_proj = nn.Linear(emb_dim, hidden_dim)
        self.query_input_proj = nn.Linear(emb_dim, num_heads*num_levels*num_points*3)
        self.query_output_proj = nn.Linear(hidden_dim, emb_dim)

        self.padding_mode = padding_mode
        self.align_corners = align_corners

    def forward(
        self,
        img: torch.Tensor, 
        img_shapes: torch.Tensor,
        queries: torch.Tensor,
        reference_points: torch.Tensor,
    ) -> torch.Tensor:
        """Multiscale deformable attention forward pass.
        
        Project the queries to get offsets and attention weights.
        Add offsets to reference points to obtain sampling points.
        Perform multiscale deformable attention like described in https://arxiv.org/abs/2010.04159.

        Args:
            img (torch.Tensor): Flattened image pyramid - tensor of shape `[batch_size, num_image, num_channels]`, where `num_image` is the total pixel count for all levels.
            img_shapes (torch.Tensor): 2D shapes of the feature pyramid levels - tensor of shape `[num_levels, 2]`.
            queries (torch.Tensor): Latent queries - tensor of shape `[batch_size, num_queries, num_channels]` which are projected for sampling offsets and attention weights.
            reference_points (torch.Tensor): XY positions of queries - tensor of shape `[batch_size, num_queries, 2]` or `[batch_size, num_queries, 4]`. In (x, y) or (cx, cy, w, h) order, where x and y are normalized to [0, 1] and (0, 0) is top-left corner and (1, 1) is bottom right corner.

        Returns:
            output (torch.Tensor): Samples aggregated based on attention weights of shape `[batch_size, num_queries, num_channels]`.
        """
        B, I, _ = img.shape  # noqa: E741
        B, N, _ = queries.shape
        L, H, P = self.num_levels, self.num_heads, self.num_points
        C = self.hidden_dim

        # project queries to get offsets and weights
        offsets, attention_weights = (
            self.query_input_proj(queries)
            .reshape(B, N, H, L, P, 3)
            .split_with_sizes((2, 1), dim=-1)
        )
        attention_weights = torch.nn.functional.softmax(
            attention_weights.reshape(B, N, H, L*P),
            dim=-1
        ).reshape(B, N, H, L, P)

        # project image
        img = (
            self.img_input_proj(img)
            .reshape(B, I, H, C//H)
        )

        # calculate sampling points
        last_dim = reference_points.shape[-1]
        if last_dim == 2:
            # [B, N, 1, 1, 1, 2] + [B, N, H, L, P, 2] * [L, 1, 2] -> [B, N, H, L, P, 2]
            sampling_points = (
                reference_points[:, :, None, None, None, :]
                + offsets / img_shapes[:, None, :]
            )
        elif last_dim == 4:
            # [B N, 1, 1, 1, 2] + [B, N, H, L, P, 2] * [B N, 1, 1, 1, 2] -> [B, N, H, L, P, 2]
            sampling_points = (
                reference_points[:, :, None, None, None, :2] 
                + offsets * reference_points[:, :, None, None, None, 2:] / (2*P)
            )
        else:
            raise ValueError(f"`reference_points` should have the last dim either 2 or 4, but got {last_dim}.")

        # deformable attention
        # [B, N, H, C//H] -> [B, N, C]
        out = multiscale_deformable_attention(
            img, img_shapes, sampling_points, attention_weights, self.padding_mode, self.align_corners)
        out = out.reshape(B, N, C)
        out = self.query_output_proj(out)
        return out
