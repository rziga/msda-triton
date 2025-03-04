import torch
from torch.autograd.function import Function, once_differentiable

from .triton_kernels import (
    triton_multi_scale_deformable_attention_fwd,
    triton_multi_scale_deformable_attention_bwd,
)


def triton_multiscale_deformable_attention(
    img: torch.Tensor,
    img_shapes: torch.Tensor,
    sampling_points: torch.Tensor,
    attention_weights: torch.Tensor,
) -> torch.Tensor:

    # Check input dtypes
    valid_dtypes = [torch.float32, torch.float64]
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
    )


class _TritonMultiscaleDeformableAttentionFunction(Function):

    @staticmethod
    def forward(
        ctx, 
        img: torch.Tensor,
        img_shapes: torch.Tensor,
        sampling_points: torch.Tensor,
        attention_weights: torch.Tensor,
    ) -> torch.Tensor:
        ctx.save_for_backward(img, img_shapes, sampling_points, attention_weights)
        out = triton_multi_scale_deformable_attention_fwd(
            img, img_shapes, sampling_points, attention_weights
        )
        return out
    
    @staticmethod
    @once_differentiable
    def backward(
        ctx, 
        out_grad: torch.Tensor,
    ) -> tuple[torch.Tensor, None, torch.Tensor, torch.Tensor]:
        img, img_shapes, sampling_points, attention_weights = ctx.saved_tensors
        img_grad, sampling_points_grad, attention_weights_grad = triton_multi_scale_deformable_attention_bwd(
            out_grad, img, img_shapes, sampling_points, attention_weights
        )
        return img_grad, None, sampling_points_grad, attention_weights_grad
