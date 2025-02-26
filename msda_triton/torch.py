import torch
from torch.autograd import Function
from torch.autograd.function import once_differentiable

from .triton import (
    triton_multi_scale_deformable_attention_fwd,
    triton_multi_scale_deformable_attention_bwd,
)


class _multiscale_deformable_attention(Function):

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


multiscale_deformable_attention = _multiscale_deformable_attention.apply
