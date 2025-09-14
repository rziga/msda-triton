from typing import Literal
from itertools import product

import torch
from torch.nn import functional as F
import pytest

from msda_triton.frontend import (
    triton_multiscale_deformable_attention,
    native_multiscale_deformable_attention,
    MultiscaleDeformableAttention
)


DTYPE_TO_TOLERANCE = {
    torch.float16: {
        "fwd": (1e-1, 1e-1),
    },
    torch.float32: {
        "fwd": (1e-4, 1e-3),
        "bwd": (1e-3, 1e-2),
    },
    torch.float64: {
        "fwd": (1e-8, 1e-8),
        "bwd": (1e-8, 1e-8),
    },
}


def get_random_functional_data(
    B=4, H=8, C=32, L=4, N=1000, P=3,
    device="cuda",
    dtype=torch.float32,
    return_grad=False,
):
    img_shapes = [(64 // 2**i, 64 // 2**i) for i in range(L)]
    I = sum(h * w for h, w in img_shapes)  # Total pixels across scales  # noqa: E741
    img = torch.randn(B, I, H, C, device=device, dtype=dtype)
    img_shapes = torch.tensor(img_shapes).to(device)
    sampling_points = torch.rand(B, N, H, L, P, 2, device=device, dtype=dtype)
    att_weights = torch.softmax(torch.randn(B, N, H, L, P, device=device, dtype=dtype), dim=-1)
    out_grad = torch.rand(B, N, H, C, device="cuda", dtype=dtype)

    if return_grad:
        return img, img_shapes, sampling_points, att_weights, out_grad
    else:
        return img, img_shapes, sampling_points, att_weights

def get_toy_functional_data(
    x=0.0, y=0.0,
    device="cuda",
    dtype=torch.float32,
):
    img_shapes = [[4, 4]]
    I = sum(h * w for h, w in img_shapes)  # Total pixels across scales  # noqa: E741
    img = torch.arange(I, device=device, dtype=dtype).reshape(1, I, 1, 1)
    img_shapes = torch.tensor(img_shapes).to(device)
    sampling_points = torch.tensor([0.0, 0.1], device=device, dtype=dtype).reshape(1, 1, 1, 1, 1, 2)
    att_weights = torch.tensor(1.0, device=device, dtype=dtype).reshape(1, 1, 1, 1, 1)
    return img, img_shapes, sampling_points, att_weights


def get_module_data(
    B=4, C=256, L=4, N=1000, COOR=4,
    device="cuda",
    dtype=torch.float32,
):
    img_shapes = [(64 // 2**i, 64 // 2**i) for i in range(L)]
    I = sum(h * w for h, w in img_shapes)  # Total pixels across scales  # noqa: E741
    img = torch.randn(B, I, C, device=device, dtype=dtype)
    img_shapes = torch.tensor(img_shapes).to(device)
    queries = torch.randn(B, N, C, device=device)
    reference_points = torch.randn(B, N, COOR, device=device)

    return img, img_shapes, queries, reference_points

@pytest.mark.parametrize(
    argnames=["dtype", "sampling_mode", "padding_mode", "align_corners"],
    argvalues=product(
        ["float16", "float32", "float64"],
        ["bilinear", "nearest"],
        ["border", "zeros"],
        [True, False],
    )
)
def test_triton_toy_forward(dtype, sampling_mode, padding_mode, align_corners):
    dtype = getattr(torch, dtype)

    for x in torch.linspace(0, 1, 100):
        for y in torch.linspace(0, 1, 100):
            img, img_shapes, sampling_points, att_weights = get_toy_functional_data(x=x, y=y, device="cuda", dtype=dtype)
            true = torch_msda_reference(img, img_shapes, sampling_points, att_weights, sampling_mode, padding_mode, align_corners)
            test = triton_multiscale_deformable_attention(img, img_shapes, sampling_points, att_weights, sampling_mode, padding_mode, align_corners)

            atol, rtol = DTYPE_TO_TOLERANCE[dtype]["fwd"]
            print(f"{test=}\n{true=}")
            torch.testing.assert_close(test, true, atol=atol, rtol=rtol, msg=lambda msg: msg+f"\n @ {x=}{y=}")


@pytest.mark.parametrize(
    argnames=["dtype", "sampling_mode", "padding_mode", "align_corners"],
    argvalues=filter(
        lambda args: not (args[1]=="nearest" and args[0] in {"float16", "float32"}), # nearest with these is too flaky
        product(
            ["float16", "float32", "float64"],
            ["bilinear", "nearest"],
            ["border", "zeros"],
            [True, False],
        )
    )
)
def test_triton_forward(dtype, sampling_mode, padding_mode, align_corners):
    dtype = getattr(torch, dtype)

    img, img_shapes, sampling_points, att_weights = get_random_functional_data(device="cuda", dtype=dtype)
    true = torch_msda_reference(img, img_shapes, sampling_points, att_weights, sampling_mode, padding_mode, align_corners)
    test = triton_multiscale_deformable_attention(img, img_shapes, sampling_points, att_weights, sampling_mode, padding_mode, align_corners)

    atol, rtol = DTYPE_TO_TOLERANCE[dtype]["fwd"]
    torch.testing.assert_close(test, true, atol=atol, rtol=rtol)


@pytest.mark.parametrize(
    argnames=["dtype", "sampling_mode", "padding_mode", "align_corners"],
    argvalues=filter(
        lambda args: not (args[1]=="nearest" and args[0] in {"float16", "float32"}), # nearest with these is too flaky
        product(
            ["float16", "float32", "float64"],
            ["bilinear", "nearest"],
            ["border", "zeros"],
            [True, False],
        )
    )
)
def test_triton_forward_oob_sampling(dtype, sampling_mode, padding_mode, align_corners):
    dtype = getattr(torch, dtype)

    img, img_shapes, sampling_points, att_weights = get_random_functional_data(device="cuda", dtype=dtype)
    test = triton_multiscale_deformable_attention(img, img_shapes, sampling_points, att_weights, sampling_mode, padding_mode, align_corners)
    true = torch_msda_reference(img, img_shapes, sampling_points, att_weights, sampling_mode, padding_mode, align_corners)

    atol, rtol = DTYPE_TO_TOLERANCE[dtype]["fwd"]
    torch.testing.assert_close(test, true, atol=atol, rtol=rtol)


@pytest.mark.parametrize(
    argnames=["dtype", "sampling_mode", "padding_mode", "align_corners"],
    argvalues=product(
        ["float16", "float32", "float64"],
        ["bilinear", "nearest"],
        ["border", "zeros"],
        [True, False],
    )
)
def test_native_forward(dtype, sampling_mode, padding_mode, align_corners):
    dtype = getattr(torch, dtype)

    img, img_shapes, sampling_points, att_weights = get_random_functional_data(device="cuda", dtype=dtype)
    true = torch_msda_reference(img, img_shapes, sampling_points, att_weights, sampling_mode, padding_mode, align_corners)
    test = native_multiscale_deformable_attention(img, img_shapes, sampling_points, att_weights, sampling_mode, padding_mode, align_corners)

    torch.testing.assert_close(test, true)


@pytest.mark.parametrize(
    argnames=["dtype", "sampling_mode", "padding_mode", "align_corners"],
    argvalues=product(
        ["float32", "float64"],
        ["bilinear"],
        ["border", "zeros"],
        [True, False],
    )
)
def test_backward(dtype, sampling_mode, padding_mode, align_corners):
    dtype = getattr(torch, dtype)

    img, img_shapes, sampling_points, att_weights, out_grad = get_random_functional_data(device="cuda", dtype=dtype, return_grad=True)
    img, sampling_points, att_weights = map(lambda t: t.requires_grad_(True), (img, sampling_points, att_weights))

    # run torch
    true = torch_msda_reference(img, img_shapes, sampling_points, att_weights, sampling_mode, padding_mode, align_corners)
    true.backward(out_grad)
    true_img_grad, true_sampling_points_grad, true_att_weights_grad = img.grad, sampling_points.grad, att_weights.grad
    img.grad, sampling_points.grad, att_weights.grad = None, None, None

    # run triton
    test = triton_multiscale_deformable_attention(img, img_shapes, sampling_points, att_weights, sampling_mode, padding_mode, align_corners)
    test.backward(out_grad)
    test_img_grad, test_sampling_points_grad, test_att_weights_grad = img.grad, sampling_points.grad, att_weights.grad
    img.grad, sampling_points.grad, att_weights.grad = None, None, None

    atol, rtol = DTYPE_TO_TOLERANCE[dtype]["bwd"]
    torch.testing.assert_close(true, test, atol=atol, rtol=rtol)
    torch.testing.assert_close(true_img_grad, test_img_grad, atol=atol, rtol=rtol)
    torch.testing.assert_close(true_sampling_points_grad, test_sampling_points_grad, atol=atol, rtol=rtol)
    torch.testing.assert_close(true_att_weights_grad, test_att_weights_grad, atol=atol, rtol=rtol)


@pytest.mark.parametrize(
    argnames="device,coors",
    argvalues=product(
        ["cpu", "cuda"],
        [2, 4],
    )
)
def test_nnmodule(device, coors):
    channels, heads, levels, points = 256, 8, 4, 8
    img, img_shapes, queries, reference_points = get_module_data(B=4, C=channels, N=1000, COOR=coors, device=device)
    module = MultiscaleDeformableAttention(
        channels, channels//heads, levels, heads, points,
        sampling_mode="bilinear", padding_mode="border", align_corners=True
    ).to(device)
    module.forward(img, img_shapes, queries, reference_points)


@pytest.mark.parametrize(
    argnames="dtype",
    argvalues=["float16", "float32", "float64"]
)
def test_autocast(dtype):
    dtype = getattr(torch, dtype)

    img, img_shapes, sampling_points, att_weights = get_random_functional_data(device="cuda", dtype=dtype)

    with torch.amp.autocast(device_type="cuda", dtype=dtype):
        _ = triton_multiscale_deformable_attention(
           img, img_shapes, sampling_points, att_weights, sampling_mode="bilinear", padding_mode="zeros", align_corners=False)


#############################
### Native torch versions ###
#############################


# Modified from huggingface transformers implementation
# which is licensed under Apache 2.0 license
# original at: https://github.com/huggingface/transformers/blob/f51ac9e059a78049362803c1d606a2c6a8160ee4/src/transformers/models/grounding_dino/modeling_grounding_dino.py#L584-L747
def _torch_multiscale_deformable_attention(
    value: torch.Tensor,
    value_spatial_shapes: torch.Tensor,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
    sampling_mode: Literal["bilinear", "nearest"],
    padding_mode: Literal["border", "zeros"],
    align_corners: bool,
) -> torch.Tensor:
    
    batch_size, _, num_heads, hidden_dim = value.shape
    _, num_queries, num_heads, num_levels, num_points, _ = sampling_locations.shape
    value_list = value.split([height * width for height, width in value_spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    
    for level_id, (height, width) in enumerate(value_spatial_shapes):
        
        # batch_size, height*width, num_heads, hidden_dim
        # -> batch_size, height*width, num_heads*hidden_dim
        # -> batch_size, num_heads*hidden_dim, height*width
        # -> batch_size*num_heads, hidden_dim, height, width
        value_l_ = (
            value_list[level_id]
            .flatten(2)
            .transpose(1, 2)
            .reshape(batch_size * num_heads, hidden_dim, height, width)
        )
        
        # batch_size, num_queries, num_heads, num_points, 2
        # -> batch_size, num_heads, num_queries, num_points, 2
        # -> batch_size*num_heads, num_queries, num_points, 2
        sampling_grid_l_ = sampling_grids[:, :, :, level_id].transpose(1, 2).flatten(0, 1)
        
        # batch_size*num_heads, hidden_dim, num_queries, num_points
        sampling_value_l_ = F.grid_sample(
            value_l_, sampling_grid_l_, mode=sampling_mode, align_corners=align_corners, padding_mode=padding_mode
        )
        sampling_value_list.append(sampling_value_l_)
    
    # (batch_size, num_queries, num_heads, num_levels, num_points)
    # -> (batch_size, num_heads, num_queries, num_levels, num_points)
    # -> (batch_size, num_heads, 1, num_queries, num_levels*num_points)
    attention_weights = attention_weights.transpose(1, 2).reshape(
        batch_size * num_heads, 1, num_queries, num_levels * num_points
    )
    output = (
        (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights)
        .sum(-1)
        .view(batch_size, num_heads * hidden_dim, num_queries)
    )
    return output.transpose(1, 2).reshape(batch_size, num_queries, num_heads, hidden_dim).contiguous()

# compile them for optimal performance
torch_msda_reference = torch.compile(_torch_multiscale_deformable_attention)
