import torch
import triton
from triton import language as tl


@triton.jit()
def triton_multi_scale_deformable_attention_kernel(
    out_ptr,                # [B, N, H, C]
    img_ptr,                # [B, I, H, C]
    sampling_points_ptr,    # [B, N, H, L, P, 2]
    attention_weights_ptr,  # [B, N, H, L, P]
    shapes_ptr,             # [L, 2]
    level_start_idx_ptr,    # [L]
    B, I, C, N, H, L, P,
    BLOCK_SIZE_C: tl.constexpr,
    BLOCK_SIZE_L: tl.constexpr,
    BLOCK_SIZE_P: tl.constexpr,
):
    # block ids
    bid = tl.program_id(0)
    nid = tl.program_id(1)
    hid = tl.program_id(2)

    # load shapes
    # [L, 2]
    shapes_ptrs = tl.make_block_ptr(
        shapes_ptr, (L, 2), (2, 1), (0, 0), (BLOCK_SIZE_L, 2), (0, 1))
    shapes = tl.load(shapes_ptrs, padding_option="zero", boundary_check=(0, ))
    # [L], [L]
    h, w = tl.split(shapes)

    # load level_start_idxs
    # [L]
    level_start_idx_ptrs = tl.make_block_ptr(
        level_start_idx_ptr, (L,), (1,), (0,), (BLOCK_SIZE_L,), (0,))
    level_start_idx = tl.load(level_start_idx_ptrs, padding_option="zero", boundary_check=(0, ))

    # load sampling points
    # [1, 1, 1, L, P, 2]
    b_str, n_str, h_str, l_str, p_str, d_str = N*H*L*P*2, H*L*P*2, L*P*2, P*2, 2, 1
    sampling_points_ptrs = tl.make_block_ptr(
        sampling_points_ptr,
        shape=(B, N, H, L, P, 2),
        strides=(b_str, n_str, h_str, l_str, p_str, d_str),
        offsets=(bid, nid, hid, 0, 0, 0),
        block_shape=(1, 1, 1, BLOCK_SIZE_L, BLOCK_SIZE_P, 2),
        order=(0, 1, 2, 3, 4, 5)
    )
    sampling_points = tl.load(sampling_points_ptrs, padding_option="zero", boundary_check=(3, 4))
    # [L, P, 2]
    sampling_points = sampling_points.reshape(BLOCK_SIZE_L, BLOCK_SIZE_P, 2)

    # calculate offsets for bilinear interpolation
    # [L, P], [L, P]
    x, y = tl.split(sampling_points)

    # unnormalize, make sure that w and h are not 0
    x *= tl.maximum(0, w[:, None] - 1)
    y *= tl.maximum(0, h[:, None] - 1)

    # find neighbors
    x0 = tl.floor(x).to(tl.int32)
    y0 = tl.floor(y).to(tl.int32)
    x1 = tl.minimum(x0+1, w[:, None]-1)
    y1 = tl.minimum(y0+1, h[:, None]-1)

    # now we load the pixels
    mask = (
          (tl.arange(0, BLOCK_SIZE_L)[:, None, None] < L)
        & (tl.arange(0, BLOCK_SIZE_C)[None, None, :] < C) 
    )
    img_offsets = img_ptr + bid*I*H*C
    # [L, P, C]
    img00 = tl.load(img_offsets + (
        (level_start_idx[:, None] + y0*w[:, None] + x0)[:, :, None] * H*C   # [L, P, 1]
        + hid*C                                                             #       [1]
        + tl.arange(0, BLOCK_SIZE_C)                                        #       [C]
    ), mask)
    # [L, P, C]
    img01 = tl.load(img_offsets + (
        (level_start_idx[:, None] + y0*w[:, None] + x1)[:, :, None] * H*C   # [L, P, 1]
        + hid*C                                                             #       [1]
        + tl.arange(0, BLOCK_SIZE_C)                                        #       [C]
    ), mask)
    # [L, P, C]
    img10 = tl.load(img_offsets + (
        (level_start_idx[:, None] + y1*w[:, None] + x0)[:, :, None] * H*C   # [L, P, 1]
        + hid*C                                                             #       [1]
        + tl.arange(0, BLOCK_SIZE_C)                                        #       [C]
    ), mask)
    # [L, P, C]
    img11 = tl.load(img_offsets + (
        (level_start_idx[:, None] + y1*w[:, None] + x1)[:, :, None] * H*C   # [L, P, 1]
        + hid*C                                                             #       [1]
        + tl.arange(0, BLOCK_SIZE_C)                                        #       [C]
    ), mask)

    # bilinear interpolation
    # [L, P, 1]
    dx = (x - x0)[:, :, None]
    # [L, P, 1]
    dy = (y - y0)[:, :, None]
    # [L, P, C] -> [L*P, C]
    samples = (
          img00 * (1 - dy) * (1 - dx)
        + img10 * (    dy) * (1 - dx)
        + img01 * (1 - dy) * (    dx)
        + img11 * (    dy) * (    dx)
    ).reshape(BLOCK_SIZE_L*BLOCK_SIZE_P, BLOCK_SIZE_C)

    # load the attention weights
    str_b, str_n, str_h, str_l, str_p = N*H*L*P, H*L*P, L*P, P, 1
    attention_weights_ptrs = tl.make_block_ptr(
        attention_weights_ptr, 
        shape=(B, N, H, L, P),
        strides=(str_b, str_n, str_h, str_l, str_p),
        offsets=(bid, nid, hid, 0, 0),
        block_shape=(1, 1, 1, BLOCK_SIZE_L, BLOCK_SIZE_P),
        order=(0, 1, 2, 3, 4),
    )
    # [1, 1, 1, L, P]
    attention_weights = tl.load(attention_weights_ptrs, padding_option="zero", boundary_check=(3, 4))
    # [L*P]
    attention_weights = attention_weights.reshape(BLOCK_SIZE_L*BLOCK_SIZE_P)

    # do the attention weighting
    out = tl.sum(attention_weights[:, None] * samples, axis=0)

    # write the output
    str_b, str_n, str_h, str_c = N*H*C, H*C, C, 1
    out_ptrs = tl.make_block_ptr(
        out_ptr,
        shape=(B, N, H, C),
        strides=(str_b, str_n, str_h, str_c),
        offsets=(bid, nid, hid, 0),
        block_shape=(1, 1, 1, BLOCK_SIZE_C),
        order=(0, 1, 2, 3),
    )
    tl.store(out_ptrs, out[None, None, None, :], boundary_check=(3,))


def triton_multi_scale_deformable_attention(
    img: torch.Tensor,
    img_shapes: torch.Tensor,
    sampling_points: torch.Tensor,
    attention_weights: torch.Tensor,
) -> torch.Tensor:

    B, I, H, C = img.shape
    B, N, H, L, P, _ = sampling_points.shape

    # get level start indices
    level_start_idx = torch.cat([
        img_shapes.new_zeros(1), img_shapes.prod(-1).cumsum(0)[:-1]
    ])

    # run the kernel
    out = img.new_empty(B, N, H, C)
    triton_multi_scale_deformable_attention_kernel[B, N, H](
        out, 
        img.contiguous(), 
        sampling_points.contiguous(),
        attention_weights.contiguous(),
        img_shapes.contiguous(),
        level_start_idx.contiguous(),
        B, I, C, N, H, L, P,
        triton.next_power_of_2(C),
        triton.next_power_of_2(L),
        triton.next_power_of_2(P),
    )

    return out.reshape(B, N, H*C)