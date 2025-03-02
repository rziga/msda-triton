import torch
import triton
from triton import language as tl


# How this will go down is:
#   - we will go batch by batch, query by query and head by head
#   - (basically batch block size = 1, query block size = 1, head block size = 1)
#   - that is we will calculate blocks over [B, N, H, ||| L, P]
#   - that is B, N, H will be 1,1,1 parallelized and L, P will be blocked... ??? !!!
# TODO: If we make N dimension blocked with block size >= 16, we can use tensor cores !!!
#       I should probably profile this thing first, because I have a feeling that the whole thing IO bound due to bilinear sampling
# TODO: this is essentially padding mode == border because I clamp the sampling points


@triton.jit()
def load_shapes_and_level_offsets(
    shapes_ptr,
    L,
    BLOCK_SIZE_L: tl.constexpr,
):
    # load shapes
    shapes_ptrs = tl.make_block_ptr(
        shapes_ptr, (L, 2), (2, 1), (0, 0), (BLOCK_SIZE_L, 2), (0, 1))
    # [L, 2]
    shapes = tl.load(shapes_ptrs, padding_option="zero", boundary_check=(0, ))
    # [L], [L]
    h, w = tl.split(shapes)

    # calculate level offsets
    # [L]
    sizes = (h * w).to(tl.int32)
    # [L]
    level_offsets = tl.cumsum(sizes) - sizes

    return h, w, level_offsets

@triton.jit()
def make_sampling_points_block_ptr(
    sampling_points_ptr,
    B, N, H, L, P,
    bid, nid, hid,
    BLOCK_SIZE_L: tl.constexpr,
    BLOCK_SIZE_P: tl.constexpr,
):
    return tl.make_block_ptr(
        sampling_points_ptr,
        shape=(B, N, H, L, P, 2),
        strides=(N*H*L*P*2, H*L*P*2, L*P*2, P*2, 2, 1),
        offsets=(bid, nid, hid, 0, 0, 0),
        block_shape=(1, 1, 1, BLOCK_SIZE_L, BLOCK_SIZE_P, 2),
        order=(0, 1, 2, 3, 4, 5)
    )

@triton.jit()
def make_attention_weights_block_ptr(
    attention_weights_ptr,
    B, N, H, L, P,
    bid, nid, hid,
    BLOCK_SIZE_L: tl.constexpr,
    BLOCK_SIZE_P: tl.constexpr,
):
    return tl.make_block_ptr(
        attention_weights_ptr, 
        shape=(B, N, H, L, P),
        strides=(N*H*L*P, H*L*P, L*P, P, 1),
        offsets=(bid, nid, hid, 0, 0),
        block_shape=(1, 1, 1, BLOCK_SIZE_L, BLOCK_SIZE_P),
        order=(0, 1, 2, 3, 4),
    )

@triton.jit()
def make_output_block_ptr(
    out_ptr,
    B, N, H, C,
    bid, nid, hid,
    BLOCK_SIZE_C: tl.constexpr,
):
    return tl.make_block_ptr(
        out_ptr,
        shape=(B, N, H, C),
        strides=(N*H*C, H*C, C, 1),
        offsets=(bid, nid, hid, 0),
        block_shape=(1, 1, 1, BLOCK_SIZE_C),
        order=(0, 1, 2, 3),
    )

@triton.jit()
def sample_bilinear(
    x, y, w, h, level_offsets,
    
    img_ptr,

    B, I, C, N, H, L, P,
    
    bid, nid, hid,

    BLOCK_SIZE_L: tl.constexpr,
    BLOCK_SIZE_P: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,

    RETURN_FOR_BACKWARD: tl.constexpr
):
    # unnormalize, make sure that w and h are not 0
    # both [L, P]
    x *= tl.maximum(0, w[:, None] - 1)
    y *= tl.maximum(0, h[:, None] - 1)

    # find neighbors
    # all [L, P]
    x0 = tl.floor(x).to(tl.int32)
    y0 = tl.floor(y).to(tl.int32)
    x1 = tl.minimum(x0+1, w[:, None]-1)
    y1 = tl.minimum(y0+1, h[:, None]-1)

    # calculate mask
    # [L, P, C]
    mask = (
          (tl.arange(0, BLOCK_SIZE_L)[:, None, None] < L)
        & (tl.arange(0, BLOCK_SIZE_P)[None, :, None] < P)
        & (tl.arange(0, BLOCK_SIZE_C)[None, None, :] < C) 
    )

    # calculate start offset for image
    img_offset = img_ptr + bid*I*H*C

    # all [L, P, C]
    img00_offsets = (
        (level_offsets[:, None] + y0*w[:, None] + x0)[:, :, None] * H*C   # [L, P, 1]
        + hid*C                                                             #       [1]
        + tl.arange(0, BLOCK_SIZE_C)                                        #       [C]
    )
    img01_offsets = (
        (level_offsets[:, None] + y0*w[:, None] + x1)[:, :, None] * H*C   # [L, P, 1]
        + hid*C                                                             #       [1]
        + tl.arange(0, BLOCK_SIZE_C)                                        #       [C]
    )
    img10_offsets = (
        (level_offsets[:, None] + y1*w[:, None] + x0)[:, :, None] * H*C   # [L, P, 1]
        + hid*C                                                             #       [1]
        + tl.arange(0, BLOCK_SIZE_C)                                        #       [C]
    )
    img11_offsets = (
        (level_offsets[:, None] + y1*w[:, None] + x1)[:, :, None] * H*C   # [L, P, 1]
        + hid*C                                                             #       [1]
        + tl.arange(0, BLOCK_SIZE_C)                                        #       [C]
    )

    # all [L, P, C]
    img00 = tl.load(img_offset + img00_offsets, mask)
    img01 = tl.load(img_offset + img01_offsets, mask)
    img10 = tl.load(img_offset + img10_offsets, mask)
    img11 = tl.load(img_offset + img11_offsets, mask)

    # bilinear interpolation
    # [L, P, 1]
    dx = (x - x0)[:, :, None]
    # [L, P, 1]
    dy = (y - y0)[:, :, None]
    # [L, P, C] -> [L*P, C]
    samples = (
          img00 * (1 - dy) * (1 - dx)
        + img01 * (1 - dy) * (    dx)
        + img10 * (    dy) * (1 - dx)
        + img11 * (    dy) * (    dx)
    ).reshape(BLOCK_SIZE_L*BLOCK_SIZE_P, BLOCK_SIZE_C)

    return (
        samples,
        img00, img01, img10, img11,
        dx, dy,
        img00_offsets, img01_offsets, img10_offsets, img11_offsets,
        mask,
    ) if RETURN_FOR_BACKWARD else samples


@triton.jit()
def triton_multi_scale_deformable_attention_fwd_kernel(
    out_ptr,                          # [B, N, H, | C]
    img_ptr: tl.const,                # [B, I, H, C]
    sampling_points_ptr: tl.const,    # [B, N, H, | L, P, 2]
    attention_weights_ptr: tl.const,  # [B, N, H, | L, P]
    shapes_ptr: tl.const,             # [L, 2]
    B, I, C, N, H, L, P,
    BLOCK_SIZE_C: tl.constexpr,
    BLOCK_SIZE_L: tl.constexpr,
    BLOCK_SIZE_P: tl.constexpr,
):
    # block ids
    nid = tl.program_id(0)
    bid = tl.program_id(1)
    hid = tl.program_id(2)

    # load shapes and calculate level offsets
    # all [L]
    h, w, level_offsets = load_shapes_and_level_offsets(
        shapes_ptr, L, BLOCK_SIZE_L
    )

    # load sampling points
    # [1, 1, 1, L, P, 2]
    sampling_points_ptrs = make_sampling_points_block_ptr(
        sampling_points_ptr, 
        B, N, H, L, P,
        bid, nid, hid,
        BLOCK_SIZE_L, BLOCK_SIZE_P,
    )
    sampling_points = tl.load(sampling_points_ptrs, padding_option="zero", boundary_check=(3, 4))
    # [L, P, 2]
    sampling_points = sampling_points.reshape(BLOCK_SIZE_L, BLOCK_SIZE_P, 2)

    # calculate offsets for bilinear interpolation
    # [L, P], [L, P]
    x, y = tl.split(sampling_points)

    # bilinear sampling
    # [L*P, C]
    samples = sample_bilinear(
        x, y, w, h, level_offsets, 
        img_ptr, 
        B, I, C, N, H, L, P, 
        bid, nid, hid, 
        BLOCK_SIZE_L, BLOCK_SIZE_P, BLOCK_SIZE_C,
        RETURN_FOR_BACKWARD=False,
    )

    # load the attention weights
    attention_weights_ptrs = make_attention_weights_block_ptr(
        attention_weights_ptr,
        B, N, H, L, P,
        bid, nid, hid,
        BLOCK_SIZE_L, BLOCK_SIZE_P,
    )
    # [1, 1, 1, L, P]
    attention_weights = tl.load(attention_weights_ptrs, padding_option="zero", boundary_check=(3, 4))
    # [L*P]
    attention_weights = attention_weights.reshape(BLOCK_SIZE_L*BLOCK_SIZE_P)

    # do the attention weighting
    # [L*P, 1] * [L*P, C] -> [C]
    out = tl.sum(attention_weights[:, None] * samples, axis=0)

    # write the output
    out_ptrs = make_output_block_ptr(
        out_ptr,
        B, N, H, C,
        bid, nid, hid,
        BLOCK_SIZE_C,
    )
    tl.store(out_ptrs, out[None, None, None, :], boundary_check=(3,))


def triton_multi_scale_deformable_attention_fwd(
    img: torch.Tensor,
    img_shapes: torch.Tensor,
    sampling_points: torch.Tensor,
    attention_weights: torch.Tensor,
) -> torch.Tensor:

    B, I, H, C = img.shape
    B, N, H, L, P, _ = sampling_points.shape

    # run the kernel
    out = img.new_empty(B, N, H, C)
    triton_multi_scale_deformable_attention_fwd_kernel[N, B, H](
        out, 
        img.contiguous(), 
        sampling_points.contiguous(),
        attention_weights.contiguous(),
        img_shapes.contiguous(),
        B, I, C, N, H, L, P,
        triton.next_power_of_2(C),
        triton.next_power_of_2(L),
        triton.next_power_of_2(P),
    )

    return out.reshape(B, N, H*C)


@triton.jit()
def triton_multi_scale_deformable_attention_bwd_kernel(
    img_grad_ptr,               # [B, I, H, C]
    sampling_points_grad_ptr,   # [B, N, H, L, P, 2]
    attention_weights_grad_ptr, # [B, N, H, L, P]

    out_grad_ptr: tl.const,           # [B, N, H, C]
    img_ptr: tl.const,                # [B, I, H, C]
    sampling_points_ptr: tl.const,    # [B, N, H, L, P, 2]
    attention_weights_ptr: tl.const,  # [B, N, H, L, P]

    shapes_ptr: tl.const,             # [L, 2]
    B, I, C, N, H, L, P,
    BLOCK_SIZE_C: tl.constexpr,
    BLOCK_SIZE_L: tl.constexpr,
    BLOCK_SIZE_P: tl.constexpr,
):
    # block ids
    nid = tl.program_id(0)
    bid = tl.program_id(1)
    hid = tl.program_id(2)

    #####################
    # RECOMPUTE FORWARD #
    #####################

    # load shapes
    # [L, 2]
    h, w, level_offsets = load_shapes_and_level_offsets(
        shapes_ptr, L, BLOCK_SIZE_L
    )

    # load sampling points
    # [1, 1, 1, L, P, 2]
    sampling_points_ptrs = make_sampling_points_block_ptr(
        sampling_points_ptr, 
        B, N, H, L, P,
        bid, nid, hid,
        BLOCK_SIZE_L, BLOCK_SIZE_P,
    )
    sampling_points = tl.load(sampling_points_ptrs, padding_option="zero", boundary_check=(3, 4))
    # [L, P, 2]
    sampling_points = sampling_points.reshape(BLOCK_SIZE_L, BLOCK_SIZE_P, 2)

    # calculate offsets for bilinear interpolation
    # [L, P], [L, P]
    x, y = tl.split(sampling_points)

    # bilinear sampling
    (
        samples,                    # [L*P, C]
        img00, img01, img10, img11, # [L, P]
        dx, dy,                     # [L, P]
        img00_offsets, img01_offsets, img10_offsets, img11_offsets, # [L, P, C]
        mask, # [L, P, C]
    ) = sample_bilinear(
        x, y, w, h, level_offsets, 
        img_ptr, 
        B, I, C, N, H, L, P, 
        bid, nid, hid, 
        BLOCK_SIZE_L, BLOCK_SIZE_P, BLOCK_SIZE_C,
        RETURN_FOR_BACKWARD=True,
    )

    # load the attention weights
    attention_weights_ptrs = make_attention_weights_block_ptr(
        attention_weights_ptr,
        B, N, H, L, P,
        bid, nid, hid,
        BLOCK_SIZE_L, BLOCK_SIZE_P,
    )
    # [1, 1, 1, L, P]
    attention_weights = tl.load(attention_weights_ptrs, padding_option="zero", boundary_check=(3, 4))
    # [L*P]
    attention_weights = attention_weights.reshape(BLOCK_SIZE_L*BLOCK_SIZE_P)

    #################
    # BACKWARD PASS #
    #################

    # load the output gradient
    out_grad_ptrs = make_output_block_ptr(
        out_grad_ptr,
        B, N, H, C,
        bid, nid, hid,
        BLOCK_SIZE_C,
    )
    # [1, 1, 1, C]
    out_grad = tl.load(out_grad_ptrs, padding_option="zero", boundary_check=(3,))
    # [C]
    out_grad = out_grad.reshape(BLOCK_SIZE_C)

    # calculate and store att weights gradient
    # [1, C] * [L*P, C] sum -> [L*P]
    attention_weights_grad = tl.sum(out_grad[None, :] * samples, axis=1)
    # [1, 1, 1, L, P]
    attention_weights_grad = attention_weights_grad.reshape(1, 1, 1, BLOCK_SIZE_L, BLOCK_SIZE_P)
    attention_weights_grad_ptrs = make_attention_weights_block_ptr(
        attention_weights_grad_ptr,
        B, N, H, L, P,
        bid, nid, hid,
        BLOCK_SIZE_L, BLOCK_SIZE_P,
    )
    tl.store(attention_weights_grad_ptrs, attention_weights_grad, boundary_check=(3, 4))

    # calculate sampling points grad
    attention_weights = attention_weights.reshape(BLOCK_SIZE_L, BLOCK_SIZE_P)
    # [L, P, C] sum -> [L, P]
    x_grad = tl.sum(out_grad * attention_weights[:, :, None] * (w[:, None, None] - 1) * (
        (1-dy) * (img01-img00) + dy * (img11-img10)
    ), axis=2)
    # [L, P, C] sum -> [L, P]
    y_grad = tl.sum(out_grad * attention_weights[:, :, None] * (h[:, None, None] - 1) * (
        (1-dx) * (img10-img00) + dx * (img11-img01)
    ), axis=2)
    # [L, P, 2]
    sampling_points_grad = tl.join(x_grad, y_grad)

    # store sampling points grad
    # [1, 1, 1, L, P, 2]
    sampling_points_grad = sampling_points_grad.reshape(1, 1, 1, BLOCK_SIZE_L, BLOCK_SIZE_P, 2)
    sampling_points_grad_ptrs = make_sampling_points_block_ptr(
        sampling_points_grad_ptr, 
        B, N, H, L, P,
        bid, nid, hid,
        BLOCK_SIZE_L, BLOCK_SIZE_P,
    )
    tl.store(sampling_points_grad_ptrs, sampling_points_grad, boundary_check=(3, 4))

    # calculate and store img grad
    # now we load the pixels
    img_grad_offset = img_grad_ptr + bid*I*H*C
    # all [L, P, C]
    img00_grad = out_grad * attention_weights[:, :, None] * (1-dy) * (1-dx)
    img01_grad = out_grad * attention_weights[:, :, None] * (1-dy) * (  dx)
    img10_grad = out_grad * attention_weights[:, :, None] * (  dy) * (1-dx)
    img11_grad = out_grad * attention_weights[:, :, None] * (  dy) * (  dx)

    # store to output tensor
    # NOTE: atomic_add here since multiple queries can sample the same pixel
    tl.atomic_add(img_grad_offset + img00_offsets, img00_grad, mask)
    tl.atomic_add(img_grad_offset + img01_offsets, img01_grad, mask)
    tl.atomic_add(img_grad_offset + img10_offsets, img10_grad, mask)
    tl.atomic_add(img_grad_offset + img11_offsets, img11_grad, mask)


def triton_multi_scale_deformable_attention_bwd(
    out_grad: torch.Tensor,
    img: torch.Tensor,
    img_shapes: torch.Tensor,
    sampling_points: torch.Tensor,
    attention_weights: torch.Tensor,
) -> torch.Tensor:

    B, I, H, C = img.shape
    B, N, H, L, P, _ = sampling_points.shape

    # init grad buffers
    img_grad = torch.zeros_like(img)
    sampling_points_grad = torch.zeros_like(sampling_points)
    attention_weights_grad = torch.zeros_like(attention_weights)

    # run the kernel
    triton_multi_scale_deformable_attention_bwd_kernel[N, B, H](
        img_grad.contiguous(),
        sampling_points_grad.contiguous(),
        attention_weights_grad.contiguous(),
        out_grad.contiguous(),
        img.contiguous(), 
        sampling_points.contiguous(),
        attention_weights.contiguous(),
        img_shapes.contiguous(),
        B, I, C, N, H, L, P,
        triton.next_power_of_2(C),
        triton.next_power_of_2(L),
        triton.next_power_of_2(P),
    )

    return img_grad, sampling_points_grad, attention_weights_grad
