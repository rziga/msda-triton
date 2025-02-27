import torch
from torch.nn import functional as F
import pytest

from msda_triton import multiscale_deformable_attention


def test_forward():
    # Define input dimensions
    B, H, C, L, N, P = 4, 8, 32, 4, 1000, 3
    img_shapes = [(64, 64), (32, 32), (16, 16), (8, 8)]  # Ensure sum(h*w) = I
    I = sum(h * w for h, w in img_shapes)  # Total pixels across scales

    # Generate test inputs
    img = torch.randn(B, I, H, C, device="cuda")
    img_shapes = torch.tensor(img_shapes).to("cuda")
    sampling_points = torch.rand(B, N, H, L, P, 2, device="cuda")
    att_weights = torch.softmax(torch.randn(B, N, H, L, P, device="cuda"), dim=-1)

    true = torch_msda_manual(img, img_shapes, sampling_points, att_weights)
    test = multiscale_deformable_attention(img, img_shapes, sampling_points, att_weights)

    torch.testing.assert_close(test, true)


@pytest.mark.parametrize(argnames="dtype", argvalues=("float32", "float64"))
def test_backward(dtype):
    dtype = getattr(torch, dtype)

    # Define input dimensions
    B, H, C, L, N, P = 4, 8, 64, 4, 14213, 4
    #img_shapes = [(32, 32)]
    img_shapes = [(64, 64), (32, 32), (16, 16), (8, 8)]  # Ensure sum(h*w) = I
    I = sum(h * w for h, w in img_shapes)  # Total pixels across scales

    # Generate test inputs
    img = torch.randn(B, I, H, C, device="cuda", requires_grad=True, dtype=dtype)
    img_shapes = torch.tensor(img_shapes, device="cuda")
    sampling_points = torch.rand(B, N, H, L, P, 2, device="cuda", dtype=dtype)
    #sampling_points[..., 0, :] = 1
    #sampling_points[..., 0, :] = 1/31
    sampling_points.requires_grad_(True)
    att_weights = torch.rand(B, N, H, L, P, device="cuda", requires_grad=True, dtype=dtype)
    out_grad = torch.rand(B, N, H*C, device="cuda", dtype=dtype)

    # run torch
    a = torch_msda_manual(img, img_shapes, sampling_points, att_weights)
    a.backward(out_grad)
    a_img_grad, a_sampling_points_grad, a_att_weights_grad = img.grad, sampling_points.grad, att_weights.grad
    img.grad, sampling_points.grad, att_weights.grad = None, None, None

    b = multiscale_deformable_attention(img, img_shapes, sampling_points, att_weights)
    b.backward(out_grad)
    b_img_grad, b_sampling_points_grad, b_att_weights_grad = img.grad, sampling_points.grad, att_weights.grad
    img.grad, sampling_points.grad, att_weights.grad = None, None, None

    if dtype == torch.float32:
        atol, rtol = 1e-4, 1e-3
    else:
        atol, rtol = 1e-8, 1e-6

    torch.testing.assert_close(a, b)
    torch.testing.assert_close(a_img_grad, b_img_grad, atol=atol, rtol=rtol)
    torch.testing.assert_close(a_sampling_points_grad, b_sampling_points_grad, atol=atol, rtol=rtol)
    torch.testing.assert_close(a_att_weights_grad, b_att_weights_grad, atol=atol, rtol=rtol)


def test_memory():

    # Define input dimensions
    B, H, C, L, N, P = 4, 8, 32, 4, 20000, 4
    img_shapes = [(128, 128), (64, 64), (32, 32), (16, 16)]  # Ensure sum(h*w) = I
    I = sum(h * w for h, w in img_shapes)  # Total pixels across scales

    # Generate test inputs
    img = torch.randn(B, I, H, C, device="cuda", requires_grad=True)
    img_shapes = torch.tensor(img_shapes, device="cuda")
    sampling_points = torch.rand(B, N, H, L, P, 2, device="cuda", requires_grad=True)
    att_weights = torch.rand(B, N, H, L, P, device="cuda", requires_grad=True)
    
    def measure_memory_usage(func, *args, **kwargs):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()  # Reset peak memory tracking
        start_mem = torch.cuda.memory_allocated()
        
        out = func(*args, **kwargs)  # Run the function
        out.sum().backward()
        
        torch.cuda.synchronize()
        max_mem = torch.cuda.max_memory_allocated()  # Get peak memory usage
        return (max_mem - start_mem) / 1e6  # Convert bytes to MB

    # Benchmarking memory usage for PyTorch implementation
    repeats = 100
    torch_mem_usage = 0
    for _ in range(repeats):
        torch_mem_usage += measure_memory_usage(
            lambda: torch_msda_bilinear(img, img_shapes, sampling_points, att_weights)
        )
    torch_mem_usage /= repeats

    # Benchmarking memory usage for Triton implementation
    triton_mem_usage = 0
    for _ in range(repeats):
        triton_mem_usage += measure_memory_usage(
            lambda: multiscale_deformable_attention(img, img_shapes, sampling_points, att_weights)
        )
    triton_mem_usage /= 100

    print(f"PyTorch Memory Usage: {torch_mem_usage:.2f} MB")
    print(f"Triton Memory Usage: {triton_mem_usage:.2f} MB")
    print(f"Triton Memory Efficiency: {torch_mem_usage / triton_mem_usage:.2f}x")


#############################
### Native torch versions ###
#############################

def _torch_multiscale_deformable_attention(
    value: torch.Tensor,
    value_spatial_shapes: torch.Tensor,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
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
            value_l_, sampling_grid_l_, mode="bilinear", align_corners=True, padding_mode="border"
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
    return output.transpose(1, 2).contiguous()

def _torch_multiscale_deformable_attention2(
    img: torch.Tensor,                  # [B, I, H, C]
    img_shapes: torch.Tensor,           # [L, 2]
    sampling_points: torch.Tensor,      # [B, N, H, L, P, 2]
    attention_weights: torch.Tensor,    # [B, N, H, L, P]
) -> torch.Tensor:
    B, I, H, C = img.shape
    B, N, H, L, P, _ = sampling_points.shape

    # calculate level offsets
    # [L]
    level_offsets = img_shapes.prod(-1)
    level_offsets = level_offsets.cumsum(0) - level_offsets
    
    # unnormalize sampling points and find neighbors
    # [L, 1, 2]
    wh_max = img_shapes.flip(-1)[:, None, :] - 1
    # [B, N, H, L, P, 2] * [L, 1, 2]
    sampling_points = sampling_points * wh_max
    # [B, N, H, L, P, 2]
    tl = sampling_points.floor().to(torch.long)
    # [B, N, H, L, P, 2] clamp with [L, 1, 2]
    br = torch.clamp(tl + 1, max=wh_max)
    # split xs and ys, all [B, N, H, L, P]
    x, y = sampling_points.unbind(-1)
    (x0, y0), (x1, y1) = tl.unbind(-1), br.unbind(-1)

    # calculate weights
    dx = x - x0
    dy = y - y0

    # sample
    def sample_img(y, x):
        # [B, N, H, L, P]
        idxs = level_offsets[:, None] + y * img_shapes[:, [0]] + x
        # [B, N, L, P, H]
        idxs = idxs.permute(0, 1, 3, 4, 2)
        # [B, N*L*P, H]
        idxs = idxs.reshape(B, N*L*P, H)
        # [B, N*L*P, H, C]
        idxs = idxs[..., None].expand(B, N*L*P, H, C)
        # [B, N*L*P, H, C]
        samples = torch.gather(img, 1, idxs)
        # [B, N, L, P, H, C]
        samples = samples.reshape(B, N, L, P, H, C)
        # [B, N, H, L, P, C]
        return samples.permute(0, 1, 4, 2, 3, 5)

    # all [B, N, H, L, P, C]
    img00 = sample_img(y0, x0)
    img01 = sample_img(y0, x1)
    img10 = sample_img(y1, x0)
    img11 = sample_img(y1, x1)
    
    # [B, N, H, L, P, C]
    interpolated = (
        img00 * (1-dy[..., None]) * (1-dx[..., None]) +
        img01 * (1-dy[..., None]) * (  dx[..., None]) +
        img10 * (  dy[..., None]) * (1-dx[..., None]) +
        img11 * (  dy[..., None]) * (  dx[..., None])
    )

    # weigh the interpolated samples
    # [B, N, H, L, P, C]
    out = interpolated * attention_weights[..., None]
    # [B, N, H, C]
    out = torch.sum(out, dim=(3, 4))
    # [B, N, H*C]
    out = out.reshape(B, N, H*C)
    return out

# compile them for optimal performance
torch_msda_bilinear = torch.compile(_torch_multiscale_deformable_attention)
torch_msda_manual = torch.compile(_torch_multiscale_deformable_attention2)
