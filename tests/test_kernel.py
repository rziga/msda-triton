import torch
from torch.nn import functional as F

from msda_triton.triton.msda_fw import triton_multi_scale_deformable_attention

def test_kernel():
    # Define input dimensions
    B, H, C, L, N, P = 4, 8, 32, 4, 1000, 3
    img_shapes = [(64, 64), (32, 32), (16, 16), (8, 8)]  # Ensure sum(h*w) = I
    I = sum(h * w for h, w in img_shapes)  # Total pixels across scales

    # Generate test inputs
    img = torch.randn(B, I, H, C, device="cuda")
    img_shapes = torch.tensor(img_shapes).to("cuda")
    sampling_points = torch.rand(B, N, H, L, P, 2, device="cuda")
    att_weights = torch.softmax(torch.randn(B, N, H, L, P, device="cuda"), dim=-1)
    
    true = torch_multi_scale_deformable_attention(img, img_shapes, sampling_points, att_weights)
    test = triton_multi_scale_deformable_attention(img, img_shapes, sampling_points, att_weights)

    torch.testing.assert_close(test, true)


############################
### Native torch version ###
############################

def torch_multi_scale_deformable_attention(
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
            value_l_, sampling_grid_l_, mode="bilinear", align_corners=True
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
