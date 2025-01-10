import torch

from msda_triton.triton.msda_fw import triton_deformable_att

def test_kernel():
    img = torch.rand(6, 128, 128, 256).cuda()
    sampling_points = torch.rand(6, 128, 4, 2).cuda()
    att_weights = torch.rand(6, 128, 4).cuda()
    
    true = torch_deformable_att(img, sampling_points, att_weights)
    test = triton_deformable_att(img, sampling_points, att_weights)

    torch.testing.assert_close(test, true)

def torch_deformable_att(img, sampling_points, att_weights):
    """
    img - [B, H, W, C]
    sampling_points - [B, N, P, 2] in xy relative [0-1]
    att_weights - [B, N, P]

    outputs: samples - [B, N, C]
    """
    B, H, W, C = img.shape
    B, N, P, _ = sampling_points.shape
    device = img.device
    
    # unnormalize coordinates
    x, y = sampling_points.unbind(-1)
    x = x * (W-1) # [B, N, P]
    y = y * (H-1) # [B, N, P]

    # get idxs for interpolation
    xl = x.floor().long()
    xr = xl + 1
    xr_ = xr.clamp(max=W-1)
    yt = y.floor().long()
    yb = yt + 1
    yb_ = yb.clamp(max=H-1)
    
    # get the interpolation point values
    b = torch.arange(B, device=device)[:, None, None]
    point_tl, point_tr = img[b, yt , xl, :], img[b, yt , xr_, :] # [B, N, P, C], [B, N, P, C]
    point_bl, point_br = img[b, yb_, xl, :], img[b, yb_, xr_, :] # [B, N, P, C], [B, N, P, C]
    #print("point:")
    #print(point_tl, point_tr)
    #print(point_bl, point_br)

    # get the interpolation weights
    weight_tl, weight_tr = (yb-y)*(xr-x), (yb-y)*(x-xl) # [B, N, P], [B, N, P]
    weight_bl, weight_br = (y-yt)*(xr-x), (y-yt)*(x-xl) # [B, N, P], [B, N, P]
    #print("weight:")
    #print(weight_tl, weight_tr)
    #print(weight_bl, weight_br)
    
    # interpolate the result
    samples = (
          weight_tl[..., None]*point_tl + weight_tr[..., None]*point_tr
        + weight_bl[..., None]*point_bl + weight_br[..., None]*point_br
    ) # [B, N, P, C]
    #print(samples)

    # use attention weights
    weighted = (samples * att_weights[..., None]).sum(-2) # [B, N, C]
    #print(weighted)

    return weighted
