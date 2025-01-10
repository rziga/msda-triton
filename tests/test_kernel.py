import torch

from msda_triton.triton.msda_fw import bilinear_sample_triton

def test_kernel():
    img = torch.rand(1, 128, 128, 256).cuda()
    sampling_points = torch.rand(1, 128, 2).cuda()
    
    true = bilinear_sample_torch(img, sampling_points)
    test = bilinear_sample_triton(img, sampling_points)

    torch.testing.assert_close(test, true)

def bilinear_sample_torch(img, sampling_points):
    """
    img - [B, H, W, C]
    sampling_points - [B, N, 2] in xy relative [0-1]

    outputs: samples - [B, N]
    """
    B, H, W, _ = img.shape
    device = img.device
    
    # unnormalize coordinates
    x, y = sampling_points.unbind(-1)
    x = x * (W-1)
    y = y * (H-1)

    # get idxs for interpolation
    xl = x.floor().long()
    xr = xl + 1
    xr_ = xr.clamp(max=W-1)
    yt = y.floor().long()
    yb = yt + 1
    yb_ = yb.clamp(max=H-1)
    
    # get the interpolation point values
    b = torch.arange(B, device=device)[:, None]
    point_tl, point_tr = img[b, yt , xl, :], img[b, yt , xr_, :]
    point_bl, point_br = img[b, yb_, xl, :], img[b, yb_, xr_, :]
    #print("point:")
    #print(point_tl, point_tr)
    #print(point_bl, point_br)

    # get the interpolation weights
    weight_tl, weight_tr = (yb-y)*(xr-x), (yb-y)*(x-xl)
    weight_bl, weight_br = (y-yt)*(xr-x), (y-yt)*(x-xl)
    #print("weight:")
    #print(weight_tl, weight_tr)
    #print(weight_bl, weight_br)
    
    # interpolate the result
    samples = (
          weight_tl[..., None]*point_tl + weight_tr[..., None]*point_tr
        + weight_bl[..., None]*point_bl + weight_br[..., None]*point_br
    )

    return samples