import triton
from triton import language as tl

@triton.jit
def _get_nearest(coor):
    prev = tl.cast(tl.floor(coor), tl.int64)
    next = prev + 1
    return prev, next

@triton.jit
def _get_point(img_ptr, y, x, W, C, BLOCK_SIZE_C):
    c = tl.arange(0, BLOCK_SIZE_C)
    offsets = (
          W*C * y[:, :, None]
        +   C * x[:, :, None]
        +   1 * c[None, None, :]
    )
    mask = c[None, None, :] < C
    return tl.load(
        img_ptr+offsets,
        mask
    )

@triton.jit
def _sample_bilinear(img_ptr, sampling_points, H, W, C, C_BLOCK):
    
    # unnormalize coordinates
    x, y = tl.split(sampling_points)
    x = x * (W - 1) # [N, P]
    y = y * (H - 1) # [N, P]

    # calculate the xy coordinates of nearest pixels
    xl, xr = _get_nearest(x) # [N, P], [N, P]
    yt, yb = _get_nearest(y) # [N, P], [N, P]
    
    # sample the neighbors
    img_tl = _get_point(img_ptr, yt, xl, W, C, C_BLOCK) # [N, P, C]
    img_tr = _get_point(img_ptr, yt, xr, W, C, C_BLOCK) # [N, P, C]
    img_bl = _get_point(img_ptr, yb, xl, W, C, C_BLOCK) # [N, P, C]
    img_br = _get_point(img_ptr, yb, xr, W, C, C_BLOCK) # [N, P, C]

    # calculate sampling weights
    wei_tl, wei_tr = (yb-y)*(xr-x), (yb-y)*(x-xl) # [N, P], [N, P]
    wei_bl, wei_br = (y-yt)*(xr-x), (y-yt)*(x-xl) # [N, P], [N, P]

    # sum the contribution of neighboring points
    samples = (
          wei_tl[:, :, None]*img_tl + wei_tr[:, :, None]*img_tr
        + wei_bl[:, :, None]*img_bl + wei_br[:, :, None]*img_br
    ) # [N, P, C]

    return samples

@triton.jit
def deformable_att_kernel(
    out_ptr, # output pointer           - [B, N, C]
    img_ptr, # image pointer            - [B, H, W, C]
    smp_ptr, # sampling points pointer  - [B, N, P, C]
    wei_ptr, # att weights pointer      - [B, N, P]
    H, # image height
    W, # image width
    C, # image channels
    N, # number of point groups
    P, # number of points
    N_BLOCK: tl.constexpr, # assume BLOCK_SIZE_N > N
    C_BLOCK: tl.constexpr, # assume BLOCK_SIZE_C > C
    P_BLOCK: tl.constexpr, # assume BLOCK_SIZE_P > P
):
    # get program ids
    pid_batch = tl.program_id(0) # id of batch
    
    # compute offsets
    smp_size = N * P * 2
    smp_ptr = smp_ptr + pid_batch * smp_size

    wei_size = N * P
    wei_ptr = wei_ptr + pid_batch * wei_size
    
    img_size = H * W * C
    img_ptr = img_ptr + pid_batch * img_size
    
    out_size = N * C
    out_ptr = out_ptr + pid_batch * out_size

    # compute ranges and masks
    N_block_offsets = tl.arange(0, N_BLOCK)
    N_block_mask = N_block_offsets < N

    C_block_offsets = tl.arange(0, C_BLOCK)
    C_block_mask = C_block_offsets < C

    P_block_offsets = tl.arange(0, P_BLOCK)
    P_block_mask = P_block_offsets < P

    # load the sampling points
    smp_offsets = (
        2*P * N_block_offsets[:, None, None]
        + 2 * P_block_offsets[None, :, None]
        + 1 * tl.arange(0, 2)[None, None, :]
    ) # [N, P, 2]
    smp_mask = (
          N_block_mask[:, None, None]
        & P_block_mask[None, :, None]        
    ) # [N, P, 1]
    sampling_points = tl.load(
        smp_ptr+smp_offsets,
        smp_mask
    ) # [N, P, 2]

    # load the att weights
    wei_offsets = (
          P * N_block_offsets[:, None]
        + 1 * P_block_offsets[None, :]
    ) # [N, P]
    wei_mask = (
          N_block_mask[:, None]
        & P_block_mask[None, :]
    ) # [N, P]
    att_weights = tl.load(
        wei_ptr+wei_offsets,
        wei_mask
    ) # [N, P]

    samples = _sample_bilinear(img_ptr, sampling_points, H, W, C, C_BLOCK)

    # weigh the points in samples
    weighted = tl.sum(
        att_weights[:, :, None] * samples,
        axis=1
    ) # [N, P, 1] * [N, P, C] -> [N, C]

    # store in the output
    out_offsets = (
          C * N_block_offsets[:, None]
        + 1 * C_block_offsets[None, :]
    ) # [N, C]
    out_mask = (
          N_block_mask[:, None]
        & C_block_mask[None, :]
    )
    tl.store(
        out_ptr+out_offsets,
        weighted,
        out_mask,
    )

def triton_deformable_att(img, sampling_points, att_weights):
    B, H, W, C = img.shape
    B, N, P, _ = sampling_points.shape

    # prepare output buffer
    output = sampling_points.new_empty(B, N, C)

    # call the kernel
    BLOCK_SIZE_N = triton.next_power_of_2(N)
    BLOCK_SIZE_C = triton.next_power_of_2(C)
    BLOCK_SIZE_P = triton.next_power_of_2(P)
    NUM_PROGS_B = B
    deformable_att_kernel[(NUM_PROGS_B, 1, 1)](
        output,
        img,
        sampling_points,
        att_weights,
        H, W, C, N, P,
        BLOCK_SIZE_N,
        BLOCK_SIZE_C,
        BLOCK_SIZE_P,
    )

    return output
