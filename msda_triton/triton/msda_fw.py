import triton
from triton import language as tl

@triton.jit
def get_nearest(coor):
    prev = tl.cast(tl.floor(coor), tl.int64)
    next = prev + 1
    return prev, next

@triton.jit
def sample_point(img_ptr, y, x, W, C, BLOCK_SIZE_C):
    offsets = (
          W*C * y[:, None]
        +   C * x[:, None]
        +   1 * tl.arange(0, BLOCK_SIZE_C)[None, :] 
    )
    mask = tl.arange(0, BLOCK_SIZE_C)[None, :] < C
    return tl.load(
        img_ptr+offsets,
        mask
    )

@triton.jit
def bilinear_sample_kernel(
    out_ptr, # output pointer
    img_ptr, # image pointer
    smp_ptr, # sampling points pointer
    H, # image height
    W, # image width
    C, # image channels
    S, # number of sampling points
    BLOCK_SIZE_S: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr, # assume BLOCK_SIZE_C > C
):
    #img_start = tl.program_id
    smp_start = tl.program_id(0) * BLOCK_SIZE_S
    smp_size = S * 2

    # load the sampling points
    smp_offsets = (
        + 2 * (smp_start + tl.arange(0, BLOCK_SIZE_S))[:, None]
        + 1 *              tl.arange(0,            2) [None, :]
    )
    smp_mask = smp_offsets < smp_size
    sampling_points = tl.load(
        smp_ptr+smp_offsets,
        smp_mask
    ) # [Q, 2]
    
    # unnormalize coordinates
    x, y = tl.split(sampling_points)
    x = x * (W - 1) # [Q]
    y = y * (H - 1) # [Q]

    # calculate the xy coordinates of nearest pixels
    xl, xr = get_nearest(x) # [Q], [Q]
    yt, yb = get_nearest(y) # [Q], [Q]
    
    # sample the neighbors
    img_tl = sample_point(img_ptr, yt, xl, W, C, BLOCK_SIZE_C) # [Q, C]
    img_tr = sample_point(img_ptr, yt, xr, W, C, BLOCK_SIZE_C) # [Q, C]
    img_bl = sample_point(img_ptr, yb, xl, W, C, BLOCK_SIZE_C) # [Q, C]
    img_br = sample_point(img_ptr, yb, xr, W, C, BLOCK_SIZE_C) # [Q, C]

    # calculate sampling weights
    wei_tl, wei_tr = (yb-y)*(xr-x), (yb-y)*(x-xl) # [Q], [Q]
    wei_bl, wei_br = (y-yt)*(xr-x), (y-yt)*(x-xl) # [Q], [Q]

    # sum the contribution of neighboring points
    sample = (
          wei_tl[:, None]*img_tl + wei_tr[:, None]*img_tr
        + wei_bl[:, None]*img_bl + wei_br[:, None]*img_br
    ) # [Q, C]

    # store in the output
    out_offsets = (
          C * (smp_start + tl.arange(0, BLOCK_SIZE_S))[:, None]
        + 1 * (            tl.arange(0, BLOCK_SIZE_C))[None, :]
    ) # [Q, C]
    out_mask = (
          (smp_start + tl.arange(0, BLOCK_SIZE_S) < S)[:, None]
        & (            tl.arange(0, BLOCK_SIZE_C) < C)[None, :]
    ) # [Q, C]
    tl.store(
        out_ptr+out_offsets,
        sample,
        out_mask,
    )

def bilinear_sample_triton(img, sampling_points):
    B, H, W, C = img.shape
    B, Ns, _ = sampling_points.shape

    # prepare output buffer
    output = sampling_points.new_empty(B, Ns, C)

    # call the kernel
    BLOCK_SIZE_Ns = triton.next_power_of_2(Ns)
    BLOCK_SIZE_Ns = 32
    NUM_PROGS_Ns  = triton.cdiv(Ns, BLOCK_SIZE_Ns)
    BLOCK_SIZE_C = triton.next_power_of_2(C)
    bilinear_sample_kernel[(NUM_PROGS_Ns, 1, 1)](
        output,
        img,
        sampling_points,
        H, W, C, Ns, 
        BLOCK_SIZE_Ns, BLOCK_SIZE_C
    )

    return output