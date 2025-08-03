import pytest
import torch
from msda_triton.frontend import triton_multiscale_deformable_attention


#@pytest.fixture(scope="session", autouse=True)
#def triton_warmup():
#    shapes = [(1024,), (2048,), (4096,)]  # Add all tested shapes
#    dtypes = [torch.float16, torch.float32, torch.bfloat16]  # Add all tested dtypes
#
#    for shape in shapes:
#        for dtype in dtypes:
#            x = torch.randn(*shape, dtype=dtype, device='cuda')
#            triton_multiscale_deformable_attention()