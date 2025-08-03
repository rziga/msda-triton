# Multiscale deformable attention in Triton

A very naive implementation of multiscale deformable attention in triton.


## Performance

Here is a performance comparison with the PyTorch-native multiscale deformable attention:

<p align="center">
  <img src="assets/images/msda fwd runtime (ms).png" width="30%" />
  <img src="assets/images/msda fwd+bwd runtime (ms).png" width="30%" />
  <img src="assets/images/msda memory consumption (MB).png" width="30%" />
</p>

The results are also in line with the [original CUDA implementation](https://github.com/fundamentalvision/Deformable-DETR/tree/main/models/ops) from deformable DETR.
Running the same benchmark for CUDA, I get:
* FWD with 10k queries: 5.37 ms in CUDA vs. 3.78 ms in Triton (1.42x).
* FWD+BWD with 10k queries: 28.04 ms in CUDA vs. 22.78 ms in Triton (1.23x).
* Memory with 10k queries: 166.14 MB in CUDA vs. 166.14 MB in Triton (same).

*Results obtained on my RTX 2060 (gpu poor).*


## Detection Example

Replacing the original implementation with this kernel in a Deformable DETR-like model yields the same results.

*Model: [Grounding DINO](https://huggingface.co/docs/transformers/model_doc/grounding-dino), Image source: [COCO](http://images.cocodataset.org/val2017/000000039769.jpg)*

<p align="center">
  <img src="assets/images/torch_cats.png" width="45%" />
  <img src="assets/images/triton_cats.png" width="45%" />
</p>

> **Note**  
> The original version uses `padding_mode="zeros"` and `align_corners=False`. Make sure you match these arguments when using this implementation.


## Installation

### 1. Install PyTorch & Triton

This package **requires PyTorch and Triton**, but does not install them automatically.
Make sure you have them installed before proceeding.  

Check if Pytorch is installed with:
```sh
python -c "import torch; print(torch.__version__)"
```
It should print something like `2.6.0+cu124`.
If it does not, follow the [install instructions](https://pytorch.org/get-started/locally/).

Triton can already come bundled with PyTorch.
Check if it is installed with:
```sh
python -c "import triton; print(triton.__version__)"
```
It should print something like `3.2.0`.
If it does not, follow the [install instructions](https://triton-lang.org/main/getting-started/installation.html).


### 2. Install this package

Clone the repository:
```sh
git clone https://github.com/rziga/msda-triton
cd msda-triton
```

and install via `pip`:
```sh
pip install .
```


### 3. Run tests

You need to install `pytest` to run the tests in the `tests` directory.

Then run:
```sh
pytest ./tests
```


### 4. Run benchmark

To run the benchmark, you also need matplotlib and pandas.

Then run:
```sh
python scripts/benchmark.py
```
The results will be printed in terminal and saved in `outputs/benchmark_results` folder.


### 5. Debugging

Since triton can be pretty finnicky, I also provide the dependencies that I used for development.

Install via `pip`:
```sh
pip install -e .[dev]
```

or with `uv`:
```sh
uv sync --dev
```


## Usage

The package exposes two things:

1) `multiscale_deformable_attention` - a differentiable PyTorch function defining the multiscale deformable attention operator proposed in [Deformable DETR](https://arxiv.org/abs/2010.04159). Usage:

```py
import torch
from msda_triton import multiscale_deformable_attention

# input params
batch = 2
head_dim = 32
num_queries = 900
num_heads  = 8
num_points = 4
img_shapes = [(64, 64), (32, 32), (16, 16), (8, 8)]
num_pixels = sum(h * w for h, w in img_shapes)
num_levels = len(img_shapes)
device = "cuda" # "cpu" uses fallback native torch version

# generate random inputs
img = torch.randn(batch, num_pixels, num_heads, head_dim, device=device)
img_shapes = torch.tensor(img_shapes, device=device)
sampling_points = torch.rand(batch, num_queries, num_heads, num_levels, num_points, 2, device=device)
attention_weights = torch.rand(batch, num_queries, num_heads, num_levels, num_points, device=device)
padding_mode = "zeros" # or "border"
align_corners = False # or True

# perform MSDA
output = multiscale_deformable_attention(
    img, img_shapes, sampling_points, attention_weights,
    padding_mode, align_corners,
)

assert output.shape == (batch, num_queries, num_heads, head_dim)
```

2) `MultiscaleDeformableAttention` - a PyTorch `nn.Module`, which also handles the input and output projections. Usage:

```py
import torch
from msda_triton import MultiscaleDeformableAttention

# input params
batch = 2
emb_dim = 256
hidden_dim = 512
num_queries = 900
num_heads  = 8
num_points = 4
img_shapes = [(64, 64), (32, 32), (16, 16), (8, 8)]
num_pixels = sum(h * w for h, w in img_shapes)
num_levels = len(img_shapes)
device = "cuda" # "cpu" uses fallback native torch version

# generate random inputs
img = torch.randn(batch, num_pixels, emb_dim, device=device)
img_shapes = torch.tensor(img_shapes, device=device)
queries = torch.rand(batch, num_queries, emb_dim, device=device)
reference_points = torch.rand(batch, num_queries, 2, device=device)

# init module
msda = MultiscaleDeformableAttention(
    emb_dim,
    hidden_dim,
    num_levels,
    num_heads,
    num_points,
    padding_mode="border", # or "zeros"
    align_corners=True, # or False
).to(device)

# perform MSDA
output = msda(img, img_shapes, queries, reference_points)

assert output.shape == (batch, num_queries, emb_dim)
```


## Contributing

The kernels are quite basic, as this is my first experience with Triton. I have tested the functions as much as I could, but there could still be some issues. Performance can definitely be improved. Feel free to open an issue and/or submit improvements.


## License

[MIT](LICENSE)
