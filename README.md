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
Running the same benchmark for CUDA I get:
* FWD with 10k queries: 5.37 ms in CUDA vs. 4.81 ms in Triton.
* FWD+BWD with 10k queries: 28.04 ms in CUDA vs. 24.95 ms in Triton.
* Memory with 10k queries: 166.14 MB in CUDA vs 166.14 MB in Triton.

*Results obtained on my RTX 2060 (gpu poor).*

> **Note**
> This version uses `padding_mode = "border"`, while the original version uses `padding_mode = "zeros"`, so they are likely not 1-1 compatible.

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

and install using `pip`:
```sh
pip install .
```
or using `uv`:
```sh
uv install
```


### 3. Run tests

You need to install `pytest` to run the tests in the `tests` directory.

Then run:
```sh
pytest ./tests
```

> **Note**  
> The `float32` backward test sometimes fails on my machine. I have spent quite a lot of time debugging this and came to the conclusion that it's probably due to rounding errors when doing bilinear sampling. 


### 4. Run benchmark

To run the benchmark:
```sh
python scripts/benchmark
```
The results will be printed in terminal and saved in `outputs/benchmakr_resuls` folder.


### Debugging

Since triton can be pretty finnicky, I also provide the dependencies that I used for development.

Using `pip`:
```sh
pip install -e .[dev]
```
or using `uv`:
```sh
uv install --dev
```


## Contributing

The kernels are very basic. This is pretty much my first experience with triton. I have tested the functions as much as I could, but there could still be some issues. Performance can definitely be improved. Feel free to open an issue and/or submit improvements.