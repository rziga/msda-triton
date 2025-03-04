# Multiscale deformable attention in Triton

A very naive implementation of multiscale deformable attention in triton.

## Installation

### 1. Install PyTorch & Triton

This package **requires PyTorch and Triton**, but does not install them automatically.
Make sure you have PyTorch installed before proceeding.  

Install PyTorch following the instructions at [pytorch.org](https://pytorch.org/get-started/locally/).

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

```sh
pytest ./tests
```

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