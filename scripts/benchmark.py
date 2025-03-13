import triton
import torch

from msda_triton.torch_frontend import (
    native_multiscale_deformable_attention,
    triton_multiscale_deformable_attention,
)


@triton.testing.perf_report(
triton.testing.Benchmark(
    x_names=["num_queries"],  # Argument names to use as an x-axis for the plot.
    x_vals=[10, 100, 300, 900, 1000, 10000],  # Different possible values for `x_name`.
    x_log=True,  # x axis is logarithmic.
    line_arg="provider",  # Argument name whose value corresponds to a different line in the plot.
    line_vals=["triton", "torch"],  # Possible values for `line_arg`.
    line_names=["Triton", "Torch"],  # Label name for the lines.
    styles=[("blue", "-"), ("green", "-")],  # Line styles.
    ylabel="fwd runtime (ms)",  # Label name for the y-axis.
    plot_name="msda fwd runtime (ms)",  # Name for the plot. Used also as a file name for saving the plot.
    args={},  # Values for function arguments not in `x_names` and `y_name`.
))
def benchmark_fwd(num_queries, provider):
    N = num_queries
    B, H, C, P = 4, 8, 32, 4
    img_shapes = [(64, 64), (32, 32), (16, 16), (8, 8)]
    L = len(img_shapes)
    I = sum(h * w for h, w in img_shapes)
    dtype = torch.float32
    padding_mode = "border"

    img = torch.randn(B, I, H, C, device="cuda", dtype=dtype)
    img_shapes = torch.tensor(img_shapes, device="cuda")
    sampling_points = torch.rand(B, N, H, L, P, 2, device="cuda", dtype=dtype)
    att_weights = torch.softmax(torch.randn(B, N, H, L, P, device="cuda", dtype=dtype), dim=-1)

    quantiles = [0.5, 0.2, 0.8]
    if provider == "triton":
        def run():
            with torch.no_grad():
                triton_multiscale_deformable_attention(
                    img, img_shapes, sampling_points, att_weights, padding_mode
                )
    if provider == "torch":
        def run():
            with torch.no_grad():
                native_multiscale_deformable_attention(
                    img, img_shapes, sampling_points, att_weights, padding_mode
                )

    output = triton.testing.do_bench(
        run, quantiles=quantiles, warmup=100, rep=1000,
    )
    return output


@triton.testing.perf_report(
triton.testing.Benchmark(
    x_names=["num_queries"],  # Argument names to use as an x-axis for the plot.
    x_vals=[10, 100, 300, 900, 1000, 10000],  # Different possible values for `x_name`.
    x_log=True,  # x axis is logarithmic.
    line_arg="provider",  # Argument name whose value corresponds to a different line in the plot.
    line_vals=["triton", "torch"],  # Possible values for `line_arg`.
    line_names=["Triton", "Torch"],  # Label name for the lines.
    styles=[("blue", "-"), ("green", "-")],  # Line styles.
    ylabel="fwd+bwd runtime (ms)",  # Label name for the y-axis.
    plot_name="msda fwd+bwd runtime (ms)",  # Name for the plot. Used also as a file name for saving the plot.
    args={},  # Values for function arguments not in `x_names` and `y_name`.
))
def benchmark_fwdbwd(num_queries, provider):
    N = num_queries
    B, H, C, P = 4, 8, 32, 4
    img_shapes = [(64, 64), (32, 32), (16, 16), (8, 8)]
    L = len(img_shapes)
    I = sum(h * w for h, w in img_shapes)
    dtype = torch.float32
    padding_mode = "border"

    img = torch.randn(B, I, H, C, device="cuda", dtype=dtype, requires_grad=True)
    img_shapes = torch.tensor(img_shapes, device="cuda")
    sampling_points = torch.rand(B, N, H, L, P, 2, device="cuda", dtype=dtype, requires_grad=True)
    att_weights = torch.softmax(torch.randn(B, N, H, L, P, device="cuda", dtype=dtype), dim=-1)
    att_weights.requires_grad_(True)

    quantiles = [0.5, 0.2, 0.8]
    if provider == "triton":
        def run():
            out = triton_multiscale_deformable_attention(
                img, img_shapes, sampling_points, att_weights, padding_mode
            )
            out.backward(torch.rand_like(out))
            img.grad = sampling_points.grad = att_weights.grad = None

    if provider == "torch":
        def run():
            out = native_multiscale_deformable_attention(
                img, img_shapes, sampling_points, att_weights, padding_mode
            )
            out.backward(torch.rand_like(out))
            img.grad = sampling_points.grad = att_weights.grad = None

    output = triton.testing.do_bench(
        run, quantiles=quantiles, warmup=100, rep=1000,
    )
    return output


@triton.testing.perf_report(
triton.testing.Benchmark(
    x_names=["num_queries"],  # Argument names to use as an x-axis for the plot.
    x_vals=[10, 100, 300, 900, 1000, 10000],  # Different possible values for `x_name`.
    x_log=True,  # x axis is logarithmic.
    line_arg="provider",  # Argument name whose value corresponds to a different line in the plot.
    line_vals=["triton", "torch"],  # Possible values for `line_arg`.
    line_names=["Triton", "Torch"],  # Label name for the lines.
    styles=[("blue", "-"), ("green", "-")],  # Line styles.
    ylabel="memory consumption (MB)",  # Label name for the y-axis.
    plot_name="msda memory consumption (MB)",  # Name for the plot. Used also as a file name for saving the plot.
    args={},  # Values for function arguments not in `x_names` and `y_name`.
))
def benchmark_memory(num_queries, provider):
    N = num_queries
    B, H, C, P = 4, 8, 32, 4
    img_shapes = [(64, 64), (32, 32), (16, 16), (8, 8)]
    L = len(img_shapes)
    I = sum(h * w for h, w in img_shapes)
    dtype = torch.float32
    padding_mode = "border"

    img = torch.randn(B, I, H, C, device="cuda", dtype=dtype, requires_grad=True)
    img_shapes = torch.tensor(img_shapes, device="cuda")
    sampling_points = torch.rand(B, N, H, L, P, 2, device="cuda", dtype=dtype, requires_grad=True)
    att_weights = torch.softmax(torch.randn(B, N, H, L, P, device="cuda", dtype=dtype), dim=-1)
    att_weights.requires_grad_(True)

    if provider == "triton":
        def run():
            out = triton_multiscale_deformable_attention(
                img, img_shapes, sampling_points, att_weights, padding_mode
            )
            out.backward(torch.rand_like(out))
            img.grad = sampling_points.grad = att_weights.grad = None

    if provider == "torch":
        def run():
            out = native_multiscale_deformable_attention(
                img, img_shapes, sampling_points, att_weights, padding_mode
            )
            out.backward(torch.rand_like(out))
            img.grad = sampling_points.grad = att_weights.grad = None


    warmups = 10
    for _ in range(warmups):
        run()

    repeats = 100
    mem_usage = 0
    for _ in range(repeats):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start_mem = torch.cuda.memory_allocated()
        
        run()
        
        torch.cuda.synchronize()
        max_mem = torch.cuda.max_memory_allocated()
        mem_usage += (max_mem - start_mem) / 1e6 # convert bytes to MB
    mem_usage /= repeats

    return mem_usage


if __name__ == "__main__":
    benchmark_fwd.run(print_data=True, show_plots=False, save_path="outputs/benchmark_results/fwd")
    benchmark_fwdbwd.run(print_data=True, show_plots=False, save_path="outputs/benchmark_results/fwdbwd")
    benchmark_memory.run(print_data=True, show_plots=False, save_path="outputs/benchmark_results/memory")
