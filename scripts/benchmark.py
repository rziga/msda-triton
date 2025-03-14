import triton
import torch

from msda_triton.torch_frontend import (
    native_multiscale_deformable_attention,
    triton_multiscale_deformable_attention,
)


@triton.testing.perf_report(
triton.testing.Benchmark(
    x_names=["num_queries"],
    x_vals=[10, 100, 300, 900, 1000, 10000],
    x_log=True,
    line_arg="provider",
    line_vals=["triton", "torch"],
    line_names=["Triton", "Torch"],
    styles=[("blue", "-"), ("green", "-")],
    ylabel="fwd runtime (ms)",
    plot_name="msda fwd runtime (ms)",
    args={},
))
def benchmark_fwd(num_queries, provider):
    N = num_queries
    B, H, C, P = 4, 8, 32, 4
    img_shapes = [(64, 64), (32, 32), (16, 16), (8, 8)]
    L = len(img_shapes)
    I = sum(h * w for h, w in img_shapes) # noqa: E741
    dtype = torch.float32
    padding_mode = "border"
    align_corners = True

    img = torch.randn(B, I, H, C, device="cuda", dtype=dtype)
    img_shapes = torch.tensor(img_shapes, device="cuda")
    sampling_points = torch.rand(B, N, H, L, P, 2, device="cuda", dtype=dtype)
    att_weights = torch.softmax(torch.randn(B, N, H, L, P, device="cuda", dtype=dtype), dim=-1)

    quantiles = [0.5, 0.2, 0.8]
    if provider == "triton":
        def run():
            with torch.no_grad():
                triton_multiscale_deformable_attention(
                    img, img_shapes, sampling_points, att_weights, padding_mode, align_corners
                )
    if provider == "torch":
        def run():
            with torch.no_grad():
                native_multiscale_deformable_attention(
                    img, img_shapes, sampling_points, att_weights, padding_mode, align_corners
                )

    output = triton.testing.do_bench(
        run, quantiles=quantiles, warmup=100, rep=1000,
    )
    return output


@triton.testing.perf_report(
triton.testing.Benchmark(
    x_names=["num_queries"],
    x_vals=[10, 100, 300, 900, 1000, 10000],
    x_log=True,
    line_arg="provider",
    line_vals=["triton", "torch"],
    line_names=["Triton", "Torch"],
    styles=[("blue", "-"), ("green", "-")],
    ylabel="fwd+bwd runtime (ms)",
    plot_name="msda fwd+bwd runtime (ms)",
    args={},
))
def benchmark_fwdbwd(num_queries, provider):
    N = num_queries
    B, H, C, P = 4, 8, 32, 4
    img_shapes = [(64, 64), (32, 32), (16, 16), (8, 8)]
    L = len(img_shapes)
    I = sum(h * w for h, w in img_shapes) # noqa: E741
    dtype = torch.float32
    padding_mode = "border"
    align_corners = True

    img = torch.randn(B, I, H, C, device="cuda", dtype=dtype, requires_grad=True)
    img_shapes = torch.tensor(img_shapes, device="cuda")
    sampling_points = torch.rand(B, N, H, L, P, 2, device="cuda", dtype=dtype, requires_grad=True)
    att_weights = torch.softmax(torch.randn(B, N, H, L, P, device="cuda", dtype=dtype), dim=-1)
    att_weights.requires_grad_(True)

    quantiles = [0.5, 0.2, 0.8]
    if provider == "triton":
        def run():
            out = triton_multiscale_deformable_attention(
                img, img_shapes, sampling_points, att_weights, padding_mode, align_corners
            )
            out.backward(torch.rand_like(out))
            img.grad = sampling_points.grad = att_weights.grad = None

    if provider == "torch":
        def run():
            out = native_multiscale_deformable_attention(
                img, img_shapes, sampling_points, att_weights, padding_mode, align_corners
            )
            out.backward(torch.rand_like(out))
            img.grad = sampling_points.grad = att_weights.grad = None

    output = triton.testing.do_bench(
        run, quantiles=quantiles, warmup=100, rep=1000,
    )
    return output


@triton.testing.perf_report(
triton.testing.Benchmark(
    x_names=["num_queries"],
    x_vals=[10, 100, 300, 900, 1000, 10000],
    x_log=True,
    line_arg="provider",
    line_vals=["triton", "torch"],
    line_names=["Triton", "Torch"],
    styles=[("blue", "-"), ("green", "-")],
    ylabel="memory consumption (MB)",
    plot_name="msda memory consumption (MB)",
    args={},
))
def benchmark_memory(num_queries, provider):
    N = num_queries
    B, H, C, P = 4, 8, 32, 4
    img_shapes = [(64, 64), (32, 32), (16, 16), (8, 8)]
    L = len(img_shapes)
    I = sum(h * w for h, w in img_shapes) # noqa: E741
    dtype = torch.float32
    padding_mode = "border"
    align_corners = True

    img = torch.randn(B, I, H, C, device="cuda", dtype=dtype, requires_grad=True)
    img_shapes = torch.tensor(img_shapes, device="cuda")
    sampling_points = torch.rand(B, N, H, L, P, 2, device="cuda", dtype=dtype, requires_grad=True)
    att_weights = torch.softmax(torch.randn(B, N, H, L, P, device="cuda", dtype=dtype), dim=-1)
    att_weights.requires_grad_(True)

    if provider == "triton":
        def run():
            out = triton_multiscale_deformable_attention(
                img, img_shapes, sampling_points, att_weights, padding_mode, align_corners
            )
            out.backward(torch.rand_like(out))
            img.grad = sampling_points.grad = att_weights.grad = None

    if provider == "torch":
        def run():
            out = native_multiscale_deformable_attention(
                img, img_shapes, sampling_points, att_weights, padding_mode, align_corners
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
