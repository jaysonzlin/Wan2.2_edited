"""Measure a fixed-size NCCL all-reduce across an Accelerate process group."""

import os
import socket
import time

import torch
import torch.distributed as dist


TENSOR_MEBIBYTES = 256
WARMUP_ITERATIONS = 5
MEASURED_ITERATIONS = 20
FLOAT32_BYTES = 4


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("NCCL smoke test requires a CUDA-capable GPU")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")

    try:
        tensor = torch.ones(
            TENSOR_MEBIBYTES * 1024 * 1024 // FLOAT32_BYTES,
            device="cuda",
            dtype=torch.float32,
        )
        for _ in range(WARMUP_ITERATIONS):
            dist.all_reduce(tensor)
        torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(MEASURED_ITERATIONS):
            dist.all_reduce(tensor)
        torch.cuda.synchronize()

        seconds_per_all_reduce = (time.perf_counter() - start) / MEASURED_ITERATIONS
        logical_bandwidth_gbps = TENSOR_MEBIBYTES / seconds_per_all_reduce / 1000
        print(
            f"rank={dist.get_rank()} host={socket.gethostname()} "
            f"allreduce_{TENSOR_MEBIBYTES}MiB={seconds_per_all_reduce:.4f}s "
            f"logical_bw={logical_bandwidth_gbps:.2f} GB/s",
            flush=True,
        )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
