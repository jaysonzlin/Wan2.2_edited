"""Measure a fixed-size NCCL all-reduce across a PyTorch process group."""

import datetime
import os
import socket
import time

import torch
import torch.distributed as dist
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs


TENSOR_MEBIBYTES = 256
WARMUP_ITERATIONS = 5
MEASURED_ITERATIONS = 20
FLOAT32_BYTES = 4
NCCL_SMOKE_INIT_TIMEOUT_SECONDS = int(
    os.environ.get("NCCL_SMOKE_INIT_TIMEOUT_SECONDS", "120")
)


def main() -> None:
    init_process_group_kwargs = InitProcessGroupKwargs(
        backend="nccl",
        timeout=datetime.timedelta(seconds=NCCL_SMOKE_INIT_TIMEOUT_SECONDS),
    )
    accelerator = Accelerator(kwargs_handlers=[init_process_group_kwargs])

    if accelerator.device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("NCCL smoke test requires a CUDA-capable GPU")

    if accelerator.num_processes != 2:
        raise RuntimeError(
            "NCCL smoke test requires exactly two Accelerate processes; "
            f"received {accelerator.num_processes}"
        )

    torch.cuda.set_device(accelerator.local_process_index)
    if not dist.is_initialized():
        raise RuntimeError(
            "Accelerate did not initialize the NCCL process group; "
            "launch this test with accelerate launch"
        )

    try:
        tensor = torch.ones(
            TENSOR_MEBIBYTES * 1024 * 1024 // FLOAT32_BYTES,
            device="cuda",
            dtype=torch.float32,
        )
        for _ in range(WARMUP_ITERATIONS):
            tensor.fill_(1)
            dist.all_reduce(tensor)
        torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(MEASURED_ITERATIONS):
            tensor.fill_(1)
            dist.all_reduce(tensor)
        torch.cuda.synchronize()

        expected_value = float(accelerator.num_processes)
        if not torch.all(tensor == expected_value):
            raise RuntimeError(
                "NCCL all-reduce returned an unexpected value; "
                f"expected every element to equal {expected_value}"
            )

        seconds_per_all_reduce = (time.perf_counter() - start) / MEASURED_ITERATIONS
        logical_bandwidth_gbps = TENSOR_MEBIBYTES / seconds_per_all_reduce / 1000
        print(
            f"rank={accelerator.process_index} host={socket.gethostname()} "
            f"allreduce_{TENSOR_MEBIBYTES}MiB={seconds_per_all_reduce:.4f}s "
            f"logical_bw={logical_bandwidth_gbps:.2f} GB/s",
            flush=True,
        )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
