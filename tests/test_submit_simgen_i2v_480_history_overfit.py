import os
import subprocess
import sys
from pathlib import Path


def test_submit_script_is_valid_eight_h200_requeue_launcher() -> None:
    script_path = Path("submit_simgen_i2v_480_history_overfit.sh")

    result = subprocess.run(["bash", "-n", script_path], capture_output=True, text=True)
    script = script_path.read_text()

    assert result.returncode == 0, result.stderr
    assert "#SBATCH --partition=gpu_requeue" in script
    assert "#SBATCH --constraint=h200" in script
    assert "#SBATCH --nodes=2" in script
    assert "#SBATCH --ntasks=2" in script
    assert "#SBATCH --ntasks-per-node=1" in script
    assert "#SBATCH --gres=gpu:4" in script
    assert "#SBATCH --requeue" in script
    assert "configs/accelerate/h200_8gpu_2node.yaml" in script
    assert "srun" in script
    assert '--machine_rank "${SLURM_NODEID}"' in script
    assert '"${PROJECT_DIR}/cur.sif"' in script
    assert "train_i2v_simgen_480_overfit.py" in script
    assert "training.resume_from_checkpoint=latest" in script


def test_four_gpu_benchmark_launcher_requests_one_h200_node_and_starts_fresh(
    tmp_path: Path,
) -> None:
    """A benchmark launcher must not resume or use the two-node topology."""
    script_path = Path("submit_simgen_i2v_480_history_overfit_4gpu_1k.sh")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    captured_args = tmp_path / "singularity_args.txt"

    (bin_dir / "nvidia-smi").write_text("#!/bin/bash\nexit 0\n")
    (bin_dir / "singularity").write_text(
        "#!/bin/bash\nprintf '%s\\n' \"$@\" > \"${SINGULARITY_ARGS_FILE}\"\n"
    )
    (bin_dir / "nvidia-smi").chmod(0o755)
    (bin_dir / "singularity").chmod(0o755)

    environment = os.environ | {
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "SLURM_JOB_ID": "12345",
        "SINGULARITY_ARGS_FILE": str(captured_args),
    }
    result = subprocess.run(
        ["bash", script_path], capture_output=True, text=True, env=environment
    )
    script = script_path.read_text()

    assert result.returncode == 0, result.stderr
    assert "#SBATCH --partition=gpu_h200" in script
    assert "#SBATCH --nodes=1" in script
    assert "#SBATCH --gres=gpu:4" in script
    assert "#SBATCH --exclude=holygpu8a12204" in script
    assert "#SBATCH --requeue" not in script
    assert "configs/accelerate/h200_4gpu.yaml" in captured_args.read_text()
    assert "training.max_train_steps=2000" in captured_args.read_text()
    assert "training.resume_from_checkpoint" not in captured_args.read_text()
    assert "logging.output_dir=outputs/simgen_i2v_480_history_overfit_4gpu_1k" in captured_args.read_text()
    assert "logging.wandb_run_name=simgen-i2v-480-history-overfit-4gpu-1k" in captured_args.read_text()


def test_nccl_smoke_launcher_observes_two_node_transport_selection(
    tmp_path: Path,
) -> None:
    """The NCCL smoke test must observe auto-selected two-node transport."""
    script_path = Path("submit_nccl_smoke_8gpu_2node.sh")
    smoke_path = Path("nccl_smoke.py")

    result = subprocess.run(["bash", "-n", script_path], capture_output=True, text=True)
    script = script_path.read_text()
    compile_result = subprocess.run(
        [sys.executable, "-m", "py_compile", smoke_path], capture_output=True, text=True
    )
    smoke_source = smoke_path.read_text()

    assert result.returncode == 0, result.stderr
    assert compile_result.returncode == 0, compile_result.stderr
    assert "#SBATCH --nodes=2" in script
    assert "#SBATCH --ntasks=2" in script
    assert "#SBATCH --ntasks-per-node=1" in script
    assert "#SBATCH --gres=gpu:4" in script
    assert "configs/accelerate/h200_8gpu_2node.yaml" in script
    assert 'NCCL_DEBUG=INFO' in script
    assert 'NCCL_DEBUG_SUBSYS=INIT,NET,GRAPH' in script
    assert 'NCCL_DEBUG_FILE=' in script
    assert 'NCCL_SOCKET_IFNAME' not in script
    assert 'NCCL_IB_HCA' not in script
    assert 'NCCL_IB_DISABLE' not in script
    assert '-B /dev/infiniband' in script
    assert 'ibv_devices || true' in script
    assert 'ldconfig -p | grep libibverbs || true' in script
    assert 'RDMA_LIBRARIES' in script
    assert 'RDMA_LIBRARY_BIND_ARGS' in script
    assert '"${RDMA_LIBRARY_BIND_ARGS[@]}"' in script
    assert 'bash "${RDMA_LIBRARIES[@]}"' in script
    assert 'nccl_smoke.py' in script
    assert 'dist.init_process_group("nccl")' in smoke_source
    assert 'dist.all_reduce(tensor)' in smoke_source

    test_script_path = tmp_path / script_path.name
    test_script_path.write_text(
        script.replace(
            'PROJECT_DIR="/n/lab_storage/ydu_lab/jaysonzlin/Wan2.2_edited"',
            f'PROJECT_DIR="{tmp_path}"',
        )
    )
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for command, contents in {
        "scontrol": "#!/bin/bash\nprintf 'node-a\\nnode-b\\n'\n",
        "getent": "#!/bin/bash\nprintf '10.0.0.1 STREAM node-a\\n'\n",
        "ldconfig": (
            "#!/bin/bash\n"
            "printf '%s\\n' \\\n"
            "  'libibverbs.so.1 => /usr/lib64/libibverbs.so.1' \\\n"
            "  'libmlx5.so.1 => /usr/lib64/libmlx5.so.1' \\\n"
            "  'librdmacm.so.1 => /usr/lib64/librdmacm.so.1'\n"
        ),
        "nvidia-smi": "#!/bin/bash\nexit 0\n",
        "srun": (
            "#!/bin/bash\n"
            "while (($#)); do\n"
            "    if [[ $1 == bash ]]; then\n"
            "        shift\n"
            "        exec bash \"$@\"\n"
            "    fi\n"
            "    shift\n"
            "done\n"
        ),
        "singularity": (
            "#!/bin/bash\n"
            "while (($#)); do\n"
            "    if [[ $1 == bash ]]; then\n"
            "        shift\n"
            "        exec bash \"$@\"\n"
            "    fi\n"
            "    shift\n"
            "done\n"
        ),
        "accelerate": "#!/bin/bash\nexit 0\n",
    }.items():
        command_path = bin_dir / command
        command_path.write_text(contents)
        command_path.chmod(0o755)

    result = subprocess.run(
        ["bash", test_script_path],
        capture_output=True,
        text=True,
        env=os.environ
        | {
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "SLURM_JOB_ID": "12345",
            "SLURM_JOB_NODELIST": "node-a,node-b",
        },
    )

    assert result.returncode == 0, result.stderr
