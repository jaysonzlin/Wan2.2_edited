"""Static contract checks for the Mamba-based two-node smoke test.

These run without CUDA, PyTorch, or a Slurm allocation.  The actual benchmark
is intentionally exercised on the cluster by the Slurm launcher.
"""

import ast
import pathlib
import subprocess
import unittest


PROJECT_DIR = pathlib.Path(__file__).resolve().parents[1]


class MambaAccelerateSmokeContractTest(unittest.TestCase):
    def test_environment_builder_exposes_shared_prefix_and_recreate_guard(self):
        script = PROJECT_DIR / "create_mamba_env.sh"
        self.assertTrue(script.is_file())
        source = script.read_text()
        self.assertIn("/n/holylabs/ydu_lab/Lab/jaysonzlin/wan2-2-mamba", source)
        self.assertIn("--recreate", source)
        self.assertIn("--refresh-metadata", source)
        self.assertIn("CONDA_PKGS_DIRS", source)
        self.assertIn("MAMBA_PKGS_DIRS", source)
        self.assertIn("clean --yes --index-cache", source)
        self.assertIn("torch==2.4.1", source)
        self.assertIn("accelerate>=1.1.1", source)
        self.assertIn("export PYTHONNOUSERSITE=1", source)
        self.assertIn("--no-user", source)
        self.assertIn('"numpy>=1.23.5,<2"', source)
        self.assertIn('"opencv-python>=4.9.0.80,<5"', source)
        self.assertIn('"${PYTHON}" -m pip check', source)
        self.assertIn("PIP_NO_USER_ARGS=(--no-user)", source)
        self.assertIn('"${PIP_NO_USER_ARGS[@]}" --no-cache-dir', source)
        self.assertNotIn('"${PIP_INSTALL_ARGS[@]}" --no-cache-dir', source)
        self.assertEqual(subprocess.run(["bash", "-n", script], check=False).returncode, 0)

    def test_slurm_launcher_runs_two_nodes_through_accelerate_without_singularity(self):
        script = PROJECT_DIR / "submit_accelerate_smoke_2gpu_2node_mamba.sh"
        self.assertTrue(script.is_file())
        source = script.read_text()
        self.assertIn("#SBATCH --nodes=2", source)
        self.assertIn("#SBATCH --gres=gpu:nvidia_h200:1", source)
        self.assertIn("h200_2gpu_2node.yaml", source)
        self.assertIn("accelerate", source)
        self.assertIn("export PYTHONNOUSERSITE=1", source)
        self.assertNotIn("singularity", source.lower())
        self.assertEqual(subprocess.run(["bash", "-n", script], check=False).returncode, 0)

    def test_launcher_uses_shared_checkout_and_log_directory(self):
        script = PROJECT_DIR / "submit_accelerate_smoke_2gpu_2node_mamba.sh"
        source = script.read_text()

        self.assertIn(
            "#SBATCH --output=/n/lab_storage/ydu_lab/jaysonzlin/Wan2.2_edited/logs/"
            "accelerate_smoke_2gpu_2node_mamba_%j.out",
            source,
        )
        self.assertIn(
            "#SBATCH --error=/n/lab_storage/ydu_lab/jaysonzlin/Wan2.2_edited/logs/"
            "accelerate_smoke_2gpu_2node_mamba_%j.err",
            source,
        )
        self.assertIn(
            'PROJECT_DIR="${PROJECT_DIR:-/n/lab_storage/ydu_lab/jaysonzlin/Wan2.2_edited}"',
            source,
        )
        self.assertNotIn('SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"', source)

    def test_benchmark_initializes_accelerator_and_requires_two_processes(self):
        tree = ast.parse((PROJECT_DIR / "nccl_smoke.py").read_text())
        imported_names = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module == "accelerate"
            for alias in node.names
        }
        self.assertIn("Accelerator", imported_names)

        attributes = {
            node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
        }
        self.assertIn("num_processes", attributes)

        source = (PROJECT_DIR / "nccl_smoke.py").read_text()
        self.assertIn("InitProcessGroupKwargs", source)
        self.assertIn("NCCL_SMOKE_INIT_TIMEOUT_SECONDS", source)


if __name__ == "__main__":
    unittest.main()
