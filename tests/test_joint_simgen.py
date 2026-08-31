import json
import os
from pathlib import Path
import subprocess
import textwrap

import joint_simgen
import pytest
import torch
import yaml


class SumReducer:
    def __init__(self, reduced_values):
        self.reduced_values = iter(reduced_values)

    def reduce(self, value, reduction):
        assert reduction == "sum"
        return next(self.reduced_values)


def test_load_pretrained_pc_weights_strictly_initializes_pc_model(tmp_path):
    """A PC-only export must replace every downstream PC-model parameter."""
    pretrained = torch.nn.Linear(3, 2)
    with torch.no_grad():
        pretrained.weight.fill_(2.5)
        pretrained.bias.fill_(-0.5)
    weights_path = tmp_path / "pc_model.pt"
    torch.save(pretrained.state_dict(), weights_path)
    downstream = torch.nn.Linear(3, 2)

    joint_simgen.load_pretrained_pc_weights(downstream, weights_path)

    assert torch.equal(downstream.weight, torch.full((2, 3), 2.5))
    assert torch.equal(downstream.bias, torch.full((2,), -0.5))


def test_load_pretrained_pc_weights_rejects_incompatible_pc_model(tmp_path):
    """A changed PC architecture must not silently receive a partial export."""
    weights_path = tmp_path / "incompatible_pc_model.pt"
    torch.save({"weight": torch.ones(2, 3)}, weights_path)

    with pytest.raises(RuntimeError, match="Missing key.*bias"):
        joint_simgen.load_pretrained_pc_weights(torch.nn.Linear(3, 2), weights_path)


def test_prepare_cache_uses_sample_zero_panda_ball_can_sources(monkeypatch, tmp_path):
    metadata_path = tmp_path / "sample_0" / "metadata.json"
    metadata_path.parent.mkdir()
    metadata_path.write_text(
        json.dumps(
            {
                "instances": [
                    {"id": "001", "name": "panda"},
                    {"id": "000", "name": "ball"},
                    {"id": "002", "name": "can"},
                ]
            }
        )
    )
    config = {
        "data": {"dataset_root": str(tmp_path), "utonia_cache_root": str(tmp_path / "cache")},
        "training": {"seed": 0},
    }
    calls = []
    monkeypatch.setattr(joint_simgen, "UtoniaFeatureExtractor", lambda root, seed: object())
    monkeypatch.setattr(
        joint_simgen, "prepare_simgen_utonia_cache",
        lambda sources, root, extractor: calls.append((sources, Path(root))) or 8,
    )

    assert joint_simgen.prepare_cache(config) == 8
    assert calls[0][0] == {
        "panda": tmp_path / "sample_0" / "objects" / "001" / "pc.hdf5",
        "ball": tmp_path / "sample_0" / "objects" / "000" / "pc.hdf5",
        "can": tmp_path / "sample_0" / "objects" / "002" / "pc.hdf5",
    }


def test_build_datasets_uses_fixed_configured_train_and_validation_ranges(monkeypatch):
    calls = []

    class Dataset:
        def __init__(self, root, sample_ids, **kwargs):
            calls.append((root, sample_ids, kwargs))

    monkeypatch.setattr(joint_simgen, "SimGenJointDataset", Dataset)
    config = {
        "data": {
            "dataset_root": "simgen", "train_start": 0, "train_end": 489,
            "validation_start": 490, "validation_end": 499,
            "num_points": 2048, "utonia_cache_root": "cache",
        }
    }

    joint_simgen.build_datasets(config)

    assert calls[0][1] == list(range(490))
    assert calls[1][1] == list(range(490, 500))
    assert calls[0][2]["utonia_cache_root"] == "cache"


def test_main_dispatches_training_when_cache_preparation_is_not_requested(monkeypatch):
    monkeypatch.setattr(joint_simgen, "load_simgen_joint_config", lambda path, overrides: {"mode": "train"})
    calls = []
    monkeypatch.setattr(joint_simgen, "run_training", lambda config: calls.append(config))
    monkeypatch.setattr("sys.argv", ["joint_simgen.py", "--config", "config.yaml"])

    joint_simgen.main()

    assert calls == [{"mode": "train"}]


def test_main_rejects_distributed_cache_preparation(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setattr(joint_simgen, "load_simgen_joint_config", lambda *_: {"data": {}})
    monkeypatch.setattr(
        "sys.argv", ["joint_simgen.py", "--config", "config.yaml", "--prepare-utonia-cache"]
    )

    with pytest.raises(ValueError, match="single GPU"):
        joint_simgen.main()


def test_shared_generator_restarts_from_configured_seed():
    first = joint_simgen._shared_generator("cpu", 42)
    second = joint_simgen._shared_generator("cpu", 42)

    assert torch.equal(torch.rand(4, generator=first), torch.rand(4, generator=second))


def test_reduced_mean_uses_global_loss_sum_and_example_count():
    accelerator = SumReducer([torch.tensor(5.0), torch.tensor(2)])

    assert joint_simgen._reduced_mean(accelerator, torch.tensor(5.0), 2).item() == 2.5


def test_validation_metrics_include_mean_pc_loss():
    metrics = joint_simgen._validation_metrics(
        [torch.tensor(2.0), torch.tensor(4.0)],
        [torch.tensor(0.5), torch.tensor(1.5)],
    )

    assert metrics == {"validation/loss": 3.0, "validation/pc_loss_sum": 1.0}


def test_validation_loss_components_select_total_and_pc_loss_sum():
    total_loss, pc_loss_sum = joint_simgen._validation_loss_components(
        (torch.tensor(2.0), torch.tensor(0.75), torch.tensor(1.5), None)
    )

    assert total_loss.item() == 2.0
    assert pc_loss_sum.item() == 1.5


def test_visualization_history_matches_the_sampler_input_contract():
    point_clouds = torch.zeros(1, 3, 49, 1, 2048, 3)

    history = joint_simgen._visualization_history(point_clouds)

    assert history.shape == (3, 4, 2048, 3)


def test_4gpu_profile_has_four_processes_and_separate_output():
    accelerate_config = yaml.safe_load(
        Path("configs/accelerate/h200_4gpu.yaml").read_text()
    )
    training_config = yaml.safe_load(
        Path("configs/train/joint_simgen_480_4gpu.yaml").read_text()
    )

    assert accelerate_config["distributed_type"] == "MULTI_GPU"
    assert accelerate_config["num_processes"] == 4
    assert training_config["training"]["max_train_steps"] == 10_000
    assert training_config["logging"]["output_dir"] == "outputs/joint_simgen_4gpu"


def test_8gpu_profile_has_two_machines_and_long_run_settings():
    accelerate_config = yaml.safe_load(
        Path("configs/accelerate/h200_8gpu_2node.yaml").read_text()
    )
    training_config = yaml.safe_load(
        Path("configs/train/joint_simgen_480_8gpu.yaml").read_text()
    )

    assert accelerate_config["distributed_type"] == "MULTI_GPU"
    assert accelerate_config["num_machines"] == 2
    assert accelerate_config["num_processes"] == 8
    assert training_config["training"]["max_train_steps"] == 100_000
    assert training_config["training"]["checkpoint_every_steps"] == 1000
    assert training_config["visualization"]["every_steps"] == 1000
    assert training_config["validation"]["every_steps"] == 250
    assert training_config["logging"]["output_dir"] == "outputs/joint_simgen_8gpu"


def test_8gpu_launcher_exports_its_image_path_to_each_srun_task(tmp_path):
    """A task launched with a minimal environment must still find cur.sif."""
    tools = tmp_path / "tools"
    tools.mkdir()

    (tools / "scontrol").write_text("#!/bin/bash\necho gpu001\n")
    (tools / "getent").write_text("#!/bin/bash\necho '10.0.0.1 STREAM gpu001'\n")
    (tools / "nvidia-smi").write_text("#!/bin/bash\nexit 0\n")
    (tools / "singularity").write_text(
        "#!/bin/bash\n"
        "set -eu\n"
        "image=\n"
        "for argument in \"$@\"; do\n"
        "    [[ \"$argument\" == *.sif ]] && image=\"$argument\"\n"
        "done\n"
        "if [[ \"$image\" != \"/n/lab_storage/ydu_lab/jaysonzlin/Wan2.2_edited/cur.sif\" ]]; then\n"
        "    echo \"FATAL: could not open image $image\" >&2\n"
        "    exit 255\n"
        "fi\n"
        "echo \"image=$image\"\n"
    )
    (tools / "srun").write_text(
        textwrap.dedent(
            """\
            #!/bin/bash
            set -eu
            export_spec=""
            while [[ "$#" -gt 0 && "$1" != "bash" ]]; do
                case "$1" in
                    --export=*) export_spec="${1#--export=}" ;;
                esac
                shift
            done
            task_environment=("PATH=$PATH" "SLURM_NODEID=0")
            IFS=, read -ra entries <<< "$export_spec"
            for entry in "${entries[@]}"; do
                [[ "$entry" == *=* ]] && task_environment+=("$entry")
            done
            exec env -i "${task_environment[@]}" "$@"
            """
        )
    )
    for command in tools.iterdir():
        command.chmod(0o755)

    result = subprocess.run(
        ["bash", "submit_joint_simgen_8gpu_2node.sh"],
        cwd=Path.cwd(),
        env={
            **os.environ,
            "PATH": f"{tools}:{os.environ['PATH']}",
            "SLURM_JOB_ID": "12345",
            "SLURM_JOB_NODELIST": "gpu[001-002]",
        },
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "image=/n/lab_storage/ydu_lab/jaysonzlin/Wan2.2_edited/cur.sif" in result.stdout


def test_readme_documents_simgen_cache_preparation_and_training():
    readme = Path("README.md").read_text()

    assert "joint_simgen.py --prepare-utonia-cache" in readme
    assert "joint_simgen_480.yaml" in readme
