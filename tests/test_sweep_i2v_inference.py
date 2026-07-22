from pathlib import Path
import subprocess
import sys

import sweep_i2v_inference


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "sweep_i2v_inference.py"


def test_lists_the_cfg_scale_sweep_for_the_fixed_scheduler_experiment() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--list-experiments"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        "no_prompt_shift_1_steps_50_cfg_0.mp4",
        "no_prompt_shift_1_steps_50_cfg_0.5.mp4",
        "no_prompt_shift_1_steps_50_cfg_0.75.mp4",
        "no_prompt_shift_1_steps_50_cfg_1.mp4",
        "no_prompt_shift_1_steps_50_cfg_2.mp4",
        "no_prompt_shift_1_steps_50_cfg_5.mp4",
        "standard_negative_shift_1_steps_50_cfg_0.mp4",
        "standard_negative_shift_1_steps_50_cfg_0.5.mp4",
        "standard_negative_shift_1_steps_50_cfg_0.75.mp4",
        "standard_negative_shift_1_steps_50_cfg_1.mp4",
        "standard_negative_shift_1_steps_50_cfg_2.mp4",
        "standard_negative_shift_1_steps_50_cfg_5.mp4",
    ]


def test_configures_scheduler_to_begin_at_its_first_sigma() -> None:
    calls = []

    class FakeScheduler:
        def set_timesteps(self, num_steps, device, shift):
            calls.append(("set_timesteps", num_steps, device, shift))

        def set_begin_index(self, begin_index):
            calls.append(("set_begin_index", begin_index))

    sweep_i2v_inference.configure_scheduler(
        FakeScheduler(), num_steps=500, device="cuda", shift=5
    )

    assert calls == [
        ("set_timesteps", 500, "cuda", 5),
        ("set_begin_index", 0),
    ]
