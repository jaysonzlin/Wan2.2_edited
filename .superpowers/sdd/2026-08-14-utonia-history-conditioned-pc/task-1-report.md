# Task 1 report: conditioning configuration and dataset splitting

## Scope completed

- `training/pc_config.py` now treats missing `model.conditioning` as the
  compatible velocity path, accepts explicit `velocity`, and rejects unknown
  modes. `history` requires `model.history_frames: 4`.
- `training/pc_dataset.py` accepts `history_frames` (default `1`). The default
  sample remains `points_src = frame 0` and `points_tgt = frames 1--48`.
  With four history frames, it additionally returns `points_history = frames
  0--3` and returns `points_tgt = frames 4--48` (45 frames).
- Tests cover default/explicit velocity compatibility, accepted four-frame
  history configuration, invalid history configuration fields, and an HDF5
  clip whose per-frame values prove the 4/45 boundary.

## Test-driven development evidence

### Red

Command:

```bash
PYTHONPATH=. /Users/jaysonlin/miniconda3/envs/utonia-dev/bin/python -m pytest tests/test_pc_config.py tests/test_pc_dataset.py -q
```

Result before the implementation: **5 failed, 33 passed** in 0.63s.

- Four invalid history/conditioning configurations were silently accepted.
- `PCTrajectoryDataset(..., history_frames=4)` raised `TypeError` because the
  argument was not implemented.

### Green

The same focused command after the minimal implementation reported:

```text
38 passed in 0.61s
```

Additional verification:

```bash
PYTHONPATH=. /Users/jaysonlin/miniconda3/envs/utonia-dev/bin/python -m compileall -q training/pc_config.py training/pc_dataset.py
git diff --check
```

Both commands completed successfully with no output.

## Environment note

The default interpreter lacked project dependencies; `pc_env` had h5py but no
PyTorch, pytest, or PyYAML. The documented `wan2-2` Conda environment was not
present. The available `utonia-dev` environment provided h5py 3.16.0, PyTorch
2.13.0, pytest 9.1.1, and PyYAML 6.0.3, and was used for the executable tests.
The unavailable `black` module prevented a Black check; the changed code was
manually kept in the repository's Black-compatible style.
