# Joint Trajectory-Window Training Design

## Goal

Add a separate joint Wan--PhysCtrl trainer for controlled experiments over a
fixed point-cloud trajectory window. Each run selects a start frame `i` and
predicts `N` contiguous future frames while training the Wan video branch on
the same temporal segment.

## Window contract

The source sample remains a 49-frame clip. For one run, configuration fixes
`trajectory.start_frame = i` and `trajectory.future_frames = N`:

- The conditioned video and point cloud are source frame `i - 1`.
- PC targets are source frames `i` through `i + N - 1`, inclusive.
- The video clip is source frames `i - 1` through `i + N - 1`, inclusive.
- The PC model is constructed with `n_future_frames=N`.

The trainer accepts only `N` in `{4, 8, ..., 48}` and validates
`1 <= i <= 49 - N`. This ensures the entire segment exists and that the Wan
video clip has the supported `N + 1` temporal shape.

## Condition velocities

The existing HDF5 files provide velocities only at source frame zero. The new
dataset derives per-window condition velocities from the ordered point clouds:

- At `i=1`, it uses the forward difference from source frame 0 to source frame
  1.
- At `i>=2`, it uses the centered difference across source frames `i-2` and
  `i` to estimate the velocity at source frame `i-1`.
- Linear velocity is the point-cloud centroid velocity.
- Angular velocity is the least-squares rigid rotational component after
  subtracting that translation.

The derived values retain the existing `(1, 3)` linear and angular velocity
interfaces. The point indices are treated as particle correspondences across
frames.

## Trainer boundary

Create `train_joint_wan_physctrl_trajectory.py` and a dedicated training
configuration. The existing `train_joint_wan_physctrl.py` and its fixed
49-frame behavior remain unchanged. The new trainer uses a windowing dataset
adapter/helper, constructs matching DDPM tensors and PC positional embeddings,
and encodes only the matching video segment.

Rigid and deformation auxiliary losses are disabled for this experiment. The
optimization objective is the existing Wan video flow loss plus the sum of
per-object PC x0 MSEs.

## Experiment isolation

Every `(i, N)` experiment starts from the same Wan initialization and random
seed, writes to a distinct output directory, and does not warm-start from a
different horizon. Resume is valid only for a run with matching window
settings.

## Validation

Unit tests cover window bounds and slicing, velocity derivation (including the
`i=1` forward-difference case), Wan-compatible horizon validation, and a
variable-length PC DDPM batch. Trainer-local tests verify that auxiliary terms
remain disabled and that the PC model uses the configured horizon. Focused
tests run on CPU with tiny point clouds and no Wan checkpoint.
