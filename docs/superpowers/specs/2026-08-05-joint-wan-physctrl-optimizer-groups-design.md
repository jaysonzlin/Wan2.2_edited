# Joint Wan--PhysCtrl optimizer groups

## Scope

Joint training uses one AdamW optimizer with three parameter groups:

- `video`: `JointWanPhysCtrlModel.wan_model`
- `bca`: `JointWanPhysCtrlModel.bridges`
- `pc`: `JointWanPhysCtrlModel.pc_model`

`configs/train/joint_wan_physctrl_832x480.yaml` exposes `lr`, `betas`, `eps`, and
`weight_decay` for each group. The video and BCA defaults retain the current joint
training values (`1e-5`, `[0.9, 0.95]`, `1e-8`, `0.1`). PC uses the established
point-cloud defaults from `configs/train/config_pc.yaml` (`1e-4`, `[0.9, 0.999]`,
`1e-8`, `1e-2`).

The existing learning-rate scheduler receives the multi-group optimizer, so it
applies its normal schedule to every group while preserving their configured base
learning-rate ratios.

## Observability

The trainer calculates the global L2 norm of PC-model gradients immediately after
backpropagation and before clipping, matching the timing and definition of the
existing bridge and video gradient metrics. It logs the value as
`train/pc_gradient_norm`.

## Configuration and checkpoint behavior

Validation requires all three optimizer groups and validates their four AdamW
fields. The single-group optimizer checkpoint format is intentionally not migrated;
resuming one of those checkpoints requires a training restart.

## Tests

Focused unit tests will verify distinct parameter groups and hyperparameters, PC
gradient-norm isolation, and acceptance of the nested optimizer configuration.
