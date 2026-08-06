# Rigid Loss Toggle Design

## Goal

Make rigid-loss computation and rigid-loss logging optional in joint Wan--PhysCtrl training.

## Configuration

Add `objective.enable_rigid_loss` to `configs/train/joint_wan_physctrl_832x480.yaml` with a default of `false`. The joint configuration loader accepts only booleans for this setting and rejects all other values.

`rigid_loss_weight` and `rigid_loss_neighbors` remain validated as they are today. They take effect only when `enable_rigid_loss` is true.

## Trainer behavior

When disabled, `train_joint_wan_physctrl.py` does not call `per_object_rigid_edge_length_loss`. It supplies a scalar zero rigid term to the existing total-loss helper, so the total remains the video loss plus the sum of point-cloud losses.

When disabled, metrics omit `train/rigid_loss_sum` and all `train/rigid_loss_object_*` keys. All video, point-cloud, gradient-norm, learning-rate, checkpoint, and visualization behavior remains unchanged.

When enabled, rigid-loss computation, weighting, total-loss contribution, and metrics are unchanged from the current implementation.

## Validation

Tests cover configuration acceptance and rejection, metric inclusion/exclusion, and both trainer paths. The disabled-path test patches the rigid-loss function to raise if invoked, proving the expensive neighbor computation is skipped.
