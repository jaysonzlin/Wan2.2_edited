# Rigid Point-Cloud Loss Design

## Goal

Regularize each rigid object trajectory in joint Wan--PhysCtrl training so its
predicted point cloud preserves the object's frame-zero local geometry. The
loss is logged per object and may be included in the optimization objective
through configuration.

## Scope

This change applies only to rigid objects. It does not add MPM data fields or
port PhysCtrl's deformation-gradient loss. A future MPM path may use
simulator-provided `vol`, `F`, and `C`, but is deliberately out of scope.

## Objective

For every object, construct a fixed directed k-nearest-neighbor graph from its
frame-zero point cloud. For every predicted future frame, penalize the
scale-normalized change in each graph edge's length:

\[
\frac{(\|\\hat{x}_{t,i} - \\hat{x}_{t,j}\|_2 - \|x_{0,i} - x_{0,j}\|_2)^2}
{\|x_{0,i} - x_{0,j}\|_2^2 + \epsilon}.
\]

The loss is averaged over future frames and selected edges, producing one
scalar per batch/object slot. It is invariant to a rigid translation or
rotation of the predicted object. The existing x0 loss continues to supervise
the correct absolute motion.

The objective utility accepts the initial point clouds with shape
`[B, K, 1, N, 3]` and future predictions with shape `[B, K, 48, 1, N, 3]` and
returns `[B, K]` losses. It rejects incompatible shapes and neighbor counts
outside `1 <= k < N`.

## Training and Logging

The joint-training configuration gains:

```yaml
objective:
  rigid_loss_weight: 0.0
  rigid_loss_neighbors: 16
```

Both values are validated: the weight must be non-negative and the neighbor
count must be a positive integer strictly smaller than `data.num_points`.

Every optimizer step calculates and logs `train/rigid_loss_sum` and one
`train/rigid_loss_object_<slot>` scalar for each object. The loss enters the
training objective only as `rigid_loss_weight * rigid_loss_sum`; the default
weight of zero preserves current optimization behavior while retaining the
diagnostic logs.

## Tests

Tests will demonstrate that the objective is zero for exact rigid transforms,
positive for non-rigid changes, returns independently addressable per-object
losses, and rejects invalid input. Configuration tests will cover the two new
settings and invalid values. Joint-loss tests will verify the configured
weighted contribution.
