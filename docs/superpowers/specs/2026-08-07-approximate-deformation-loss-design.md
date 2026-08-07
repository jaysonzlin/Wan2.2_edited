# Approximate Deformation Loss Design

## Goal

Train the joint Wan--PhysCtrl point-cloud branch with an optional, baseline-corrected approximation of PhysCtrl's deformation update for `td_832x480_3_soft`.

## Data contract

Each `sample_0/objects/<id>/pc.hdf5` retains its existing trajectory and velocity datasets and gains:

- `deform_F`: float32 `(49, 1, 2048, 3, 3)`, weighted-MLS deformation gradients derived from the fixed frame-zero KNN graph.
- `deform_C`: float32 `(49, 1, 2048, 3, 3)`, weighted-MLS affine velocity gradients derived from finite-difference GT velocities.
- `deform_volume`: float32 `(1, 2048)`, uniform particle quadrature weights.
- `deform_baseline`: float32 `(47, 1, 2048, 3, 3)`, the absolute deformation-update residual measured on the GT trajectory.
- `deform_grid_origin`: float32 `(3,)` and `deform_grid_scale`: float32 `(1,)`, the fixed sample-wide map from stored world coordinates into the loss grid.

The attributes `deform_dt=0.02`, `deform_grid_size=125`, and `deform_grid_lim=10.0` make the source conventions explicit. The grid map, derived fields, and baseline must match exactly across every object in the sample.

## Objective

The trainer concatenates the conditioned initial point cloud to its 48 x0-predicted future frames. It uses the GT `F`, `C`, and volume weights in a PhysCtrl-style P2G/G2P update, producing `r_pred = abs(F_update - F_next)`. The auxiliary term is the mean Smooth-L1 penalty of `relu(r_pred - r_gt)`, where `r_gt` is the saved `deform_baseline`.

The correction keeps GT motion at zero auxiliary loss despite the estimated, rather than native, MPM states. The existing x0 MSE continues to anchor the prediction to GT.

## Configuration

`objective.enable_deform_loss` gates both loading and evaluation. The configuration defaults to disabled and declares `deform_loss_weight: 0.001` and `deform_loss_neighbors: 32`. A disabled run retains the existing data and objective behavior.

## Validation

The preprocessing validates every generated tensor shape and reruns the approximate update on GT. It records the baseline per matrix entry. Tests cover the optional dataset contract, the disabled objective path, exact baseline cancellation, and the weighted total-loss calculation. A focused CPU test uses a tiny grid while the production path uses the recorded 125-cell grid.
