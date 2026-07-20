# PC Visualization Parity Design

## Goal

Make the point-cloud comparison MP4 written by `train_pc.py` visually match
`edited-physctrl`'s PC comparison visualization, without coupling the Wan
repository to PhysCtrl or changing the PC training/model contract.

## Scope

Only replace the internals of
`training.pc_visualization.save_pointcloud_comparison_mp4`.  Its public
signature, training call site, configured cadence, output directory, and MP4
format remain unchanged.

## Rendering contract

The renderer will accept matching `(frames, objects, points, 3)` prediction and
ground-truth arrays and produce a two-panel MP4.

For every frame it will:

- color each object's points from that object's initial-frame height with the
  Viridis colormap;
- use common, cubic X/Y/Z limits derived from both trajectories, with the Z
  lower bound at zero;
- render XYZ labels, a grid, and per-object legend entries;
- title the panes `Prediction` and `Ground Truth`;
- title the figure with the zero-padded frame index plus PhysCtrl-equivalent
  position error (mean object-centroid distance) and mean point error.

The existing shape validation remains: arrays must match and have shape
`(frames, objects, points, 3)`.

## Design boundaries

Wan keeps a self-contained implementation rather than importing PhysCtrl.  A
small set of helpers owns color assignment, trajectory error calculation, axis
limit calculation, and single-pane drawing.  The public MP4 writer owns file
creation and cleanup only.

## Testing

Tests will cover the numerical PE/ME calculation, stable height-color behavior
for flat objects, malformed input rejection, and successful MP4 output.  The
existing training visualization cadence test remains unchanged because the
training integration does not change.

## Non-goals

- Changing the model, flow objective, scheduler, dataset, or visualization
  cadence.
- Adding PhysCtrl as a runtime dependency.
- Adding a standalone trajectory-only renderer to Wan.
