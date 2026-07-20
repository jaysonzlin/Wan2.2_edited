# PC Flow Absolute Input Design

## Goal

Replace the PC flow model's mixed coordinate input with absolute noisy future
positions so all frames passed to `PointEmbed` use the same XYZ reference
frame, matching PhysCtrl's frame-conditioning representation.

## Change

`PCFlowModel.forward` continues to receive the displacement-space flow state
`x_t` and the initial cloud `p0`.  Before point embedding, it will construct
the 49-frame coordinate sequence as:

```text
[p0, p0 + x_t]
```

where `p0 + x_t` broadcasts `p0` across the 48 future frames.

## Preserved contract

- Flow training still constructs `x_t` from future displacements and predicts
  `epsilon - displacement`.
- The model head still returns a displacement-space flow vector.
- The sampler still integrates a displacement-space state and adds `p0` once,
  after integration.
- The two velocity controls, fixed 49/2048 data shape, PointEmbed, and
  factorized model topology are unchanged.
- This is the sole behavior: there is no configuration switch for the old
  mixed representation.

## Rationale

The old sequence `[p0, x_t]` supplied an absolute point cloud at frame zero
and displacement/noise vectors for future frames.  The new sequence supplies
position-coordinate point clouds at every frame while retaining the same flow
state and objective.  It is therefore a representation change at the model
input boundary, not an objective, control, or sampling change.

## Validation

Update the model unit test to intercept the tensor received by `input_encoder`
and assert that future frames equal `p0 + x_t`.  Retain the existing test that
confirms a zero flow head does not add source coordinates to its output, and
run the full pytest suite.
