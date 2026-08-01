# Joint Wan–PhysCtrl training design

## Goal

Train a joint model that generates a 49-frame Wan 2.2 TI2V video and one
48-future-frame PhysCtrl point-cloud trajectory per supplied object. The model
uses bidirectional cross-attention between Wan's last eight DiT blocks and all
eight PhysCtrl blocks, so video generation and object trajectories can improve
one another.

## Scope

The deliverable includes a paired-dataset trainer, checkpoint/resume support,
a synchronized joint sampler, and periodic video plus trajectory
visualization. It does not add a learned velocity estimator, a new point-cloud
renderer, multi-video batches, trajectory pretraining, text prompting, or a
separate sampling service.

## Model

Load the base Wan 2.2 TI2V 5B checkpoint. The Wan DiT is trainable; only the
Wan VAE and text encoder are frozen. Initialize the PhysCtrl trajectory model
and every new bridge module from scratch.

Pair Wan blocks 22 through 29 with PhysCtrl blocks 0 through 7. The paired
block order is:

1. Wan self-attention.
2. Wan text cross-attention (the text input is always the empty prompt).
3. New trajectory-to-video residual cross-attention.
4. Wan MLP.
5. PhysCtrl spatial attention, MLP, and temporal attention.
6. New video-to-trajectory residual cross-attention on PhysCtrl point tokens
   only. PhysCtrl velocity/control tokens are not updated by the bridge.

Each bridge projects Wan's 3,072-wide tokens and PhysCtrl's 256-wide tokens
into a shared 512-wide, eight-head attention space, then projects them back.
The two residual outputs have independently trainable, zero-initialized gates.
At initialization, the bridge has no effect on either backbone.

For a clip with `K` valid objects, concatenate the valid trajectory token
sequences for the video-to-trajectory direction. Wan tokens attend to all
valid objects. Each object's trajectory tokens attend only to the one Wan
video stream; no direct object-to-object bridge attention is allowed. There is
no architectural cap on `K`; the initial data curriculum only admits clips
with `K <= 4`. The first trainer requires batch size one, so no padded object
slots are introduced on the attention path.

## Paired data contract

Each sample has this structure:

```text
sample_0001/
  rgba_00000.png ... rgba_00048.png
  metadata.json
  objects/
    000/pc.hdf5
    001/pc.hdf5
```

Object directory names sorted lexically define the stable object slot order.
Every object HDF5 uses the existing `pc.hdf5` keys:
`point_cloud`, `initial_linear_velocity`, and `initial_angular_velocity`.
`point_cloud` has shape `(49, 1, 2048, 3)`; point index is a fixed physical
identity over all frames. All point clouds use one fixed frame-zero camera
reference coordinate system. Every trajectory frame aligns one-to-one with the
rendered video frame. Each inference object must provide its initial point
cloud and initial linear and angular velocity.

The prompt is always the empty string. The dataset stores no caption and
classifier-free text dropout is disabled.

## Training noising and loss

Wan retains its existing shifted flow-matching batch construction and masked
velocity MSE. Its protected first latent-time slice stays clean and has a zero
timestep.

The trajectory branch uses DDPM x0 prediction only. For each video sample,
sample one discrete PC DDPM timestep. Every valid object in that video uses
that same timestep, but each object receives independently drawn Gaussian
noise. The video flow time is sampled independently from the PC DDPM timestep;
the two branches retain their own time embeddings and native schedules.

Keep an individual trajectory loss for every valid object:

```text
L_pc[b, j] = mean((predicted_x0[b, j] - target_x0[b, j])^2)
L_total = L_video + sum(L_pc[b, j] for valid objects j)
```

Do not average trajectory losses over objects or include invalid/padded object
slots in the loss. Log the video loss, every object loss, trajectory-loss sum,
and bridge gradient norms.

Use one AdamW optimizer over all trainable parameters with the existing Wan
I2V settings: learning rate `1e-5`, betas `(0.9, 0.95)`, epsilon `1e-8`, and
weight decay `0.1`.

## Joint sampling and validation

Use 50 synchronized outer denoising steps by default. At every step, pass the
current Wan flow solver timestep and current PC DDIM timestep to the coupled
model. Use Wan's flow prediction for its Flow-UniPC update and each object's
DDPM x0 prediction for its DDIM update. Keep the I2V first video latent slice
and every initial object point cloud as clean conditions throughout sampling.

Run validation every 250 optimizer steps. For each selected validation sample,
write a decoded `video.mp4` and one
`object_<slot>_trajectory_comparison.mp4` per object. Trajectory videos must
reuse `training.pc_visualization.save_pointcloud_comparison_mp4`, the renderer
used by `train_pc.py`.

## Acceptance criteria

- A batch with any `K >= 1` loads into one video and `K` object trajectories;
  the initial curriculum rejects `K > 4`.
- The bridge has no effect at initialization and excludes padded slots and
  object-to-object bridge attention.
- One forward/backward update trains the Wan DiT, PhysCtrl model, and bridge;
  VAE/text parameters receive no gradients.
- PC DDPM timestep is identical across valid objects in one video, while their
  noise tensors are independent.
- The loss is one Wan velocity MSE plus the sum of individual object x0 MSEs.
- The sampler executes 50 coupled Wan Flow-UniPC and PC DDIM updates and
  produces the specified video and existing-renderer trajectory artifacts.
