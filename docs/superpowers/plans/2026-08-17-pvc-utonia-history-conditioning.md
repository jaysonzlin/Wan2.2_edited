# PVC Utonia History Conditioning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a dedicated history-DDPM training path that conditions each trajectory frame's spatial attention on padded, Utonia-encoded RGB-D point views.

**Architecture:** Keep legacy PC training unchanged. Add a PVC dataset/cache pair, an optional masked view-stream operation in the PhysCtrl spatial block, a history-only PVC model that uses it, and a dedicated trainer/pipeline/config that forwards the complete 49-frame view condition at training and sampling time.

**Tech Stack:** Python, PyTorch, HDF5/h5py, Accelerate, Diffusers, PyYAML, pytest.

## Global Constraints

- Do not modify `train_pc.py` behavior, `PCTrajectoryModel` behavior, or existing PC checkpoint layouts.
- PVC accepts only 49 frames, 2,048 trajectory points, four history frames, 45 DDPM futures, and frozen Utonia.
- Each sample has `point_views/0000.h5` through `point_views/0048.h5`; a view has `0 <= N <= 2048` XYZ/RGB rows.
- Point-view Utonia cache records use `[49, 2048, D]` features and `[49, 2048]` Boolean masks.
- Encode only real point-view rows with Utonia; pad features and XYZ afterward.
- View tokens use shared trajectory XYZ/Utonia encoders, a learned shared type vector, and temporal-only position features; they do not use fixed 0--2047 slot features.
- Real view and trajectory tokens attend bidirectionally in each spatial block; padded view tokens are excluded as keys/values.
- View token states persist across spatial blocks only. They do not enter temporal attention or the output head.
- PVC sampling requires all 49 aligned point views. `depth.h5` is not a PVC input.
- Preserve unrelated dirty working-tree changes; stage only files named in each task.

---

## File structure

| File | Responsibility |
| --- | --- |
| `training/pvc_config.py` | Load PC YAML then enforce the PVC-only contract. |
| `training/pvc_dataset.py` | Validate/read 49 variable-length view HDF5 files and return padded view tensors. |
| `training/pvc_utonia_features.py` | Atomically cache/reload per-sample, per-frame padded Utonia features. |
| `wan/modules/pc_physctrl.py` | Add a backward-compatible masked attention input and an explicit spatial-only view-stream block method. |
| `wan/modules/pvc_trajectory.py` | Encode the PVC view stream, run joined spatial blocks, and decode only trajectory futures. |
| `wan/pvc_pipeline.py` | Sample a PVC history DDIM trajectory while forwarding full view conditions. |
| `train_pvc.py` | Dedicated PVC entry point and training/visualization helpers. |
| `configs/train/config_pvc_utonia_history_overfit.yaml` | Isolated PVC overfit experiment configuration. |

### Task 1: PVC configuration and point-view dataset

**Files:**

- Create: `training/pvc_config.py`
- Create: `training/pvc_dataset.py`
- Create: `tests/test_pvc_config.py`
- Create: `tests/test_pvc_dataset.py`

**Interfaces:**

- `load_pvc_config(path: str | Path, overrides: list[str]) -> dict` calls `load_pc_config` then `validate_pvc_config`.
- `validate_pvc_config(config: dict) -> None` requires `model.conditioning == "history"`, `history_frames == 4`, `model.utonia_enabled is True`, `objective.type == "ddpm"`, a non-empty `data.object_id`, `data.utonia_cache_root`, and `data.point_view_utonia_cache_root`.
- `PVCTrajectoryDataset(dataset_root, *, object_id, utonia_cache_root=None, point_view_utonia_cache_root=None)` returns the legacy history keys plus `point_views [49, 2048, 3]` and `point_view_mask [49, 2048]`; with cache roots it also returns both Utonia feature tensors.
- `point_view_source_paths: dict[str, tuple[Path, ...]]` maps every dataset sample id to its ordered 49 source HDF5 paths for cache preparation.

- [ ] **Step 1: Write failing config and dataset tests.**

```python
def test_pvc_config_rejects_velocity_flow_or_missing_view_cache(tmp_path):
    path = tmp_path / "pvc.yaml"
    path.write_text(valid_pvc_config().replace("conditioning: history", "conditioning: velocity"))
    with pytest.raises(ValueError, match="PVC requires model.conditioning 'history'"):
        load_pvc_config(path, [])

def test_dataset_pads_each_view_and_preserves_validity(tmp_path):
    write_pvc_sample(tmp_path / "sample_0", counts=[2, 0] + [1] * 47)
    sample = PVCTrajectoryDataset(tmp_path, object_id="000")[0]
    assert sample["point_views"].shape == (49, 2048, 3)
    assert sample["point_view_mask"].sum(dim=1).tolist()[:2] == [2, 0]
```

Also test missing frame `0007.h5`, missing `xyz`/`rgb`, non-finite XYZ, invalid RGB, and 2,049 rows raise a path-specific `ValueError`; test `depth.h5` is not required.

- [ ] **Step 2: Run the focused tests and verify they fail because PVC modules do not exist.**

Run: `pytest tests/test_pvc_config.py tests/test_pvc_dataset.py -q`

Expected: collection/import failure for `training.pvc_config` and `training.pvc_dataset`.

- [ ] **Step 3: Implement the fixed config validator and the dataset.**

```python
class PVCTrajectoryDataset(PCTrajectoryDataset):
    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        sample = super().__getitem__(index)
        views, mask = self._read_point_views(self.point_view_source_paths[sample["sample_id"]])
        sample["point_views"] = views
        sample["point_view_mask"] = mask
        return sample

    @staticmethod
    def _read_point_views(paths: tuple[Path, ...]) -> tuple[torch.Tensor, torch.Tensor]:
        views = torch.zeros(49, 2048, 3, dtype=torch.float32)
        mask = torch.zeros(49, 2048, dtype=torch.bool)
        # validate xyz/rgb and copy only real rows; do not read depth.h5
        return views, mask
```

Build `point_view_source_paths` from the enclosing `sample_*` directory, not the selected object directory. Require the filenames exactly once and in numeric order. Reuse `validate_utonia_rgb` for RGB validation with the view's real count.

- [ ] **Step 4: Run focused tests until green.**

Run: `pytest tests/test_pvc_config.py tests/test_pvc_dataset.py -q`

Expected: PASS.

- [ ] **Step 5: Commit the bounded data-contract change.**

```bash
git add training/pvc_config.py training/pvc_dataset.py tests/test_pvc_config.py tests/test_pvc_dataset.py
git commit -m "feat: add PVC dataset contract"
```

### Task 2: Per-sample point-view Utonia cache

**Files:**

- Create: `training/pvc_utonia_features.py`
- Modify: `training/pvc_dataset.py`
- Create: `tests/test_pvc_utonia_features.py`
- Modify: `tests/test_pvc_dataset.py`

**Interfaces:**

- `prepare_point_view_utonia_feature_cache(sources, cache_root, extractor, *, feature_dim: int) -> None` writes/reuses one cache record per sample id.
- `load_cached_point_view_utonia_features(cache_root, sample_id, *, feature_dim: int) -> tuple[torch.Tensor, torch.Tensor]` returns finite float32 `[49,2048,D]` features and Boolean `[49,2048]` mask.
- A cache record has `metadata`, `features`, and `mask`; metadata includes a fingerprint of all ordered source HDF5 bytes and extractor checkpoint/preprocess identities.
- When both cache roots are configured, `PVCTrajectoryDataset.__getitem__` returns `point_view_utonia_features` and verifies its cached mask equals the freshly read source mask.

- [ ] **Step 1: Write failing cache tests.**

```python
def test_point_view_cache_encodes_real_rows_then_pads_and_reuses(tmp_path):
    sources = {"sample_0/objects/000": write_view_sources(tmp_path, [2] + [0] * 48)}
    extractor = RecordingExtractor(width=5)
    prepare_point_view_utonia_feature_cache(sources, tmp_path / "cache", extractor, feature_dim=5)
    features, mask = load_cached_point_view_utonia_features(tmp_path / "cache", "sample_0/objects/000", feature_dim=5)
    assert features.shape == (49, 2048, 5)
    assert mask.sum().item() == 2
    assert extractor.row_counts == [2]
```

Add tests that unchanged sources invoke no extractor calls on a second prepare, a changed `0012.h5` rebuilds the whole record, a width mismatch fails, and an all-empty sample returns zero features using the supplied `feature_dim` without calling the extractor.

- [ ] **Step 2: Run tests and verify the new cache API is missing.**

Run: `pytest tests/test_pvc_utonia_features.py tests/test_pvc_dataset.py -q`

Expected: import failure for `training.pvc_utonia_features`.

- [ ] **Step 3: Implement cache validation, fingerprinting, and atomic replacement.**

```python
def prepare_point_view_utonia_feature_cache(sources, cache_root, extractor, *, feature_dim):
    for sample_id, paths in sorted(sources.items()):
        fingerprint = _view_sources_fingerprint(paths)
        cache_path = _cache_path(cache_root, sample_id)
        record = _matching_record(
            cache_path, sample_id=sample_id, source_fingerprint=fingerprint,
            extractor=extractor, feature_dim=feature_dim,
        )
        if record is not None:
            continue
        features, mask = _encode_real_views(paths, extractor, feature_dim)
        _write_record(cache_path, _metadata(sample_id, fingerprint, extractor, feature_dim), features, mask)
```

Use `temporary = tempfile.NamedTemporaryFile(dir=path.parent, delete=False)`, then `Path(temporary.name).replace(path)`, like the existing Utonia cache. `_encode_real_views` must skip Utonia for a zero-row view, write zero padded rows, and reject an extractor result whose shape is not `(real_count, feature_dim)`.

- [ ] **Step 4: Run focused cache/dataset tests until green.**

Run: `pytest tests/test_pvc_utonia_features.py tests/test_pvc_dataset.py -q`

Expected: PASS.

- [ ] **Step 5: Commit cache support.**

```bash
git add training/pvc_utonia_features.py training/pvc_dataset.py tests/test_pvc_utonia_features.py tests/test_pvc_dataset.py
git commit -m "feat: cache Utonia point-view features"
```

### Task 3: Masked joined spatial attention

**Files:**

- Modify: `wan/modules/pc_physctrl.py`
- Modify: `tests/test_pc_physctrl_components.py`

**Interfaces:**

- `PhysCtrlAttention.forward(tokens, key_mask: torch.Tensor | None = None) -> torch.Tensor` accepts a Boolean `[B,L]` key/value validity mask without changing its no-mask behavior.
- `PhysCtrlSpatialTemporalBlock.forward_with_point_views(points, point_views, point_view_mask, temb) -> tuple[torch.Tensor, torch.Tensor]` runs joined spatial AdaLN/attention/MLP, keeps view states spatial-only, and applies temporal attention only to `points`.

- [ ] **Step 1: Write failing component tests.**

```python
def test_attention_excludes_invalid_key_values():
    attention = PhysCtrlAttention(dim=4, heads=2)
    valid = torch.tensor([[[1., 2., 3., 4.], [5., 6., 7., 8.]]])
    baseline = attention(valid)
    joined = torch.cat((valid, torch.full((1, 1, 4), 1e6)), dim=1)
    torch.testing.assert_close(
        attention(joined, key_mask=torch.tensor([[True, True, False]]))[:, :2],
        baseline,
    )

def test_view_states_skip_temporal_attention():
    seen_shapes = []
    hook = block.temporal_attention.register_forward_pre_hook(
        lambda _module, inputs: seen_shapes.append(inputs[0].shape)
    )
    try:
        output_points, output_views = block.forward_with_point_views(points, views, mask, temb)
    finally:
        hook.remove()
    assert seen_shapes == [torch.Size((points.shape[0] * points.shape[2], points.shape[1], points.shape[3]))]
    assert output_points.shape == points.shape
    assert output_views.shape == views.shape
```

Also assert joined spatial attention receives `2048 + 2048` tokens in the PVC path and changing an invalid padded view value cannot change trajectory output.

- [ ] **Step 2: Run focused component tests and observe missing keyword/method failures.**

Run: `pytest tests/test_pc_physctrl_components.py -q`

Expected: FAIL because `key_mask` and `forward_with_point_views` are absent.

- [ ] **Step 3: Implement the backward-compatible mask and explicit view method.**

```python
def forward(self, tokens, key_mask=None):
    # validate bool [B, length]; True means the key/value is visible
    attention_mask = None if key_mask is None else key_mask[:, None, None, :]
    attended = F.scaled_dot_product_attention(self.q_norm(q), self.k_norm(k), v, attn_mask=attention_mask)
    return self.to_out(attended.transpose(1, 2).reshape(batch, length, -1))

def forward_with_point_views(self, points, point_views, point_view_mask, temb):
    joined = torch.cat((flat_points, flat_views), dim=1)
    flat_view_mask = point_view_mask.reshape(batch * frames, view_count)
    trajectory_mask = torch.ones(batch * frames, count, device=points.device, dtype=torch.bool)
    key_mask = torch.cat((trajectory_mask, flat_view_mask), dim=1)
    mod_joined, _, gate, _ = self.norm1(joined, None, flat_temb)
    joined = joined + gate * self.spatial_attention(mod_joined, key_mask=key_mask)
    mod_joined, _, gate, _ = self.norm2(joined, None, flat_temb)
    joined = joined + gate * self.mlp(mod_joined)
    trajectory_points, next_view_points = joined.split((count, view_count), dim=1)
    next_view_points = next_view_points.masked_fill(~flat_view_mask[..., None], 0)
    # use the unchanged temporal branch below on trajectory_points only
    return trajectory_points, next_view_points
```

After each spatial sublayer, zero invalid view rows with `masked_fill` before carrying the view stream to the next block. Leave `forward(points, controls, temb)` byte-for-byte behaviorally equivalent for all existing callers.

- [ ] **Step 4: Run component and legacy model tests until green.**

Run: `pytest tests/test_pc_physctrl_components.py tests/test_pc_trajectory_model.py -q`

Expected: PASS.

- [ ] **Step 5: Commit spatial conditioning support.**

```bash
git add wan/modules/pc_physctrl.py tests/test_pc_physctrl_components.py
git commit -m "feat: add masked point-view spatial attention"
```

### Task 4: PVC history trajectory model

**Files:**

- Create: `wan/modules/pvc_trajectory.py`
- Create: `tests/test_pvc_trajectory_model.py`

**Interfaces:**

- `PVCTrajectoryModel(n_points: int = 2048, n_future_frames: int = 45, latent_dim: int = 256, n_layers: int = 8, num_heads: int = 4, utonia_feature_dim: int | None = None)` is history/DDPM-only.
- `forward(noisy_future_state, frame_times, points_history, point_views, point_view_mask, utonia_features, point_view_utonia_features) -> torch.Tensor` returns `[B,45,1,2048,3]`.
- `encode_states(noisy_future_state, frame_times, points_history, point_views, point_view_mask, utonia_features, point_view_utonia_features) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]` uses the same `PointEmbed`, `utonia_feature_norm`, and `utonia_feature_projection` for both streams.

- [ ] **Step 1: Write failing model tests.**

```python
def test_pvc_model_uses_shared_encoders_and_temporal_only_view_position():
    model = make_tiny_pvc_model(feature_dim=5)
    input_batches, fusion_batches = [], []
    input_hook = model.input_encoder.register_forward_pre_hook(
        lambda _module, inputs: input_batches.append(inputs[0].shape)
    )
    fusion_hook = model.utonia_feature_projection.register_forward_pre_hook(
        lambda _module, inputs: fusion_batches.append(inputs[0].shape)
    )
    output = model(noisy, times, history, views, mask, trajectory_features, view_features)
    input_hook.remove(); fusion_hook.remove()
    assert output.shape == (2, 45, 1, 8, 3)
    assert input_batches == [torch.Size((98, 8, 3)), torch.Size((98, 8, 3))]
    assert fusion_batches == [torch.Size((2, 49, 8, 69)), torch.Size((2, 49, 8, 69))]
```

Add rejection tests for wrong view shape, wrong Boolean mask shape/dtype, wrong Utonia widths, nonzero history times, and wrong 4/45 layout. Add a hook test showing the output head receives only 45 trajectory states.

- [ ] **Step 2: Run model tests and verify collection fails.**

Run: `pytest tests/test_pvc_trajectory_model.py -q`

Expected: import failure for `wan.modules.pvc_trajectory`.

- [ ] **Step 3: Implement the isolated PVC model.**

```python
view_positions = torch.cat((temporal_positions, torch.zeros_like(spatial_positions)), dim=-1)
view_states = self.input_encoder(point_views.reshape(-1, self.n_points, 3)).reshape(batch, 49, self.n_points, self.latent_dim)
view_states = self.utonia_feature_projection(torch.cat((view_states, normalized_view_features), dim=-1))
view_states = view_states + view_positions + self.point_view_type_embedding
for block in self.blocks:
    trajectory_states, view_states = block.forward_with_point_views(
        trajectory_states, view_states, point_view_mask, temb
    )
return self.decode_states(trajectory_states, temb, points_history)
```

Implement `PVCTrajectoryModel` as a history-specialized subclass or a focused sibling of `PCTrajectoryModel`, reusing its public building blocks but not changing its constructor or forward signature. Validate all fixed dimensions before reshaping. Add the temporal-only position tensor from the existing PhysCtrl table's first `latent_dim // 4` channels and zero its slot channels.

- [ ] **Step 4: Run model and legacy regression tests until green.**

Run: `pytest tests/test_pvc_trajectory_model.py tests/test_pc_trajectory_model.py tests/test_joint_bridge.py -q`

Expected: PASS.

- [ ] **Step 5: Commit the model.**

```bash
git add wan/modules/pvc_trajectory.py tests/test_pvc_trajectory_model.py
git commit -m "feat: add PVC history trajectory model"
```

### Task 5: PVC sampling pipeline and dedicated trainer

**Files:**

- Create: `wan/pvc_pipeline.py`
- Create: `train_pvc.py`
- Create: `configs/train/config_pvc_utonia_history_overfit.yaml`
- Create: `tests/test_pvc_pipeline.py`
- Create: `tests/test_train_pvc.py`

**Interfaces:**

- `PVCHistoryDDIMPipeline(model, scheduler).__call__(points_history, point_views, point_view_mask, utonia_features, point_view_utonia_features, device, num_inference_steps, generator=None) -> torch.Tensor` samples 45 future positions.
- `build_pvc_training_dataset(config, *, dataset_factory, extractor_factory, trajectory_cache_preparer, point_view_cache_preparer) -> tuple[PVCTrajectoryDataset, int]` prepares the existing trajectory cache first, then point-view cache using its returned feature width.
- `compute_pvc_training_prediction(batch, model, noise_scheduler, generator, device) -> tuple[torch.Tensor, torch.Tensor]` builds the 4/45 DDPM batch and forwards both Utonia conditions.
- `sample_pvc_visualization(pipeline, batch, device, num_inference_steps, generator) -> tuple[torch.Tensor, torch.Tensor]` prepends the four known trajectory frames and forwards all 49 views/masks to the pipeline.

- [ ] **Step 1: Write failing pipeline and trainer helper tests.**

```python
def test_pvc_pipeline_forwards_all_view_conditions_at_every_step():
    output = pipeline(history, views, mask, traj_features, view_features, "cpu", 2)
    assert output.shape == (1, 45, 1, 2, 3)
    assert len(model.calls) == 2
    assert all(call.point_views is views and call.mask is mask for call in model.calls)

def test_pvc_dataset_builder_prepares_trajectory_then_view_cache():
    dataset, width = build_pvc_training_dataset(
        config, dataset_factory=FakeDataset, extractor_factory=FakeExtractor,
        trajectory_cache_preparer=prepare_trajectory, point_view_cache_preparer=prepare_views,
    )
    assert calls == [("trajectory", dataset.source_paths), ("views", dataset.point_view_source_paths, width)]
```

Test that `sample_pvc_visualization` returns 49 frames, `train_pvc.main` imports the PVC-only modules, and the new YAML has distinct outputs/tracker/cache root plus the original fixed history fields.

- [ ] **Step 2: Run focused tests and verify missing PVC pipeline/trainer failures.**

Run: `pytest tests/test_pvc_pipeline.py tests/test_train_pvc.py -q`

Expected: collection/import failures for `wan.pvc_pipeline` and `train_pvc`.

- [ ] **Step 3: Implement the pipeline, copy-and-specialize trainer, and config.**

```python
for timestep in self.scheduler.timesteps:
    prediction = self.model(
        sample, frame_times, points_history, point_views, point_view_mask,
        utonia_features, point_view_utonia_features,
    )
    sample = self.scheduler.step(prediction, timestep, sample, generator=generator).prev_sample
```

Copy `train_pc.py` into `train_pvc.py`, retain its checkpoint/progress/Accelerate/visualization mechanics, and remove velocity/flow condition branches. Use `load_pvc_config`, `PVCTrajectoryDataset`, `PVCTrajectoryModel`, and `PVCHistoryDDIMPipeline`; never import or alter legacy selection helpers. In the YAML, copy all operational settings from `config_pc_utonia_history_overfit.yaml`, change only experiment identity/cache-root fields, and retain `resume_from_checkpoint: null`.

- [ ] **Step 4: Run focused integration tests until green.**

Run: `pytest tests/test_pvc_pipeline.py tests/test_train_pvc.py tests/test_pvc_config.py tests/test_pvc_dataset.py -q`

Expected: PASS.

- [ ] **Step 5: Commit the usable dedicated entry point.**

```bash
git add wan/pvc_pipeline.py train_pvc.py configs/train/config_pvc_utonia_history_overfit.yaml tests/test_pvc_pipeline.py tests/test_train_pvc.py
git commit -m "feat: add PVC history training entry point"
```

### Task 6: End-to-end verification and documentation check

**Files:**

- Modify only if verification exposes a defect: files from Tasks 1--5 and their matching tests.

**Interfaces:**

- `python -m compileall training/pvc_config.py training/pvc_dataset.py training/pvc_utonia_features.py wan/modules/pc_physctrl.py wan/modules/pvc_trajectory.py wan/pvc_pipeline.py train_pvc.py` succeeds.
- The full test suite passes without modifying legacy tests solely to accommodate PVC.

- [ ] **Step 1: Run all PVC-focused tests together.**

Run: `pytest tests/test_pvc_config.py tests/test_pvc_dataset.py tests/test_pvc_utonia_features.py tests/test_pc_physctrl_components.py tests/test_pvc_trajectory_model.py tests/test_pvc_pipeline.py tests/test_train_pvc.py -q`

Expected: PASS.

- [ ] **Step 2: Compile all modified Python modules.**

Run: `python -m compileall training/pvc_config.py training/pvc_dataset.py training/pvc_utonia_features.py wan/modules/pc_physctrl.py wan/modules/pvc_trajectory.py wan/pvc_pipeline.py train_pvc.py`

Expected: all listed modules compile with exit code 0.

- [ ] **Step 3: Run the complete test suite.**

Run: `pytest -q`

Expected: PASS.

- [ ] **Step 4: Inspect the staged feature for whitespace errors and accidental scope expansion.**

Run: `git diff --check HEAD && git status --short`

Expected: no `git diff --check` output; only PVC feature files and pre-existing user changes appear.

- [ ] **Step 5: Perform the manual GPU integration check.**

Run from the remote project directory: `accelerate launch --config_file configs/accelerate/h200_single_gpu.yaml train_pvc.py --config configs/train/config_pvc_utonia_history_overfit.yaml max_train_steps=1 num_train_epochs=1 dataloader_num_workers=0`

Expected: both Utonia caches are created/reused, one PVC forward/backward/optimizer step completes, and `outputs/pvc_trajectory_utonia_history_overfit` contains its resolved `config.yaml`.

- [ ] **Step 6: Commit verified integration fixes when the previous commands changed feature files.**

```bash
git add training/pvc_config.py training/pvc_dataset.py training/pvc_utonia_features.py wan/modules/pc_physctrl.py wan/modules/pvc_trajectory.py wan/pvc_pipeline.py train_pvc.py configs/train/config_pvc_utonia_history_overfit.yaml tests/test_pvc_config.py tests/test_pvc_dataset.py tests/test_pvc_utonia_features.py tests/test_pc_physctrl_components.py tests/test_pvc_trajectory_model.py tests/test_pvc_pipeline.py tests/test_train_pvc.py
git commit -m "fix: verify PVC training integration"
```
