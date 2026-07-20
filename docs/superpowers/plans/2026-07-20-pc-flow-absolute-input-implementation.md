# PC Flow Absolute Input Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace mixed PC-flow encoder inputs with absolute noisy positions while preserving the displacement-space flow contract.

**Architecture:** `PCFlowModel.forward` continues to accept a displacement-space flow state and returns a displacement-space flow vector. It translates the 48 future state frames by the initial cloud only at the PointEmbed boundary, producing `[p0, p0 + x_t]` for factorized processing; the trainer and pipeline remain unchanged.

**Tech Stack:** Python 3.10, PyTorch, pytest.

## Global Constraints

- Preserve the `PCFlowModel.forward(noisy_displacements, frame_times, init_pc, initial_linear_velocity, initial_angular_velocity)` signature.
- Preserve flow target construction, the displacement-space sampler state, the final post-integration `+ p0`, and both velocity controls.
- Preserve PointEmbed and the fixed 49-frame / 2048-point factorized architecture.
- Replace the mixed representation outright; do not add a configuration option or retain a runtime path for `[p0, x_t]`.

---

## File structure

- Modify `wan/modules/pc_flow.py`: translate future flow-state coordinates before PointEmbed.
- Modify `tests/test_pc_flow_model.py`: regression coverage for the coordinates sent to PointEmbed.

### Task 1: Embed every model frame in absolute position coordinates

**Files:**
- Modify: `wan/modules/pc_flow.py:127-128`
- Modify: `tests/test_pc_flow_model.py`

**Interfaces:**
- Consumes: `noisy_displacements` with shape `(B, 48, 1, N, 3)` and `init_pc` with shape `(B, 1, N, 3)`.
- Produces: unchanged model output shape `(B, 48, 1, N, 3)` in displacement-space flow units.
- Internal invariant: the `input_encoder` receives flattened frames equivalent to `[p0, p0 + noisy_displacements]`.

- [ ] **Step 1: Write the failing regression test**

Add this test to `tests/test_pc_flow_model.py`:

```python
def test_model_embeds_future_flow_states_as_absolute_positions():
    model = make_tiny_model()
    captured = {}

    def capture_coordinates(_module, inputs):
        captured["coordinates"] = inputs[0].detach().clone()

    handle = model.input_encoder.register_forward_pre_hook(capture_coordinates)
    initial = torch.full((1, 1, 8, 3), 10.0)
    flow_state = torch.full((1, 48, 1, 8, 3), 2.0)
    try:
        model(
            flow_state,
            torch.tensor([[0.0] + [500.0] * 48]),
            initial,
            torch.zeros(1, 1, 3),
            torch.zeros(1, 1, 3),
        )
    finally:
        handle.remove()

    embedded_frames = captured["coordinates"].reshape(1, 49, 8, 3)
    assert torch.equal(embedded_frames[:, :1], initial)
    assert torch.equal(embedded_frames[:, 1:], torch.full_like(flow_state.squeeze(2), 12.0))
```

- [ ] **Step 2: Run the regression test to verify it fails**

Run:

```bash
conda run -n das python -m pytest tests/test_pc_flow_model.py::test_model_embeds_future_flow_states_as_absolute_positions -q
```

Expected: failure because the existing encoder receives `2.0` future coordinates instead of `12.0`.

- [ ] **Step 3: Translate the future state at the PointEmbed boundary**

Replace the current coordinate construction in `wan/modules/pc_flow.py` with:

```python
future_positions = init_pc.unsqueeze(1) + noisy_displacements
coordinates = torch.cat((init_pc.unsqueeze(1), future_positions), dim=1).squeeze(2)
```

Do not change `noisy_displacements` itself, the flow head, `make_pc_flow_batch`, or `PCFlowPipeline`.

- [ ] **Step 4: Run the focused model tests**

Run:

```bash
conda run -n das python -m pytest tests/test_pc_flow_model.py -q
```

Expected: all model tests pass, including the existing zero-flow-head output contract.

- [ ] **Step 5: Run the full regression suite**

Run:

```bash
MPLCONFIGDIR=/private/tmp/mplconfig conda run -n das python -m pytest -q
```

Expected: all tests pass; only existing Matplotlib/Torch deprecation warnings may remain.

- [ ] **Step 6: Commit the replacement**

```bash
git add wan/modules/pc_flow.py tests/test_pc_flow_model.py
git commit -m "feat: embed PC flow states as positions"
```
