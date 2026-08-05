# Joint Wan--PhysCtrl Optimizer Groups Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make video, BCA, and PC AdamW settings independently configurable and log the PC pre-clip gradient norm during joint training.

**Architecture:** The optimizer factory receives the joint model and nested optimizer configuration, then creates three AdamW parameter groups owned by its three child modules. Configuration validation owns the nested schema, while small norm helpers keep each metric isolated to its corresponding module.

**Tech Stack:** Python, PyTorch AdamW, Accelerate, PyYAML, pytest.

## Global Constraints

- Use `optimizer.video`, `optimizer.bca`, and `optimizer.pc`, each with `lr`, `betas`, `eps`, and `weight_decay`.
- Video and BCA defaults are `1e-5`, `[0.9, 0.95]`, `1e-8`, and `0.1`.
- PC defaults are `1e-4`, `[0.9, 0.999]`, `1e-8`, and `1e-2`.
- Log the pre-clip PC gradient L2 norm as `train/pc_gradient_norm`.
- Do not migrate one-parameter-group optimizer checkpoints.

---

### Task 1: Nested optimizer configuration contract

**Files:**
- Modify: `configs/train/joint_wan_physctrl_832x480.yaml:26-30`
- Modify: `training/joint_config.py:33-45`
- Test: `tests/test_joint_dataset.py:43-75`

**Interfaces:**
- Consumes: `load_joint_config(path, overrides) -> dict`.
- Produces: validated `config["optimizer"][group]` mappings for `video`, `bca`, and `pc`.

- [ ] **Step 1: Write the failing test**

```python
def test_joint_config_accepts_separate_optimizer_groups(tmp_path):
    path = tmp_path / "joint.yaml"
    path.write_text(_valid_config())

    optimizer = load_joint_config(path, [])["optimizer"]

    assert optimizer["video"]["lr"] == 1.0e-5
    assert optimizer["bca"]["betas"] == [0.9, 0.95]
    assert optimizer["pc"]["weight_decay"] == 1.0e-2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_joint_dataset.py::test_joint_config_accepts_separate_optimizer_groups -v`

Expected: FAIL because `_valid_config()` still defines the flat optimizer schema.

- [ ] **Step 3: Write the minimal implementation**

```yaml
optimizer:
  video: {lr: 1.0e-5, betas: [0.9, 0.95], eps: 1.0e-8, weight_decay: 0.1}
  bca: {lr: 1.0e-5, betas: [0.9, 0.95], eps: 1.0e-8, weight_decay: 0.1}
  pc: {lr: 1.0e-4, betas: [0.9, 0.999], eps: 1.0e-8, weight_decay: 1.0e-2}
```

Replace the flat validator with a loop over those mappings and expected defaults, reporting errors as `optimizer.<group>.<key>`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_joint_dataset.py::test_joint_config_accepts_separate_optimizer_groups -v`

Expected: PASS.

- [ ] **Step 5: Commit**

Run: `git add configs/train/joint_wan_physctrl_832x480.yaml training/joint_config.py tests/test_joint_dataset.py && git commit -m "feat: configure joint optimizer groups"`

### Task 2: Grouped optimizer and PC gradient metric

**Files:**
- Modify: `train_joint_wan_physctrl.py:12-21,118-120,243,299-315`
- Test: `tests/test_train_joint_wan_physctrl.py:20-62`

**Interfaces:**
- Consumes: a `JointWanPhysCtrlModel` with `wan_model`, `bridges`, and `pc_model`; nested optimizer settings from Task 1.
- Produces: `create_joint_optimizer(model, optimizer_config) -> torch.optim.AdamW` with groups named `video`, `bca`, and `pc`; `pc_gradient_norm(model) -> torch.Tensor`.

- [ ] **Step 1: Write the failing tests**

```python
def test_joint_optimizer_uses_separate_configured_parameter_groups():
    model = _joint_model_with_three_linear_children()
    optimizer = create_joint_optimizer(model, _optimizer_config())

    assert [group["name"] for group in optimizer.param_groups] == ["video", "bca", "pc"]
    assert optimizer.param_groups[2]["lr"] == 1.0e-4
    assert optimizer.param_groups[2]["betas"] == (0.9, 0.999)


def test_pc_gradient_norm_uses_pc_gradients_only():
    pc_parameter = torch.nn.Parameter(torch.zeros(2))
    pc_parameter.grad = torch.tensor([3.0, 4.0])
    model = type("Joint", (), {"pc_model": _parameters_module(pc_parameter)})()

    assert pc_gradient_norm(model).item() == 5.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_train_joint_wan_physctrl.py -v`

Expected: FAIL because the factory accepts a flat parameter iterable and there is no PC norm helper.

- [ ] **Step 3: Write the minimal implementation**

```python
def pc_gradient_norm(model) -> torch.Tensor:
    norms = [parameter.grad.detach().norm() for parameter in model.pc_model.parameters() if parameter.grad is not None]
    return torch.stack(norms).norm() if norms else torch.zeros(())


optimizer = create_joint_optimizer(model, config["optimizer"])
```

Make `create_joint_optimizer` build one AdamW group per named child module using its matching nested settings. After `accelerator.backward(loss)`, calculate `pc_gradient_norm` before clipping and add `"train/pc_gradient_norm"` to the metrics mapping.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_train_joint_wan_physctrl.py -v`

Expected: PASS.

- [ ] **Step 5: Run focused integration verification**

Run: `pytest tests/test_joint_dataset.py tests/test_train_joint_wan_physctrl.py -v`

Expected: PASS with all focused configuration and trainer tests green.

- [ ] **Step 6: Commit**

Run: `git add train_joint_wan_physctrl.py tests/test_train_joint_wan_physctrl.py && git commit -m "feat: log PC gradient norm in joint training"`
