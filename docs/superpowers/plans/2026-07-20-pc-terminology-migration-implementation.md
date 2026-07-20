# PC Terminology Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rename shared PC trajectory components and user-facing artifacts to objective-neutral terminology.

**Architecture:** Flow names remain only on flow-specific batches and UniPC. Shared model, head, module, tests, configuration artifacts, imports, exports, and active documentation use PC trajectory terminology.

**Tech Stack:** Python 3.10, PyTorch, pytest, git.

## Global Constraints

- No compatibility aliases for old shared names.
- Preserve `PCFlowPipeline`, `make_pc_flow_batch`, and UniPC names.
- Use `pc_trajectory` for default output and W&B project.
- Do not rewrite historical design/plan documents.

---

### Task 1: Rename shared PC trajectory modules and artifacts

**Files:**
- Rename `wan/modules/pc_flow.py` to `wan/modules/pc_trajectory.py`.
- Rename `training/pc_flow.py` to `training/pc_objectives.py`.
- Rename relevant tests; modify imports, exports, trainer, config, README, and active docs.

**Interfaces:**
- `PCTrajectoryModel`, `PCOutputHead`, and `training.pc_objectives.mse_loss` replace shared flow names.
- Flow-specific `make_pc_flow_batch` and `PCFlowPipeline` remain unchanged.

- [ ] **Step 1: Write a failing import test**

```python
from wan.modules.pc_trajectory import PCTrajectoryModel


def test_trajectory_model_uses_objective_neutral_name():
    assert PCTrajectoryModel.__name__ == "PCTrajectoryModel"
```

- [ ] **Step 2: Verify red**

Run: `conda run -n das python -m pytest tests/test_pc_trajectory_model.py -q`

Expected: import failure before rename.

- [ ] **Step 3: Apply migration**

Use `git mv` for shared source/test files. Rename classes/imports, use `noisy_future_state` with mode-specific `flow_state`/`noisy_positions`, update lazy exports, and set config output/W&B names to `pc_trajectory`.

- [ ] **Step 4: Verify green and commit**

Run: `MPLCONFIGDIR=/private/tmp/mplconfig conda run -n das python -m pytest -q`

```bash
git add -A
git commit -m "refactor: rename PC trajectory components"
```
