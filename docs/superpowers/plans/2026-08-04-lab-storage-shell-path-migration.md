# Lab Storage Shell Path Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the retired lab-storage prefix in all matching shell scripts with `/n/lab_storage`.

**Architecture:** The change is a mechanical, prefix-only migration across nine independent Slurm submission scripts. A small static test protects the migration’s two invariants: none of the scripts uses the retired location and all retain the expected new prefix.

**Tech Stack:** Bash, Python, pytest.

## Global Constraints

- Modify only `.sh` files that contain `/net/holy-isilon/ifs/rc_labs`.
- Replace the prefix exactly with `/n/lab_storage` while preserving every trailing path component.
- The migration affects 27 references in nine shell scripts.
- Validate all edited scripts with `bash -n`.

---

### Task 1: Protect and migrate shell-script storage paths

**Files:**
- Modify: `submit_joint.sh`
- Modify: `submit_h200_i2v_lingbot_optim_requeue.sh`
- Modify: `submit_h200_i2v_requeue.sh`
- Modify: `submit_h200_ti2v_negative_requeue.sh`
- Modify: `submit_h200.sh`
- Modify: `submit_h200_requeue.sh`
- Modify: `submit_832x480.sh`
- Modify: `submit_lingbot_nodecay.sh`
- Modify: `submit_lingbot.sh`
- Create: `tests/test_lab_storage_shell_paths.py`

**Interfaces:**
- Consumes: shell-script text containing the retired storage prefix.
- Produces: shell scripts whose project and log paths resolve under `/n/lab_storage`.

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path


SCRIPTS = (
    "submit_joint.sh",
    "submit_h200_i2v_lingbot_optim_requeue.sh",
    "submit_h200_i2v_requeue.sh",
    "submit_h200_ti2v_negative_requeue.sh",
    "submit_h200.sh",
    "submit_h200_requeue.sh",
    "submit_832x480.sh",
    "submit_lingbot_nodecay.sh",
    "submit_lingbot.sh",
)


def test_submission_scripts_use_the_current_lab_storage_prefix() -> None:
    scripts = [Path(script).read_text() for script in SCRIPTS]

    assert all("/net/holy-isilon/ifs/rc_labs" not in script for script in scripts)
    assert sum(script.count("/n/lab_storage") for script in scripts) == 27
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_lab_storage_shell_paths.py -v`

Expected: FAIL because the nine scripts still use the retired storage prefix.

- [ ] **Step 3: Make the mechanical prefix migration**

Apply this exact substitution to each listed file:

```text
/net/holy-isilon/ifs/rc_labs  ->  /n/lab_storage
```

Do not alter the scripts’ remaining content, including Slurm directives,
bind mount arguments, and path suffixes.

- [ ] **Step 4: Run focused verification**

Run:

```bash
pytest tests/test_lab_storage_shell_paths.py -v
! rg -F '/net/holy-isilon/ifs/rc_labs' --glob '*.sh' .
for script in submit_joint.sh submit_h200_i2v_lingbot_optim_requeue.sh submit_h200_i2v_requeue.sh submit_h200_ti2v_negative_requeue.sh submit_h200.sh submit_h200_requeue.sh submit_832x480.sh submit_lingbot_nodecay.sh submit_lingbot.sh; do bash -n "$script"; done
```

Expected: pytest reports one passing test, ripgrep finds no old shell-script
paths, and every `bash -n` invocation exits with status 0.

- [ ] **Step 5: Commit**

```bash
git add submit_joint.sh submit_h200_i2v_lingbot_optim_requeue.sh submit_h200_i2v_requeue.sh submit_h200_ti2v_negative_requeue.sh submit_h200.sh submit_h200_requeue.sh submit_832x480.sh submit_lingbot_nodecay.sh submit_lingbot.sh tests/test_lab_storage_shell_paths.py
git commit -m "chore: migrate shell scripts to lab storage"
```
