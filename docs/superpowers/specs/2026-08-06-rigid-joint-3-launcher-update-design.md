# Rigid Joint 3 Launcher Update Design

## Goal

Update the three-object rigid joint-training launcher to run the enabled rigid objective at weight `0.001` with four CPU cores and isolated output naming.

## Design

Modify only `submit_joint_3_rigid.sh`.

Use `joint_3_rigid_0.001` for the SLURM job name and log/error filename prefixes. Set `#SBATCH --cpus-per-task=4`. Route checkpoints and tracker output to `outputs/joint_3_rigid_0.001`.

Pass these rigid-objective overrides:

```text
objective.enable_rigid_loss=true
objective.rigid_loss_weight=0.001
objective.rigid_loss_neighbors=4
```

Retain the existing H200, container, Accelerate, dataset (`td_832x480_3`), resume, and max-step settings.

## Verification

Add a focused submission-script regression test covering all modified values. Run it and the full pytest suite.

