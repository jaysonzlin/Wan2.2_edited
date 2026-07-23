# H200 Slurm submission headers

## Purpose

Run the existing I2V and TI2V jobs through the dedicated H200 partition rather
than the requeue partition.

## Design

Modify only the Slurm headers in these scripts:

- `submit_h200_i2v_requeue.sh`
- `submit_h200_i2v_lingbot_optim_requeue.sh`
- `submit_h200_ti2v_negative_requeue.sh`

In each file, replace `#SBATCH --partition=gpu_requeue` with
`#SBATCH --partition=gpu_h200`, and remove the `#SBATCH --constraint=h200` and
`#SBATCH --requeue` directives.  Filenames, job names, output paths, commands,
and training overrides remain unchanged.

## Verification

Run `bash -n` on all three scripts.  Confirm their headers contain the H200
partition and no constraint or requeue directive.
