# PC Terminology Migration Design

## Goal

Rename PC trajectory components and user-facing artifacts so names describe
their shared DDPM/flow role rather than incorrectly implying every path is
flow matching.

## Active names

- `wan/modules/pc_flow.py` becomes `wan/modules/pc_trajectory.py`.
- `PCFlowModel` becomes `PCTrajectoryModel`.
- `PCFlowHead` becomes `PCOutputHead`.
- `training/pc_flow.py` becomes `training/pc_objectives.py`; it retains only
  flow-specific `make_pc_flow_batch` and `flow_mse` alongside shared MSE.
- `tests/test_pc_flow_model.py` becomes `tests/test_pc_trajectory_model.py`.
- `tests/test_pc_flow.py` becomes `tests/test_pc_objectives.py`.

`PCFlowPipeline`, `make_pc_flow_batch`, and UniPC remain flow-specific names.
`PCDDIMPipeline` remains the DDPM-specific name.

## User-facing names

The shipped configuration uses `outputs/pc_trajectory_8layers` and W&B project
`pc_trajectory`. README/test documentation and active design documents use the
same neutral terminology. Historical committed documents are not rewritten.

## Migration contract

All active imports, lazy exports, tests, and trainer construction use the new
names. No compatibility alias is retained for old shared component names, so
future callers cannot accidentally adopt misleading flow-only terminology.

## Validation

Run `rg` to confirm old shared names are absent outside historical documents
and flow-specific identifiers, then run the full pytest suite.
