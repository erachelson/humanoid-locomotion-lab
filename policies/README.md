# Trained policies

One folder per kept training run: `<task>/<run name>/`, using the layout of
unitree_rl_lab's `logs/rsl_rl/<task>/<run>/` with only the files worth keeping:

- `exported/policy.onnx`, `exported/policy.pt`: the actor only, for inference
  (input `obs` [N, 480], output `actions` [N, 29]).
- `params/deploy.yaml`: what the deploy controller needs (joint order, gains, action scale and offset,
  observation terms). `params/env.yaml`, `agent.yaml`, `velocity_env_cfg.py`: training config snapshot.
- `model_<it>.pt`: the last full rsl_rl checkpoint (actor, critic, optimizer), to resume training.

TensorBoard logs and intermediate checkpoints stay in `~/git_repos/unitree_rl_lab/logs/` (not versioned).

## `g1_29dof_velocity/2026-11-09_00-14-03` (current)

Task `Unitree-G1-29dof-Velocity` (unitree_rl_lab 0.2.1, Isaac Lab 2.3.0, Isaac Sim 5.1), 2048 envs, seed 42.
5000 iterations (run `2026-11-07_18-18-18`) resumed for 15000 more: last checkpoint `model_19998.pt`.
Walks in Isaac Lab and in MuJoCo via Unitree's `g1_ctrl`; does not turn (the yaw-command curriculum is
not enabled in the task). Details and evaluation in [doc/plan.md](../doc/plan.md), Phase 1.

`scripts/eval_velocity_tracking.py` (64 envs, 8 s per command, no pushes):

| Command (vx, vy, wz) | Measured (vx, vy, wz) | Fell |
|---|---|---|
| 0, 0, 0 | +0.00, +0.00, +0.00 | 0% |
| +0.40, 0, 0 | +0.39, +0.01, +0.03 | 0% |
| +0.80, 0, 0 | +0.79, +0.02, +0.01 | 0% |
| −0.30, 0, 0 | −0.22, −0.02, −0.01 | 0% |
| 0, +0.30, 0 | +0.00, +0.22, −0.01 | 0% |
| 0, 0, +0.20 | −0.00, −0.00, −0.00 | 0% |

Use it:
```bash
# Isaac Lab evaluation (env_isaaclab)
python scripts/eval_velocity_tracking.py --policy policies/g1_29dof_velocity/2026-11-09_00-14-03/exported/policy.pt

# MuJoCo sim2sim: in unitree_rl_lab/deploy/robots/g1_29dof/config/config.yaml (Velocity state);
# relative paths resolve from deploy/robots/g1_29dof/, ~ is not expanded
policy_dir: ../../../../humanoid-locomotion-lab/policies/g1_29dof_velocity/2026-11-09_00-14-03
```
