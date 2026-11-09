"""Evaluate a locomotion policy in Isaac Lab under fixed velocity commands.

For each command, all environments are reset, the command is held constant, and the script reports the
velocity the robots actually reach (body frame, averaged after a warm-up) and how many fell.
Pushes and curricula are disabled so that results only reflect the policy.

Usage (from a shell with env_isaaclab activated):
    python scripts/eval_velocity_tracking.py --policy <run>/exported/policy.pt
    python scripts/eval_velocity_tracking.py --policy <dir>/exported/policy.onnx --commands "0,0,0" "0.5,0,0"
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--policy", required=True, help="Exported policy: TorchScript .pt or ONNX .onnx (input: obs [N, 480]).")
parser.add_argument("--task", default="Unitree-G1-29dof-Velocity")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--duration", type=float, default=8.0, help="Seconds per command.")
parser.add_argument("--warmup", type=float, default=2.0, help="Seconds ignored before averaging velocities.")
parser.add_argument(
    "--commands",
    nargs="+",
    default=["0,0,0", "0.2,0,0", "0.4,0,0", "0.8,0,0", "-0.3,0,0", "0,0.3,0", "0,0,0.2"],
    help="Commands as 'vx,vy,wz' (m/s, m/s, rad/s).",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch

import unitree_rl_lab.tasks  # noqa: F401 — register Unitree RL Lab task environments
from isaaclab_tasks.utils import parse_env_cfg


class Policy:
    """Wraps a TorchScript or ONNX policy exported by rsl_rl (observation normalizer included)."""

    def __init__(self, path: str, device: str):
        self.device = device
        if path.endswith(".onnx"):
            import onnxruntime as ort

            self.session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
            self.input_name = self.session.get_inputs()[0].name
            self.jit = None
        else:
            self.jit = torch.jit.load(path, map_location=device).eval()

    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        if self.jit is not None:
            with torch.inference_mode():
                return self.jit(obs)
        # The exported ONNX models have a fixed batch size of 1, so run row by row.
        x = obs.detach().cpu().numpy().astype(np.float32)
        out = np.concatenate([self.session.run(None, {self.input_name: row[None]})[0] for row in x])
        return torch.from_numpy(out).to(self.device)


env_cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=args.num_envs)
# Hold commands fixed and remove everything that changes the test conditions.
env_cfg.commands.base_velocity.resampling_time_range = (1.0e6, 1.0e6)
env_cfg.commands.base_velocity.rel_standing_envs = 0.0
env_cfg.events.push_robot = None
env_cfg.curriculum.lin_vel_cmd_levels = None
env_cfg.curriculum.terrain_levels = None
env = gym.make(args.task, cfg=env_cfg)
base_env = env.unwrapped
robot = base_env.scene["robot"]
cmd_term = base_env.command_manager.get_term("base_velocity")
policy = Policy(args.policy, base_env.device)

# Locate the latest velocity command inside the policy observation, to check the command reaches the policy.
obs_manager = base_env.observation_manager
offset = 0
for name, dims in zip(obs_manager.active_terms["policy"], obs_manager.group_obs_term_dim["policy"]):
    if name == "velocity_commands":
        cmd_slice = slice(offset + dims[0] - 3, offset + dims[0])  # history is oldest-first: last 3 = newest
        break
    offset += dims[0]

steps = int(args.duration / base_env.step_dt)
warmup_steps = int(args.warmup / base_env.step_dt)
print(f"\nPolicy: {args.policy}", flush=True)
print(f"{args.num_envs} envs, {args.duration:.0f} s per command, velocities averaged after {args.warmup:.0f} s\n", flush=True)
print(
    f"{'command (vx, vy, wz)':>24} | {'in observation':>20} | {'measured (vx, vy, wz)':>24} | {'fell':>6} | {'base height':>11}",
    flush=True,
)
print("-" * 99, flush=True)

for spec in args.commands:
    command = torch.tensor([float(v) for v in spec.split(",")], device=base_env.device)
    obs, _ = env.reset()
    fell = torch.zeros(args.num_envs, dtype=torch.bool, device=base_env.device)
    vel_sum = torch.zeros(args.num_envs, 3, device=base_env.device)
    height_sum = torch.zeros(args.num_envs, device=base_env.device)
    for step in range(steps):
        cmd_term.vel_command_b[:] = command
        obs, _, terminated, _, _ = env.step(policy(obs["policy"]))
        fell |= terminated  # no time-outs within the test: any termination is a fall
        if step >= warmup_steps:
            vel_sum[:, :2] += robot.data.root_lin_vel_b[:, :2]
            vel_sum[:, 2] += robot.data.root_ang_vel_b[:, 2]
            height_sum += robot.data.root_pos_w[:, 2]
    standing = ~fell
    n = steps - warmup_steps
    if standing.any():
        vel = (vel_sum[standing] / n).mean(dim=0).tolist()
        height = (height_sum[standing] / n).mean().item()
        measured = f"{vel[0]:+.2f}, {vel[1]:+.2f}, {vel[2]:+.2f}"
    else:
        measured, height = "all fell", float("nan")
    cmd = ", ".join(f"{v:+.2f}" for v in command.tolist())
    seen = ", ".join(f"{v:+.2f}" for v in obs["policy"][:, cmd_slice].mean(dim=0).tolist())
    print(f"{cmd:>24} | {seen:>20} | {measured:>24} | {fell.float().mean().item():6.0%} | {height:9.2f} m", flush=True)

env.close()
simulation_app.close()
