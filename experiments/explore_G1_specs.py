"""Print the MDP interface of an Isaac Lab task: observations, actions, joint order, timing and actuator gains.

Usage:
    ~/git_repos/IsaacLab/isaaclab.sh -p experiments/explore_G1_specs.py                      # Unitree G1 29-DoF
    ~/git_repos/IsaacLab/isaaclab.sh -p experiments/explore_G1_specs.py --task Isaac-Velocity-Flat-G1-v0
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", default="Unitree-G1-29dof-Velocity", help="Registered gym task id.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import isaaclab_tasks  # noqa: F401 — register Isaac Lab task environments
import unitree_rl_lab.tasks  # noqa: F401 — register Unitree RL Lab task environments
from isaaclab_tasks.utils import parse_env_cfg

task_id = args.task
print(f"Task: {task_id}", flush=True)
env_cfg = parse_env_cfg(task_id, device="cuda:0", num_envs=1)
env = gym.make(task_id, cfg=env_cfg)

print("Observation space:", env.observation_space, flush=True)
print("Policy observation dimensions:", env.observation_space["policy"].shape, flush=True)
print("Action space dimensions:", env.action_space.shape, flush=True)

obs_manager = env.unwrapped.observation_manager
for group, terms in obs_manager.active_terms.items():
    print(f"Observation group '{group}' terms (in order):", flush=True)
    for name, dim in zip(terms, obs_manager.group_obs_term_dim[group]):
        print(f"  {name}: {dim}", flush=True)

action_manager = env.unwrapped.action_manager
print("Action terms (in order):", flush=True)
for name, dim in zip(action_manager.active_terms, action_manager.action_term_dim):
    term = action_manager.get_term(name)
    print(f"  {name}: {dim} ({type(term).__name__})", flush=True)
    # _joint_names is private in Isaac Lab, so it might change between versions
    joint_names = getattr(term, "_joint_names", None)
    if joint_names is not None:
        for i, joint in enumerate(joint_names):
            print(f"    [{i:2d}] {joint}", flush=True)

robot = env.unwrapped.scene["robot"]
print("Robot joint order (used by joint_pos / joint_vel observations):", flush=True)
for i, joint in enumerate(robot.joint_names):
    print(f"  [{i:2d}] {joint}", flush=True)
if joint_names is not None:
    print("Action joint order matches robot joint order:", list(joint_names) == list(robot.joint_names), flush=True)

base_env = env.unwrapped
print(f"Physics dt: {base_env.physics_dt} s ({1.0 / base_env.physics_dt:.0f} Hz)", flush=True)
print(f"Decimation: {base_env.cfg.decimation}", flush=True)
print(f"Policy dt: {base_env.step_dt} s ({1.0 / base_env.step_dt:.0f} Hz)", flush=True)

# Values actually applied to the simulated robot (env 0)
data = robot.data
print("Per-joint actuator parameters (env 0):", flush=True)
print(f"  {'idx':>3}  {'joint':<28} {'Kp':>7} {'Kd':>6} {'effort':>7} {'armature':>8} {'default_pos':>11}", flush=True)
for i, joint in enumerate(robot.joint_names):
    print(
        f"  {i:3d}  {joint:<28} {data.joint_stiffness[0, i].item():7.1f} {data.joint_damping[0, i].item():6.1f}"
        f" {data.joint_effort_limits[0, i].item():7.1f} {data.joint_armature[0, i].item():8.3f}"
        f" {data.default_joint_pos[0, i].item():11.3f}",
        flush=True,
    )

env.close()
simulation_app.close()