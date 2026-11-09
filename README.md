# Humanoid Locomotion Lab

Small hands-on project to gain experience with a full humanoid training pipeline.

My goal is to train a humanoid locomotion policy in one simulator, export it, deploy it in another simulator, and build a hardware-independent inference interface that could eventually be connected to a real robot.

The reference platform is a Unitree G1 robot.

## Software stack 

- Isaac Lab for GPU-accelerated simulation and training.
- Unitree RL Lab for a realistic G1 locomotion task and training setup.
- MuJoCo for independent policy evaluation.
- Unitree SDK2 as a reference for the eventual hardware interface, without attempting actual hardware communication.
- PyTorch / ONNX Runtime for policy inference and deployment.

## Quick start on a new machine

Requirements: Linux, an NVIDIA GPU with a recent driver (tested: RTX 2060 6 GB, driver 580), git.
Progress and detailed notes live in [doc/plan.md](doc/plan.md).

### 1. This repository

```bash
mkdir -p ~/git_repos && cd ~/git_repos
git clone https://github.com/erachelson/humanoid-locomotion-lab.git
```

### 2. Notebook curriculum (Isaac Lab 3.0 + Newton, no Isaac Sim)

This is all the notebooks in `notebooks/` need. About 10 GB of downloads, mostly PyTorch/CUDA.

```bash
# uv (Python package manager), if missing
curl -LsSf https://astral.sh/uv/install.sh | sh     # then open a new shell

# Isaac Lab 3.0.0 in its own folder, with its own uv environment (Python 3.12)
cd ~/git_repos
git clone --branch v3.0.0 --depth 1 https://github.com/isaac-sim/IsaacLab.git IsaacLab-3.0
cd IsaacLab-3.0 && uv sync

# Smoke test: train the cart-pole from the command line (a few minutes)
uv run isaaclab train --rl_library rsl_rl --task Isaac-Cartpole-Direct physics=newton_mjwarp
```

Start JupyterLab from this repository and open `notebooks/01_simulator_basics.ipynb`:

```bash
cd ~/git_repos/humanoid-locomotion-lab
uv run --project ~/git_repos/IsaacLab-3.0 --with jupyterlab jupyter lab
```

The skeleton notebooks are generated together with their solutions from `notebooks/_build/lessonNN.py`.
After editing a lesson source, rebuild both with:

```bash
uv run --project ~/git_repos/IsaacLab-3.0 --with nbformat python notebooks/_build/build.py notebooks/_build/lesson01.py
```

On the original machine, `IsaacLab-3.0` is a git worktree of an existing Isaac Lab checkout
(`git worktree add --detach ~/git_repos/IsaacLab-3.0 v3.0.0`); a plain clone as above is equivalent.

### 3. G1 pipeline (Phase 1: Isaac Lab 2.3 + Isaac Sim 5.1, unitree_rl_lab, MuJoCo sim2sim)

Only needed to train the G1 or run the sim2sim. Original setup: conda env `env_isaaclab` with Python 3.11,
Isaac Sim 5.1.0 (pip), torch 2.7.0+cu128, Isaac Lab v2.3.0, unitree_rl_lab 0.2.1, rsl-rl-lib 3.0.1.
Follow the official guides for exact commands:
[Isaac Lab pip installation](https://isaac-sim.github.io/IsaacLab/v2.3.0/source/setup/installation/pip_installation.html)
and [unitree_rl_lab](https://github.com/unitreerobotics/unitree_rl_lab).

```bash
conda create -n env_isaaclab python=3.11 -y && conda activate env_isaaclab
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
cd ~/git_repos && git clone --branch v2.3.0 https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab && ./isaaclab.sh --install

cd ~/git_repos
git clone https://github.com/unitreerobotics/unitree_rl_lab.git
git clone https://github.com/unitreerobotics/unitree_ros.git        # robot descriptions (set UNITREE_ROS_DIR)
git clone https://github.com/unitreerobotics/unitree_mujoco.git
git clone https://github.com/unitreerobotics/unitree_sdk2.git
pip install -e unitree_rl_lab/source/unitree_rl_lab
pip install mujoco onnxruntime
```

Then follow the Phase 1 checklist in [doc/plan.md](doc/plan.md) for the parts that need sudo or a build:
system packages (`libyaml-cpp-dev libboost-all-dev libeigen3-dev libspdlog-dev libfmt-dev libglfw3-dev`),
unitree_sdk2 (`sudo make install`), the MuJoCo 3.3.6 C release symlinked into `unitree_mujoco/simulate/`,
the simulator and the `g1_ctrl` controller builds, the config files, and "Running the MuJoCo sim2sim".

Trained policies are not in this repository: they live in `~/git_repos/unitree_rl_lab/logs/rsl_rl/`.
To keep the walking policy, copy `logs/rsl_rl/unitree_g1_29dof_velocity/2026-11-09_00-14-03/` from the
old machine (`exported/` and `params/` are enough for the sim2sim).

Training on a 6 GB GPU: use `--num_envs 2048` (the default 4096 runs out of memory).
