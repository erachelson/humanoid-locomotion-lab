# Learning plan: robot description → trained policy → MuJoCo sim2sim

Goal: take a robot description, train a locomotion policy in Isaac Lab, and deploy it in a MuJoCo
simulation of the same robot (sim2sim). Reference platform: Unitree G1 (29 DoF).

Last updated: 2026-11-07.

## Key decision: which G1 task

| Task | Robot model | Use |
|---|---|---|
| `Isaac-Velocity-Flat-G1-v0` (Isaac Lab) | Older 37-DoF G1 (Dex hand joints, single `torso_joint`) | Learning how Isaac Lab tasks are built only |
| `Unitree-G1-29dof-Velocity` (unitree_rl_lab) | G1 29-DoF, same as `unitree_mujoco` and the real robot | **The actual pipeline** |

`experiments/explore_G1_specs.py` prints the full MDP interface of either task
(`--task <id>`, default `Unitree-G1-29dof-Velocity`).

## Phase 1: run the reference pipeline end to end (about 1 week)

Run unitree_rl_lab's own train → play → MuJoCo sim2sim before trying to understand it. This gives a
known-good baseline, so later bugs can be attributed to your own code.

Status:

- [x] Isaac Lab 2.3 / Isaac Sim 5.1 in conda env `env_isaaclab`
- [x] unitree_rl_lab installed (editable), `UNITREE_ROS_DIR` set to `~/git_repos/unitree_ros`
- [x] `mujoco` 4.0.0 installed in `env_isaaclab`
- [x] `~/git_repos/unitree_mujoco` cloned; `simulate/config.yaml` set to `robot: "g1"`, `domain_id: 0`,
      `interface: "lo"`, `use_joystick: 1`, `enable_elastic_band: 1`
- [x] `~/git_repos/unitree_sdk2` cloned
- [x] Training smoke test: 2048 envs fit on the RTX 2060 (≈4.8 GB peak, ≈2.4 s/iteration);
      4096 envs (the default) runs out of GPU memory
- [x] Install system dependencies (needs sudo):
      `sudo apt install -y libyaml-cpp-dev libboost-all-dev libeigen3-dev libspdlog-dev libfmt-dev`
- [x] Build and install unitree_sdk2 (needs sudo):
      `cd ~/git_repos/unitree_sdk2 && mkdir build && cd build && cmake .. -DBUILD_EXAMPLES=OFF && sudo make install`
- [x] MuJoCo 3.3.6 C release in `~/.mujoco/mujoco-3.3.6`, symlinked as `unitree_mujoco/simulate/mujoco`
      (the simulator compiles MuJoCo's `simulate/` sources, e.g. `glfw_adapter.h`; the pip `mujoco` package does not ship them)
- [x] Install GLFW headers (needs sudo): `sudo apt install -y libglfw3-dev`
- [x] Build the unitree_mujoco simulator: `cd ~/git_repos/unitree_mujoco/simulate/build && cmake .. && make -j6`
- [x] Build the G1 deploy controller: `cd ~/git_repos/unitree_rl_lab/deploy/robots/g1_29dof && mkdir build && cd build && cmake .. && make`
      → `build/g1_ctrl`. It reads `config/config.yaml`; the `Velocity` state's `policy_dir` defaults to
      Unitree's bundled pretrained policy `config/policy/velocity/v0` (`exported/policy.onnx` + `params/deploy.yaml`).
      If `policy_dir` points to a folder without `exported/`, it takes the alphabetically last subfolder
      that has one, i.e. the latest exported run when pointed at `logs/rsl_rl/unitree_g1_29dof_velocity`.
- [x] Gamepad: Xbox One X pad (USB, `xpad` driver) on `/dev/input/js0`; its layout matches the
      simulator's `joystick_type: "xbox"` mapping.
- [x] Sim2sim with Unitree's bundled pretrained policy: the G1 walks in MuJoCo (2026-11-07).
      See "Running the MuJoCo sim2sim" below.
- [ ] Train:
      ```bash
      cd ~/git_repos/unitree_rl_lab
      python scripts/rsl_rl/train.py --headless --task Unitree-G1-29dof-Velocity --num_envs 2048 --max_iterations 5000
      ```
      (Default `max_iterations` is 50000. At ≈2.4 s/iteration, 5000 iterations take ≈3.5 h. Watch
      the curves with `tensorboard --logdir logs/rsl_rl` and stop or resume as needed.)
- [ ] Play and export: `python scripts/rsl_rl/play.py --task Unitree-G1-29dof-Velocity`
      → `logs/.../exported/policy.onnx`, plus `params/deploy.yaml` written at training time
- [ ] Sim2sim with your own policy: set `policy_dir` in
      `unitree_rl_lab/deploy/robots/g1_29dof/config/config.yaml` (`Velocity` state) to your run folder,
      e.g. `../../../logs/rsl_rl/unitree_g1_29dof_velocity/<run>`, and give it a specific run: the
      parent folder resolves to the latest exported run, which today is the aborted
      `2026-11-06_16-36-56` (only `model_0.pt`). The config is read at startup, so restart `g1_ctrl`.

### Running the MuJoCo sim2sim

Two terminals, simulator first. Both must use the same DDS domain (0) and network interface (`lo`).

```bash
# Terminal 1: simulator (settings in ~/git_repos/unitree_mujoco/simulate/config.yaml)
cd ~/git_repos/unitree_mujoco/simulate/build && ./unitree_mujoco

# Terminal 2: controller. --network lo is required: without it, CycloneDDS picks another
# interface (eno1) and the controller hangs on "Waiting for connection rt/lowstate".
cd ~/git_repos/unitree_rl_lab/deploy/robots/g1_29dof/build && ./g1_ctrl --network lo
```

The robot starts hanging from the elastic band. Then (gamepad = Xbox controller; keys are typed
with the MuJoCo window focused):

| Step | Input | Effect |
|---|---|---|
| 1 | LT + D-pad Up | Passive → FixStand: moves to the standing pose |
| 2 | `8` (a few times) | Lowers the elastic band until the feet touch the ground (`7` raises it) |
| 3 | RB + X | FixStand → Velocity: the RL policy takes over |
| 4 | `9` | Releases the elastic band |
| 5 | Sticks | Velocity commands (walk forward/sideways, turn) |

Recovery: LT + B → Passive (joints go limp); `Backspace` resets the simulation; `9` turns the band
back on. Bundled dance policies: from Velocity, hold LT 2 s then D-pad Down (Dance 102) or Left
(Gangnam Style); RB + X returns to Velocity. All transitions are defined in
`unitree_rl_lab/deploy/robots/g1_29dof/config/config.yaml`.

Troubleshooting:

- Controller stuck on "Waiting for connection rt/lowstate": the controller and simulator are on
  different interfaces or domains. Check that `g1_ctrl` was started with `--network lo` and that
  `interface: "lo"` / `domain_id: 0` are set in the simulator config. Alternative: use `eno1` in both.
- Gamepad has no effect: check that the controller is running (the simulator only forwards gamepad
  state; `g1_ctrl` acts on it), that `use_joystick: 1` is set, and that `/dev/input/js0` exists.
  `~/git_repos/unitree_mujoco/simulate/build/jstest` shows the raw gamepad input.
- Robot falls when the band is released: the policy is not active yet (step 3) or the feet were not
  on the ground; reset with `Backspace`, re-enable the band with `9` and redo steps 1–4.

## Phase 2: learn the manager-based environment by mapping it to an MDP (a few days)

| MDP concept | Isaac Lab config |
|---|---|
| S (observations) | `ObservationsCfg` |
| A (actions) | `ActionsCfg` |
| R (rewards) | `RewardsCfg` |
| Termination | `TerminationsCfg` |
| Initial state distribution, domain randomization | `EventCfg` |
| Task goal (velocity command) | `CommandsCfg` |
| Transition dynamics | `SceneCfg` + robot `ArticulationCfg` + `sim.dt` / `decimation` |

Read `unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/g1/29dof/velocity_env_cfg.py`
with this table in mind. Ignore Omniverse, USD internals and the rendering stack.

## Phase 3: the robotics you actually need (in parallel)

- Robot descriptions: URDF and MJCF (links, joints, inertias, limits); compare the G1 files in
  `unitree_ros/robots/g1_description/` and `unitree_mujoco/unitree_robots/g1/`.
- PD position control: τ = Kp(q* − q) − Kd·q̇, effort limits, armature.
- Frames: world vs base frame, quaternion conventions (Isaac: wxyz), projected gravity.

## Phase 4: write your own Python sim2sim (in `deployment/`)

Replace Unitree's C++ deploy with a Python MuJoCo loop:

1. Load the MJCF (`unitree_mujoco/unitree_robots/g1/scene_29dof.xml`).
2. Physics at the training rate (200 Hz; note MuJoCo's G1 scene defaults to 0.002 s = 500 Hz),
   PD torques every physics step.
3. Every 4th step (50 Hz): build the observation, run `policy.onnx`.
4. Target = default joint pos + action scale × action.

Facts about `Unitree-G1-29dof-Velocity` that the MuJoCo side must reproduce (from
`explore_G1_specs.py` and `params/deploy.yaml`):

- Policy observation: 480 values = 6 terms × 5-step history: `base_ang_vel` (3), `projected_gravity` (3),
  `velocity_commands` (3), `joint_pos_rel` (29), `joint_vel_rel` (29), `last_action` (29).
  No base linear velocity for the actor (the critic gets it: asymmetric actor-critic).
- Observation scales (deploy.yaml): base_ang_vel 0.2, joint_vel 0.05, others 1.0.
- Action: 29 joint position targets, scale 0.25, offset = default joint positions.
- Joint order is Isaac's breadth-first order (interleaved left/right/waist), not the MJCF order:
  map by joint name.
- Per-joint Kp / Kd / effort limit / armature / default position: printed by `explore_G1_specs.py`.

Typical sim2sim bugs, to diff term by term: joint order, default positions, action scale,
observation order/scale/history stacking, velocity frame, quaternion convention, gains/armature, timing.

## Phase 5: generalize

Use a different robot description, change domain randomization, and measure how sim2sim robustness
changes.

## Resources

- Local code: unitree_rl_lab task configs, `utils/export_deploy_cfg.py`, `deploy/robots/g1_29dof/src`.
- Isaac Lab tutorials: "Creating a Manager-Based RL Environment", "Training with an RL Agent".
- Rudin et al. 2021, *Learning to Walk in Minutes Using Massively Parallel Deep RL* (legged_gym, rsl_rl).
- Humanoid-Gym (RobotEra, 2024): Isaac Gym → MuJoCo sim2sim for humanoids; readable sim2sim script.
- MuJoCo documentation: MJCF modeling guide, actuators (`motor` vs `position`), `armature`.
- unitree_mujoco, unitree_ros: official G1 MJCF and URDF.
- *Modern Robotics* (Lynch & Park), chapters 2–4 and 8, if background is needed.
