"""Lesson 1 source: simulator basics with Isaac Lab 3.0 + Newton (cart-pole)."""

NAME = "01_simulator_basics"

CELLS = [
    (
        "md",
        r"""
# Lesson 1 — Simulator basics

**Goal:** drive a physics simulation yourself, with no RL yet. By the end you can build a scene from
configuration classes, run many copies of it in parallel on the GPU, read the robot's state, apply
forces, reset some copies but not others, run a controller at a lower rate than the physics, and get
camera images. These are exactly the pieces an RL environment is made of (lesson 2).

**Robot:** the cart-pole: a cart on a rail (prismatic joint `slider_to_cart`) carrying a pole on a
hinge (revolute joint `cart_to_pole`). Pole angle 0 is **upright**; π is hanging down.

**How to work:** cells marked `# TODO` are yours; each exercise ends with a **check cell** that
should run without error. {{SOLUTIONS_NOTE}}
Restarting the kernel is never required: every exercise builds and closes its own simulation.

**Run Jupyter** from this project's folder with the Isaac Lab 3.0 environment (`--project` selects
it; `--with jupyterlab` adds JupyterLab on top without modifying it; for the classic interface use
`--with notebook jupyter notebook`):
```bash
cd ~/git_repos/humanoid-locomotion-lab && uv run --project ~/git_repos/IsaacLab-3.0 --with jupyterlab jupyter lab
```

**Reference code:** `~/git_repos/IsaacLab-3.0/scripts/tutorials/` (`01_assets/run_articulation.py`,
`02_scene/create_scene.py`). API docs: `isaaclab.sim`, `isaaclab.scene`, `isaaclab.assets`.
""",
    ),
    (
        "md",
        r"""
## 0. The stack in one picture

```
your code ──► Isaac Lab ──────────────────────────────► Newton (physics) ──► Warp (GPU kernels)
              SimulationCfg / SimulationContext         MJWarp solver = MuJoCo's solver on GPU
              InteractiveScene (N cloned envs)
              Articulation (robot state & commands)
```

* **`SimulationCfg`**: physics step `dt`, device, which physics backend (`NewtonCfg` + a solver).
* **`launch_simulation(cfg, ...)`**: context manager that starts whatever runtime the config needs
  (with Newton: nothing heavy — no Isaac Sim, no Omniverse app).
* **`SimulationContext`**: the running simulation. A **singleton**: one per process at a time, so we
  free it (`SimulationContext.clear_instance()`) before building the next one.
* **`InteractiveSceneCfg`**: a declarative list of things in the world. Anything whose prim path
  contains `{ENV_REGEX_NS}` is **cloned** once per environment (`/World/envs/env_0/...`,
  `env_1`, ...); everything else (ground, light) exists once.
* **Data on the GPU:** the robot state lives in Warp arrays; `.torch` gives a zero-copy torch view
  of shape `(num_envs, ...)`. Everything is batched over environments.

Run the setup cell (it takes a few seconds the first time).
""",
    ),
    (
        "code",
        r"""
import time
from contextlib import contextmanager

import warp as wp

wp.config.enable_backward = False  # set before importing Isaac Lab modules that define Warp kernels

import matplotlib.pyplot as plt
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.app import launch_simulation
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass, instantiate, replace
from isaaclab_assets import CARTPOLE_CFG
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_newton.renderers import NewtonWarpRendererCfg

DEVICE = "cuda:0"
PHYSICS_DT = 1 / 200  # 200 Hz physics, as in the G1 task
CART, POLE = 0, 1  # joint indices: slider_to_cart, cart_to_pole
print(torch.cuda.get_device_name(0))
""",
    ),
    (
        "md",
        r"""
## 1. Describe the scene

`CARTPOLE_CFG` (from `isaaclab_assets`) is an `ArticulationCfg`: which USD file to load, the initial
state, and the **actuators** (how commands become joint torques). Print it and look at the
`actuators` field — you will need it in exercise 4.

**Exercise 1.** Complete the scene: add the cart-pole as an entry named `cartpole`, cloned in every
environment. Use `replace(CARTPOLE_CFG, prim_path=...)` with the path `"{ENV_REGEX_NS}/Robot"`.
""",
    ),
    (
        "code",
        r"""
print(CARTPOLE_CFG.actuators)
""",
    ),
    (
        "code",
        r"""
@configclass
class CartpoleSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0))
    #>> TODO: add the cart-pole articulation, cloned per environment, under the name `cartpole`
    cartpole: ArticulationCfg = replace(CARTPOLE_CFG, prim_path="{ENV_REGEX_NS}/Robot")
    #<<
""",
    ),
    (
        "code",
        r"""
# check
cfg = CartpoleSceneCfg(num_envs=4, env_spacing=2.0)
assert cfg.cartpole.prim_path == "{ENV_REGEX_NS}/Robot", cfg.cartpole.prim_path
print("ok")
""",
    ),
    (
        "md",
        r"""
## 2. Open a simulation

We will open many short simulations, so we wrap the boilerplate in a context manager:

```python
with open_sim(num_envs=16) as (sim, scene):
    ...  # use the simulation
# here it is closed and freed
```

**Exercise 2.** Complete `open_sim`, in this order:
1. build the `SimulationCfg` (physics `dt`, device, `physics=NewtonCfg(solver_cfg=MJWarpSolverCfg())`,
   `visualizer_cfgs=[]` for no viewer window);
2. inside `launch_simulation(sim_cfg, LAUNCH_ARGS)`: create the `SimulationContext(sim_cfg)`;
3. create the scene with `instantiate(scene_cfg_class(num_envs=..., env_spacing=2.0))`;
4. call `sim.reset()` — this initializes physics and fills the robot's data buffers;
5. `yield sim, scene`, and in a `finally:` free the singleton with `SimulationContext.clear_instance()`.
""",
    ),
    (
        "code",
        r"""
LAUNCH_ARGS = {"visualizer": None, "visualizer_explicit": True}  # headless: no viewer window


@contextmanager
def open_sim(num_envs, scene_cfg_class=CartpoleSceneCfg, dt=PHYSICS_DT):
    #>> TODO: build sim_cfg (step 1)
    sim_cfg = sim_utils.SimulationCfg(
        dt=dt, device=DEVICE, physics=NewtonCfg(solver_cfg=MJWarpSolverCfg()), visualizer_cfgs=[]
    )
    #<<
    with launch_simulation(sim_cfg, LAUNCH_ARGS):
        #>> TODO: steps 2 to 5 (SimulationContext, scene, reset, then yield and clean up)
        sim = SimulationContext(sim_cfg)
        scene = instantiate(scene_cfg_class(num_envs=num_envs, env_spacing=2.0))
        sim.reset()
        try:
            yield sim, scene
        finally:
            SimulationContext.clear_instance()
        #<<
""",
    ),
    (
        "code",
        r"""
# check (the first run compiles GPU kernels and takes longer)
with open_sim(num_envs=4) as (sim, scene):
    assert scene.num_envs == 4 and abs(sim.get_physics_dt() - PHYSICS_DT) < 1e-9
with open_sim(num_envs=8) as (sim, scene):  # a second simulation in the same kernel
    assert scene.num_envs == 8
print("ok")
""",
    ),
    (
        "md",
        r"""
## 3. Inspect the robot

`scene["cartpole"]` is an `Articulation`. Useful attributes:

| | |
|---|---|
| `robot.joint_names`, `robot.body_names`, `robot.num_joints` | structure |
| `robot.data.joint_pos.torch`, `.joint_vel.torch` | current state, shape `(num_envs, num_joints)` |
| `robot.data.default_joint_pos.torch` | initial joint positions from the config |
| `robot.data.joint_pos_limits.torch` | `(num_envs, num_joints, 2)`: lower, upper |
| `robot.data.joint_effort_limits.torch` | max force/torque per joint |
| `robot.data.body_mass.torch` | `(num_envs, num_bodies)` |
| `scene.env_origins` | `(num_envs, 3)` position of each environment's origin |

**Exercise 3.** Fill the `info` dict with values for **environment 0** (plain Python floats/lists:
use `.tolist()` or `.item()`). Then answer for yourself: how far can the cart travel? How heavy are the
cart and the pole? Where are the environments placed?
""",
    ),
    (
        "code",
        r"""
with open_sim(num_envs=4) as (sim, scene):
    robot = scene["cartpole"]
    #>> TODO: fill the dict below (environment 0 only) // info = {"joint_names": ..., "body_names": ..., "masses": ..., "joint_pos_limits": ..., "effort_limits": ..., "default_joint_pos": ..., "env_origins": ...}
    info = {
        "joint_names": robot.joint_names,
        "body_names": robot.body_names,
        "masses": robot.data.body_mass.torch[0].tolist(),
        "joint_pos_limits": robot.data.joint_pos_limits.torch[0].tolist(),
        "effort_limits": robot.data.joint_effort_limits.torch[0].tolist(),
        "default_joint_pos": robot.data.default_joint_pos.torch[0].tolist(),
        "env_origins": scene.env_origins.tolist(),  # all envs: where are they?
    }
    #<<
for key, value in info.items():
    print(f"{key:>18}: {value}")
""",
    ),
    (
        "code",
        r"""
# check
assert info["joint_names"] == ["slider_to_cart", "cart_to_pole"]
assert len(info["masses"]) == len(info["body_names"])
MASSES = dict(zip(info["body_names"], info["masses"]))
print("ok — masses by body:", MASSES)
""",
    ),
    (
        "md",
        r"""
## 4. The simulation loop: passive dynamics

One physics step is always the same three calls:

```python
scene.write_data_to_sim()       # push commands (efforts, targets) from Isaac Lab buffers to the solver
sim.step()                      # advance physics by dt
scene.update(dt)                # pull the new state into robot.data (and update sensors)
```

To set a state directly (e.g. an initial condition), write it to the simulation:
`robot.write_joint_position_to_sim_index(position=..., joint_ids=..., env_ids=...)` (same for
velocity). `joint_ids`/`env_ids` default to "all"; the tensor shape must match the selection.

**Exercise 4.** Start every environment with the pole tilted by 0.1 rad (cart at 0, everything at
rest), then simulate 2 s **without any command** and record the cart position and pole angle of env 0
at every step into `traj` (shape `(steps, 2)`).
""",
    ),
    (
        "code",
        r"""
def step(sim, scene):
    scene.write_data_to_sim()
    sim.step()
    scene.update(sim.get_physics_dt())


steps = int(2.0 / PHYSICS_DT)
with open_sim(num_envs=4) as (sim, scene):
    robot = scene["cartpole"]
    #>> TODO: write the initial state (pole at 0.1 rad, everything else 0) // then loop: step, and record robot.data.joint_pos of env 0 into a list; stack it into `traj`
    q0 = torch.zeros(scene.num_envs, 2, device=DEVICE)
    q0[:, POLE] = 0.1
    robot.write_joint_position_to_sim_index(position=q0)
    robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q0))
    rows = []
    for _ in range(steps):
        step(sim, scene)
        rows.append(robot.data.joint_pos.torch[0].clone())  # clone: the view is overwritten next step
    traj = torch.stack(rows).cpu().numpy()
    #<<

t = np.arange(steps) * PHYSICS_DT
fig, ax = plt.subplots(figsize=(8, 3))
ax.plot(t, traj[:, CART], label="cart position [m]")
ax.plot(t, traj[:, POLE], label="pole angle [rad]")
ax.axhline(np.pi, ls=":", c="gray")
ax.set_xlabel("time [s]"); ax.legend(); plt.show()
""",
    ),
    (
        "code",
        r"""
# check
assert traj.shape == (steps, 2)
assert abs(traj[0, POLE] - 0.1) < 0.05 and traj[-1, POLE] > 1.0, "the pole should fall from 0.1 rad"
print("ok")
""",
    ),
    (
        "md",
        r"""
**Questions** (answer from the plot, then check `CARTPOLE_CFG.actuators`):
1. Nobody pushes the cart, yet it moves while the pole falls. Why?
2. The cart slows down again. What in the actuator config acts on the cart even with a zero command?
   (*Hint:* an **implicit** actuator is a PD controller inside the solver:
   τ = stiffness·(q_target − q) + damping·(q̇_target − q̇) + effort command.)
3. Does the pole keep swinging or settle around π (hanging)? What does that say about damping on
   the pole joint?
""",
    ),
    (
        "md",
        r"""
## 5. Parallel environments are a batch dimension

Each environment is an independent copy of the world, simulated in the same GPU step. You can give
each one a different initial condition (or force, or gain): a parameter sweep in one run.

**Exercise 5.** With 9 environments, start env *i* with pole angle `angles[i]` from
`torch.linspace(-0.2, 0.2, 9)`. Simulate 1 s and record the pole angle of **all** environments
(`traj_all`, shape `(steps, 9)`).
""",
    ),
    (
        "code",
        r"""
angles = torch.linspace(-0.2, 0.2, 9, device=DEVICE)
steps = int(1.0 / PHYSICS_DT)
with open_sim(num_envs=9) as (sim, scene):
    robot = scene["cartpole"]
    #>> TODO: one write for all envs (only the pole joint: joint_ids=[POLE]); zero velocities // loop and record the pole angle of all envs into `traj_all` (numpy, shape (steps, 9))
    robot.write_joint_position_to_sim_index(position=angles[:, None], joint_ids=[POLE])
    robot.write_joint_velocity_to_sim_index(velocity=torch.zeros(9, 2, device=DEVICE))
    rows = []
    for _ in range(steps):
        step(sim, scene)
        rows.append(robot.data.joint_pos.torch[:, POLE].clone())
    traj_all = torch.stack(rows).cpu().numpy()
    #<<

fig, ax = plt.subplots(figsize=(8, 3))
ax.plot(np.arange(steps) * PHYSICS_DT, traj_all)
ax.set_xlabel("time [s]"); ax.set_ylabel("pole angle [rad]"); plt.show()
""",
    ),
    (
        "code",
        r"""
# check: the system is symmetric, so env i and env 8-i should mirror each other
assert traj_all.shape == (steps, 9)
assert np.allclose(traj_all[:, 0], -traj_all[:, 8], atol=1e-2), "mirror symmetry broken"
assert np.abs(traj_all[:, 4]).max() < 1e-3, "the perfectly upright pole should stay up"
print("ok")
""",
    ),
    (
        "md",
        r"""
## 6. Applying forces — and checking the simulator against physics

Commands go through the robot's **actuators**:
`robot.actuators.target_command.set_effort_index(value=..., joint_ids=[CART])` sets an effort
(here: a force in N on the cart joint) that is applied at every step until you change it.

A physics check. The pole hangs down (θ = π), everything at rest, and a force F pushes the cart.
What is the cart's acceleration **at the first instant**? Three candidate answers:

* **rigid:** cart and pole move together → a = F / (m_cart + m_pole);
* **free cart:** the pole does not resist at all → a = F / m_cart;
* **hinge:** the pole is on a frictionless hinge, so it starts to rotate backwards and only part of
  its mass is dragged along. Writing Newton's law for the cart and the pole's rotation about its
  centre of mass (distance l from the hinge, inertia I about its centre of mass) gives
  a = F / (m_cart + m_pole · I / (I + m_pole·l²)).

**Exercise 6.** Read `I` and `l` for the pole from the simulator (`robot.data.body_inertia.torch`:
`(num_envs, num_bodies, 9)`, a flattened 3×3 matrix — the pole's hinge axis is its x axis, so take
element 0; `robot.data.body_com_pos_b.torch`: centre-of-mass position in the body frame, take its
norm). Then start with the pole hanging at rest, apply F = 20 N on the cart for **one** physics step
and compute `a_measured` = (cart velocity after the step) / dt. Which candidate is right?
""",
    ),
    (
        "code",
        r"""
F = 20.0
with open_sim(num_envs=1) as (sim, scene):
    robot = scene["cartpole"]
    pole = robot.body_names.index("pole")
    #>> TODO: read I_pole and l_pole (floats) from robot.data // pole hanging at rest; apply F on the cart joint; one physics step; compute a_measured
    I_pole = robot.data.body_inertia.torch[0, pole, 0].item()
    l_pole = robot.data.body_com_pos_b.torch[0, pole].norm().item()
    robot.write_joint_position_to_sim_index(position=torch.tensor([[0.0, np.pi]], device=DEVICE))
    robot.write_joint_velocity_to_sim_index(velocity=torch.zeros(1, 2, device=DEVICE))
    robot.actuators.target_command.set_effort_index(value=torch.full((1, 1), F, device=DEVICE), joint_ids=[CART])
    step(sim, scene)
    a_measured = robot.data.joint_vel.torch[0, CART].item() / PHYSICS_DT
    #<<
m_c, m_p = MASSES["cart"], MASSES["pole"]
candidates = {
    "rigid": F / (m_c + m_p),
    "free cart": F / m_c,
    "hinge": F / (m_c + m_p * I_pole / (I_pole + m_p * l_pole**2)),
}
print(f"pole: I = {I_pole:.4f} kg·m², l = {l_pole:.3f} m")
print(f"measured {a_measured:.3f} m/s² | " + " | ".join(f"{k} {v:.3f}" for k, v in candidates.items()))
""",
    ),
    (
        "code",
        r"""
# check (within 2%)
assert abs(a_measured - candidates["hinge"]) / candidates["hinge"] < 0.02
print("ok")
""",
    ),
    (
        "md",
        r"""
**Questions:** (1) Measure the *average* acceleration over 0.05 s instead of one step: it is lower.
Which two effects lower it? (Exercise 4 gives one of them.) (2) Derive the hinge formula yourself
(two equations: horizontal forces on cart + pole, and torques on the pole about its centre of mass).

## 7. A controller running slower than the physics (decimation)

Robots (and RL policies) do not act at the physics rate: the G1 policy acts at 50 Hz while physics
runs at 200 Hz. Between two control updates, the last command is **held** (zero-order hold). The
ratio is the **decimation** (here 4).

A linear state-feedback controller can balance the cart-pole upright:
F = −(k_x·x + k_v·ẋ + k_θ·θ + k_ω·θ̇). Gains are given (found with the batch sweep below).

**Exercise 7.** Start all envs with small random pole angles (uniform in ±0.1 rad), run 5 s with the
controller updated every `DECIMATION` physics steps, and record the pole angle of every env at every
**control** step (`angles_log`, shape `(control_steps, num_envs)`).
""",
    ),
    (
        "code",
        r"""
DECIMATION = 4
K = torch.tensor([-10.0, -20.0, 400.0, 50.0], device=DEVICE)  # gains for (x, x_dot, theta, theta_dot)


def controller(robot):
    x, theta = robot.data.joint_pos.torch[:, CART], robot.data.joint_pos.torch[:, POLE]
    x_dot, theta_dot = robot.data.joint_vel.torch[:, CART], robot.data.joint_vel.torch[:, POLE]
    state = torch.stack([x, x_dot, theta, theta_dot], dim=1)
    return -(state @ K)  # force on the cart, shape (num_envs,)


num_envs = 32
control_steps = int(5.0 / (PHYSICS_DT * DECIMATION))
with open_sim(num_envs=num_envs) as (sim, scene):
    robot = scene["cartpole"]
    torch.manual_seed(0)
    #>> TODO: random initial pole angles in [-0.1, 0.1], zero velocities // for each control step: compute the force, set it on the cart, then do DECIMATION physics steps // record the pole angles after each control step into `angles_log` (numpy)
    q0 = torch.zeros(num_envs, 2, device=DEVICE)
    q0[:, POLE] = (torch.rand(num_envs, device=DEVICE) * 2 - 1) * 0.1
    robot.write_joint_position_to_sim_index(position=q0)
    robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q0))
    rows = []
    for _ in range(control_steps):
        force = controller(robot)
        robot.actuators.target_command.set_effort_index(value=force[:, None], joint_ids=[CART])
        for _ in range(DECIMATION):
            step(sim, scene)
        rows.append(robot.data.joint_pos.torch[:, POLE].clone())
    angles_log = torch.stack(rows).cpu().numpy()
    #<<

fig, ax = plt.subplots(figsize=(8, 3))
ax.plot(np.arange(control_steps) * PHYSICS_DT * DECIMATION, angles_log, lw=0.8)
ax.set_xlabel("time [s]"); ax.set_ylabel("pole angle [rad]"); plt.show()
""",
    ),
    (
        "code",
        r"""
# check: every pole balanced at the end
assert angles_log.shape == (control_steps, num_envs)
print("max |angle| over the last second:", np.abs(angles_log[-int(1 / (PHYSICS_DT * DECIMATION)):]).max())
assert np.abs(angles_log[-1]).max() < 0.05
print("ok")
""",
    ),
    (
        "md",
        r"""
**Bonus (batch sweep).** Use one environment per candidate gain: e.g. keep K but scale k_θ by
`torch.linspace(0.25, 2.0, num_envs)` (the gain becomes a tensor of shape `(num_envs,)`), start all
envs at 0.1 rad, and plot which ones balance. Then try DECIMATION = 20 (10 Hz control): what breaks?
Also look at the signs of K: the cart-position gain pushes the cart *away* from the origin. Why is
that needed to bring it back? (Think about where the pole must lean before the cart can move left.)

## 8. Resetting some environments but not others

In RL, each environment ends its episode at its own time (the pole fell, the cart left the rail)
and is reset **alone**, while the others keep running. All `write_*_to_sim_index` methods take
`env_ids` for this.

**Exercise 8.** Push every cart with random forces (normal, σ = 30 N, a new value every control step)
for 10 s. Whenever an env has |θ| > 0.4 rad or |x| > 2.5 m, reset **only that env** to the
default joint state plus a small random pole angle (±0.05 rad), with zero velocity. Count the resets.
Write the reset as a function `reset_envs(robot, env_ids)`.
""",
    ),
    (
        "code",
        r"""
def reset_envs(robot, env_ids):
    #>> TODO: default joint positions for these envs + random pole angle in ±0.05; zero velocities; write both for env_ids only
    q = robot.data.default_joint_pos.torch[env_ids].clone()
    q[:, POLE] += (torch.rand(len(env_ids), device=q.device) * 2 - 1) * 0.05
    robot.write_joint_position_to_sim_index(position=q, env_ids=env_ids)
    robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q), env_ids=env_ids)
    #<<


num_envs = 64
control_steps = int(10.0 / (PHYSICS_DT * DECIMATION))
resets = 0
with open_sim(num_envs=num_envs) as (sim, scene):
    robot = scene["cartpole"]
    reset_envs(robot, torch.arange(num_envs, device=DEVICE))
    for _ in range(control_steps):
        force = torch.randn(num_envs, 1, device=DEVICE) * 30.0
        robot.actuators.target_command.set_effort_index(value=force, joint_ids=[CART])
        for _ in range(DECIMATION):
            step(sim, scene)
        #>> TODO: find the envs out of bounds (a 1-D tensor of indices) and reset them; add their number to `resets`
        q = robot.data.joint_pos.torch
        done = (q[:, POLE].abs() > 0.4) | (q[:, CART].abs() > 2.5)
        env_ids = done.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            reset_envs(robot, env_ids)
            resets += len(env_ids)
        #<<
    final_q = robot.data.joint_pos.torch.clone()
print(f"{resets} resets in {num_envs} envs over 10 s")
""",
    ),
    (
        "code",
        r"""
# check
assert resets > num_envs, "with random pushes, poles should fall many times"
assert (final_q[:, POLE].abs() <= 0.4 + 0.2).all(), "out-of-bounds envs should have been reset"
print("ok")
""",
    ),
    (
        "md",
        r"""
## 9. Seeing the scene: a camera sensor

Sensors are scene entries too. A `CameraCfg` with prim path under `{ENV_REGEX_NS}` gives **one camera
per environment**, all rendered in a batch by Newton's Warp renderer (no RTX needed). After
`scene.update`, images are in `scene["camera"].data.output["rgb"]` (shape `(num_envs, H, W, 3)`).

Config classes compose by inheritance: `CartpoleCameraSceneCfg` is your scene plus a camera.

**Exercise 9.** Point each env's camera at its own cart-pole from the side (the rail runs along y, so
look along x): eyes at `env_origin + (4, 0, 2.5)`,
looking at `env_origin + (0, 0, 2)`, with `camera.set_world_poses_from_view(eyes=..., targets=...)`.
Start the 4 envs with pole angles 0, 0.5, 1.5 and π, step 10 times, and show the 4 images.
""",
    ),
    (
        "code",
        r"""
@configclass
class CartpoleCameraSceneCfg(CartpoleSceneCfg):
    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera",
        data_types=["rgb"],
        width=160,
        height=120,
        spawn=sim_utils.PinholeCameraCfg(),
        renderer_cfg=NewtonWarpRendererCfg(),
    )


with open_sim(num_envs=4, scene_cfg_class=CartpoleCameraSceneCfg) as (sim, scene):
    robot, camera = scene["cartpole"], scene["camera"]
    #>> TODO: place the cameras (one per env) // set the 4 pole angles, step 10 times, then copy the images to `images` (numpy, (4, H, W, 3))
    eyes = scene.env_origins + torch.tensor([4.0, 0.0, 2.5], device=DEVICE)
    targets = scene.env_origins + torch.tensor([0.0, 0.0, 2.0], device=DEVICE)
    camera.set_world_poses_from_view(eyes=eyes, targets=targets)
    poles = torch.tensor([0.0, 0.5, 1.5, np.pi], device=DEVICE)
    robot.write_joint_position_to_sim_index(position=poles[:, None], joint_ids=[POLE])
    for _ in range(10):
        step(sim, scene)
    rgb = camera.data.output["rgb"]
    images = (rgb.torch if hasattr(rgb, "torch") else rgb).cpu().numpy()
    #<<

fig, axes = plt.subplots(1, 4, figsize=(14, 3))
for ax, img, title in zip(axes, images, ["0", "0.5", "1.5", "π"]):
    ax.imshow(img[..., :3]); ax.set_title(f"pole angle {title} rad"); ax.axis("off")
plt.show()
""",
    ),
    (
        "code",
        r"""
# check
assert images.shape[0] == 4 and images.shape[-1] >= 3 and images.mean() > 0
print("ok")
""",
    ),
    (
        "md",
        r"""
## Summary — and what it has to do with RL

| You did | In an RL environment (lesson 2) it is |
|---|---|
| scene config, `open_sim` | environment construction |
| reading `joint_pos`, `joint_vel` | **observations** |
| `set_effort_index` on the cart, held for `DECIMATION` steps | **actions**, control rate vs physics rate |
| out-of-bounds test | **terminations** |
| `reset_envs(env_ids)` with random initial angles | **resets** and the **initial-state distribution** |
| 64 envs in one GPU step | the batch that makes PPO fast |

Missing for RL: a **reward**, and an agent that replaces the hand-made controller. That is lesson 2:
you will write the cart-pole as an Isaac Lab `DirectRLEnv`, then (lesson 3) train it with your own PPO.
""",
    ),
]
