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
