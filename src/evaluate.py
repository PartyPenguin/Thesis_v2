import gymnasium as gym
import torch as th
from ManiSkill.mani_skill.envs.sapien_env import BaseEnv
import yaml
from pathlib import Path
import os
import wandb
import numpy as np
from src.train import GATPolicy, POLICY_MAP

device = "cuda" if th.cuda.is_available() else "cpu"


def initialize_environment(config: dict, seed: int) -> BaseEnv:
    env: BaseEnv = gym.make(
        id=config["env"]["env_id"],
        obs_mode=config["env"]["obs_mode"],
        control_mode=config["env"]["control_mode"],
        render_mode=config["evaluate"]["render_mode"],
    )
    env.reset(seed=seed)
    np.random.seed(seed)
    th.manual_seed(seed)

    return env


def load_model(model_path: str, device: str) -> th.nn.Module:
    return th.load(model_path).to(device)


def evaluate_loop(config: dict, run_name: str = None, seed: int = 0):
    if run_name is None:
        run_name = wandb.run.name

    model_path = os.path.join(
        os.path.join(config["train"]["log_dir"], run_name),
        "checkpoints/ckpt_best.pth",
    )

    env = initialize_environment(config, seed)
    print("Observation space", env.observation_space)
    print("Action space", env.action_space)

    checkpoint = th.load(model_path)
    obs_dim = checkpoint["input_dim"]
    act_dim = checkpoint["output_dim"]

    policy = POLICY_MAP[config["train"]["model"]](obs_dim, act_dim).to(device)
    policy.load_state_dict(checkpoint["model_state_dict"])

    from src.utils.util import evaluate_policy

    success_rate = evaluate_policy(
        env, policy, config, num_episodes=50, render=config["evaluate"]["render"]
    )
    print("Success rate", success_rate)

    env.close()
    return success_rate


def evaluate(config: dict, run_name: str = None):
    success_rates = []
    for seed in config["evaluate"]["seeds"]:
        success_rate = evaluate_loop(config, run_name, seed)
        success_rates.append(success_rate)

    mean_success_rate = np.mean(success_rates)
    std_success_rate = np.std(success_rates)
    print("Success rate over seeds", mean_success_rate)
    wandb.log({"mean_success_rate": mean_success_rate})
    wandb.log({"std_success_rate": std_success_rate})


if __name__ == "__main__":
    with open("params.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Parse args for run name
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default=None)
    args = parser.parse_args()

    evaluate(config, args.run_name)
