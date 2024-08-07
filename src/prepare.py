import h5py
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch as th
import gymnasium as gym
from ManiSkill.mani_skill.envs.sapien_env import BaseEnv
from ManiSkill.mani_skill.utils.structs.pose import vectorize_pose
from ManiSkill.mani_skill.utils.io_utils import load_json
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib
import yaml
from src.utils.util import compute_joint_se3_pose
from typing import Tuple

from pathlib import Path


def load_h5_data(data):
    out = {}
    for k, v in data.items():
        if isinstance(v, h5py.Dataset):
            out[k] = v[:]
        else:
            out[k] = load_h5_data(v)
    return out


def standardize(data: np.ndarray) -> np.ndarray:
    # Try to load the standard scaler from the disk
    try:
        scaler = joblib.load("standard_scaler.pkl")
    except FileNotFoundError:
        scaler = None

    if scaler is None:
        scaler = StandardScaler()
        data = scaler.fit_transform(data)
        joblib.dump(scaler, "standard_scaler.pkl")
    else:
        data = scaler.transform(data)
    return data


def normalize(data: np.ndarray, scaler: MinMaxScaler = None) -> np.ndarray:
    if scaler is None:
        scaler = MinMaxScaler(feature_range=(-1, 1), clip=True)
        data = scaler.fit_transform(data)
        joblib.dump(scaler, "norm_scaler.pkl")
    else:
        data = scaler.transform(data)
    return data


def fourier_encode(data: np.ndarray) -> np.ndarray:
    scales = np.array([-1, 0, 1, 2, 3, 4, 5, 6], dtype=data.dtype)
    scaled_data = data.reshape(-1, 1) / (3.0**scales).reshape(1, -1)
    sin_features = np.sin(scaled_data)
    cos_features = np.cos(scaled_data)
    features = np.concatenate([sin_features, cos_features], axis=1)
    return features.reshape(data.shape[0], -1)


TRANSFORMATIONS = {
    "standardize": standardize,
    "normalize": normalize,
    "fourier": fourier_encode,
}


def load_raw_data(config: dict):
    dataset_file = config["prepare"]["raw_data_path"] + config["prepare"]["data_file"]
    data = h5py.File(dataset_file, "r")
    json_path = dataset_file.replace(".h5", ".json")
    json_data = load_json(json_path)
    episodes = json_data["episodes"]

    observations, actions, episode_map = [], [], []
    load_count = (
        len(episodes)
        if config["prepare"]["load_count"] == -1
        else config["prepare"]["load_count"]
    )
    for eps_id in tqdm(range(load_count)):
        eps = episodes[eps_id]
        trajectory = load_h5_data(data[f"traj_{eps['episode_id']}"])
        observations.append(trajectory["obs"][:-1])
        actions.append(trajectory["actions"])
        episode_map.append(np.full(len(trajectory["obs"]) - 1, eps["episode_id"]))

    observations = np.vstack(observations)
    actions = np.vstack(actions)
    episode_map = np.hstack(episode_map)

    return observations, actions, episode_map


def get_pick_cube_features(
    env: BaseEnv, obs: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    joint_positions = obs[:, :9]
    joint_velocities = obs[:, 9:18]
    is_grasped = obs[:, 18:19]
    tcp_pose = obs[:, 19:26]
    goal_position = obs[:, 26:29]
    obj_pose = obs[:, 29:36]
    tcp_to_obj_pos = obs[:, 36:39]
    obj_to_goal_pos = obs[:, 39:42]

    joint_se3_pose = compute_joint_se3_pose(env=env, joint_positions=joint_positions)
    joint_se3_reshape = joint_se3_pose.reshape(joint_se3_pose.shape[0], 8, -1)

    joint_features = np.concatenate(
        (
            joint_positions[:, :-1, None],
            joint_velocities[:, :-1, None],
            joint_se3_reshape,
        ),
        axis=2,
    )

    context_info = np.hstack(
        [
            is_grasped,
            tcp_pose,
            goal_position,
            obj_pose,
            tcp_to_obj_pos,
            obj_to_goal_pos,
        ]
    )

    context_features = np.repeat(
        context_info[:, np.newaxis, :], joint_features.shape[1], axis=1
    )

    return joint_features, context_features


def get_stack_cube_features(
    env: BaseEnv, obs: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    joint_positions = obs[:, :9]
    joint_velocities = obs[:, 9:18]
    tcp_pose = obs[:, 18:25]
    cubeA_pose = obs[:, 25:32]
    cubeB_pose = obs[:, 32:39]
    tcp_to_cubeA_pos = obs[:, 39:42]
    tcp_to_cubeB_pos = obs[:, 42:45]
    cubeA_to_cubeB_pos = obs[:, 45:48]

    joint_se3_pose = compute_joint_se3_pose(env=env, joint_positions=joint_positions)
    joint_se3_reshape = joint_se3_pose.reshape(joint_se3_pose.shape[0], 8, -1)

    # add small gaussian noise to gripper position to allow for diagonal grasp.
    joint_positions[:, -1] += np.random.normal(0, 0.01, joint_positions.shape[0])

    joint_features = np.concatenate(
        (
            joint_positions[:, :-1, None],
            joint_velocities[:, :-1, None],
            joint_se3_reshape,
        ),
        axis=2,
    )

    context_info = np.hstack(
        [
            tcp_pose,
            cubeA_pose,
            cubeB_pose,
            tcp_to_cubeA_pos,
            tcp_to_cubeB_pos,
            cubeA_to_cubeB_pos,
        ]
    )

    context_features = np.repeat(
        context_info[:, np.newaxis, :], joint_features.shape[1], axis=1
    )

    return joint_features, context_features


def graph_transform_obs(obs, env):
    joint_features, context_features = get_pick_cube_features(env=env, obs=obs)

    combined_features = np.concatenate([joint_features, context_features], axis=2)
    original_shape = combined_features.shape
    combined_features = combined_features.reshape(-1, combined_features.shape[-1])

    return combined_features, original_shape


def apply_transformations(array, config: dict):
    for transformation in config["prepare"]["transformations"]:
        t_type = transformation["type"]
        transform_func = TRANSFORMATIONS[t_type]
        array = transform_func(array)
    return array


def get_mlp_features(env: BaseEnv, obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    joint_positions = obs[:, :9]
    joint_velocities = obs[:, 9:18]
    is_grasped = obs[:, 18:19]
    tcp_pose = obs[:, 19:26]
    goal_position = obs[:, 26:29]
    obj_pose = obs[:, 29:36]
    tcp_to_obj_pos = obs[:, 36:39]
    obj_to_goal_pos = obs[:, 39:42]

    obs = np.hstack(
        [
            joint_positions,
            joint_velocities,
            is_grasped,
            tcp_pose,
            goal_position,
            obj_pose,
            tcp_to_obj_pos,
            obj_to_goal_pos,
        ]
    )
    return obs, obs[:, np.newaxis].shape


def prepare(config: dict):
    env: BaseEnv = gym.make(
        id=config["env"]["env_id"],
        obs_mode=config["env"]["obs_mode"],
        control_mode=config["env"]["control_mode"],
        render_mode=config["env"]["render_mode"],
    )
    obs, act, episode_map = load_raw_data(config)

    if config["train"]["model"] != "MLP":
        obs, obs_shape = graph_transform_obs(obs, env)
    else:
        obs, obs_shape = get_mlp_features(env, obs)
    obs = apply_transformations(obs, config).reshape(obs_shape)

    # Create a directory to save the prepared data
    Path(config["prepare"]["prepared_data_path"]).mkdir(parents=True, exist_ok=True)

    np.save(config["prepare"]["prepared_data_path"] + "obs.npy", obs)
    np.save(config["prepare"]["prepared_data_path"] + "act.npy", act)
    np.save(
        config["prepare"]["prepared_data_path"] + "episode_map.npy",
        episode_map,
    )


with open("params.yaml", "r") as f:
    config = yaml.safe_load(f)

if __name__ == "__main__":
    prepare(config)
