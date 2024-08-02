import sys

import mani_skill2.envs
import os
import numpy as np
import gymnasium as gym
from src.utils.motion_planner import MotionPlanner
from mani_skill2.agents.base_controller import CombinedController
from mani_skill2.agents.controllers import PDJointPosController
from tqdm import tqdm
from mani_skill2.utils.wrappers import RecordEpisode
from src.utils.aggregate_dataset import merge_hdf5_files, merge_json_files_into_file_a
import multiprocessing

# os.environ["OPENBLAS_NUM_THREADS"] = "1"
# os.environ["DISPLAY"] = ":0"

control_mode = "pd_joint_pos"


def run_trajectory_process(process_id, max_trajectories, output_dir):
    num_trajectories = 0
    vis = False
    env = gym.make(
        "MyEnv-v0",
        obs_mode="state",
        control_mode=control_mode,
        render_mode="human",
    )
    env = RecordEpisode(
        env=env,
        output_dir=output_dir,
        trajectory_name=f"trajectories_{process_id}",
        save_video=False,
    )
    mp = MotionPlanner(env=env)
    progress = tqdm(
        total=max_trajectories, position=process_id, desc=f"Process {process_id}"
    )

    obs, reset_info = env.reset(seed=np.random.randint(1000))
    while num_trajectories < max_trajectories:
        env.reset()
        terminated, truncated = False, False
        plan = mp.move_to_pose_with_screw(env.unwrapped.goal_site.pose)
        if plan["status"] != "success":
            continue
        n = len(plan["position"])

        for i in range(n):
            # Velocity control
            # q_vel_limits = mp.planner.joint_vel_limits
            # q_vel = plan["velocity"][i]
            # action = np.hstack([q_vel, 0.1])

            # Position control
            q_pos = plan["position"][i]
            action = np.hstack([q_pos, 1])

            obs, reward, done, truncated, info = env.step(action)

            prev_qpos = env.agent.robot.get_qpos()
            if vis:
                env.render()

        num_trajectories += 1
        progress.update(1)

    progress.close()
    env.close()


def main():
    max_trajectories = 6000
    num_processes = 4
    trajectories_per_process = max_trajectories // num_processes
    output_dir = "trajectories_output"
    os.makedirs(output_dir, exist_ok=True)

    # Set the start method to 'spawn'
    multiprocessing.set_start_method("spawn")

    processes = []
    for i in range(num_processes):
        process = multiprocessing.Process(
            target=run_trajectory_process,
            args=(i, trajectories_per_process, output_dir),
        )
        processes.append(process)
        process.start()

    for process in processes:
        process.join()

    # List generated HDF5 files
    h5_files = [
        os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith(".h5")
    ]
    print("Generated HDF5 files:", h5_files)

    # Merge HDF5 files and JSON files
    for i in range(1, len(h5_files)):
        merge_hdf5_files(h5_files[0], h5_files[i])
        merge_json_files_into_file_a(
            h5_files[0].replace(".h5", ".json"), h5_files[i].replace(".h5", ".json")
        )

    # Delete all files except the first one
    for i in range(1, len(h5_files)):
        os.remove(h5_files[i])
        os.remove(h5_files[i].replace(".h5", ".json"))

    # Rename the first file
    os.rename(h5_files[0], "trajectories.h5")
    os.rename(h5_files[0].replace(".h5", ".json"), "trajectories.json")


if __name__ == "__main__":
    main()
