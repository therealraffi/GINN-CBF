#!/usr/bin/env python3
import os
import csv
import subprocess
import argparse
from datetime import datetime

import numpy as np
import CalSim as cs
import trimesh
from controllers.force_controllers import QRotorGeometricPD
from trajectories import TrajectoryManager

from simulate_pair import points_to_obstacles, downsample_point_cloud_random

NUM_PTS = 10_000

def main():
    p = argparse.ArgumentParser(
        description="Submit one SLURM job per start/goal in the CSV"
    )
    p.add_argument('--csv',       required=True,
                   help="path to start/goal CSV")
    p.add_argument('--mesh',      required=True,
                   help="path to mesh.ply")
    p.add_argument('--output-dir',required=True,
                   help="where to collect all trajectory_*.txt")
    p.add_argument('--cpus',      type=int, default=4,
                   help="CPUs per SLURM job")
    p.add_argument('--mem',       default='32G',
                   help="RAM per job (e.g. 32G)")
    p.add_argument('--time',      default='02:00:00',
                   help="walltime per job (HH:MM:SS)")
    args = p.parse_args()

    # prepare
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir   = os.path.join("logs",   timestamp)
    os.makedirs(log_dir,   exist_ok=True)

    # create timestamped sub-dir for outputs
    out_base  = args.output_dir
    out_dir   = os.path.join(out_base, timestamp)
    os.makedirs(out_dir,   exist_ok=True)

    ###############################################################

    mesh      = trimesh.load_mesh(args.mesh)  # obstacles from mesh
    coord     = np.array(mesh.vertices)
    coord = downsample_point_cloud_random(coord, NUM_PTS)
    start = coord.mean(axis=0)

    pos0   = np.array([start]).T
    vel0   = np.zeros((3,1))
    omega0 = np.zeros((3,1))
    R0     = np.eye(3).reshape((9,1))
    x0     = np.vstack((pos0, R0, omega0, vel0))

    dynamics        = cs.Qrotor3D(x0)
    observerManager = cs.ObserverManager(dynamics)

    obstacleM = points_to_obstacles(coord, radius=0.1)

    depthM = cs.DepthCamManager(observerManager, obstacleM, mean=None, sd=None)
    
    point_cloud = np.transpose(depthM.get_depth_cam_i(0).calc_ptcloud_world())
    np.save(os.path.join(out_dir, "world_pointcloud.npy"), point_cloud)


    ###############################################################

    # read CSV
    with open(args.csv, newline='', encoding='latin-1') as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            start = [row['start_x'], row['start_y'], row['start_z']]
            goal  = [row['goal_x'],  row['goal_y'],  row['goal_z']]
            test_start = start

            job_name = f"qrotor_{idx}"
            out_log  = f"{log_dir}/{job_name}.out"
            err_log  = f"{log_dir}/{job_name}.err"

            wrap_cmd = "python -u simulate_pair.py " + " ".join([
                f"--start {' '.join(start)}",
                f"--goal {' '.join(goal)}",
                f"--mesh {args.mesh}",
                f"--job-index {idx}",
                f"--output-dir {out_dir}",
                f"--num-pts {NUM_PTS}"
            ])

            cmd = [
                'sbatch',
                f'--account=cral',
                '--partition=standard',
                f'--cpus-per-task={args.cpus}',
                f'--mem={args.mem}',
                f'--time={args.time}',
                f'--job-name={job_name}',
                f'--output={out_log}',
                f'--error={err_log}',
                # single arg with =, no space!
                f'--wrap={wrap_cmd}'
            ]            

            print("Submitting:", " ".join(cmd), "\n")
            subprocess.run(cmd, check=True)



if __name__ == "__main__":
    main()

'''

./submit_jobs.py \
  --csv        /scratch/rhm4nj/cral/cral-ginn/ginn/goal_csvs/room_0_start_goal.csv \
  --mesh       /scratch/rhm4nj/cral/datasets/Replica-Dataset/ReplicaSDK/room_0/mesh.ply \
  --output-dir /scratch/rhm4nj/cral/cral-ginn/depth_cbf/qrotor_sim_examples/outs/room_0 \
  --cpus       1 \
  --mem        32G \
  --time       01:00:00

'''