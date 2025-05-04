#!/usr/bin/env python3
import os
import time
import argparse
import numpy as np
import CalSim as cs
import trimesh
from controllers.force_controllers import QRotorGeometricPD
from trajectories import TrajectoryManager

MAX_POINT_SIZE = 5_000

def points_to_obstacles(point_cloud_arr, radius):
    qObs   = point_cloud_arr.T
    rObs   = [radius] * point_cloud_arr.shape[0]
    numObs = point_cloud_arr.shape[0]
    return cs.ObstacleManager(qObs, rObs, NumObs=numObs)

def downsample_point_cloud_random(point_cloud, num_points):
    if point_cloud.shape[0] <= num_points:
        return point_cloud 

    indices = np.random.choice(point_cloud.shape[0], num_points, replace=False)
    return point_cloud[indices]

def main():
    print("STARTING!")

    p = argparse.ArgumentParser(
        description="Run one qrotor sim for a given start/goal pair"
    )
    p.add_argument('--start',      nargs=3, type=float, required=True,
                   help="start_x start_y start_z")
    p.add_argument('--goal',       nargs=3, type=float, required=True,
                   help="goal_x  goal_y  goal_z")
    p.add_argument('--mesh',       type=str,   required=True,
                   help="path to mesh.ply")
    p.add_argument('--job-index',  type=int,   required=True,
                   help="integer index for naming output")
    p.add_argument('--output-dir', type=str,   required=True,
                   help="where to write trajectory_<index>.txt")
    p.add_argument('--num-pts', type=str,   required=True,
                help="max number of pts to sample from")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    start = args.start
    goal  = args.goal
    mesh_file = args.mesh
    idx   = args.job_index
    outd  = args.output_dir
    num_pts  = int(args.num_pts)

    t0 = time.time()

    pos0   = np.array([start]).T
    vel0   = np.zeros((3,1))
    omega0 = np.zeros((3,1))
    R0     = np.eye(3).reshape((9,1))
    x0     = np.vstack((pos0, R0, omega0, vel0))

    dynamics        = cs.Qrotor3D(x0)
    observerManager = cs.ObserverManager(dynamics)

   
    mesh      = trimesh.load_mesh(mesh_file)  # obstacles from mesh
    coord     = np.array(mesh.vertices)
    coord = downsample_point_cloud_random(coord, num_pts)
    obstacleM = points_to_obstacles(coord, radius=0.1)


    depthM = cs.DepthCamManager(observerManager, obstacleM, mean=None, sd=None)
    point_cloud = np.transpose(depthM.get_depth_cam_i(0).calc_ptcloud_world())

    xD = np.vstack((np.array([goal]).T, R0, omega0, vel0))
    trajM = TrajectoryManager(x0, xD, Ts=5, N=1)

    ctrlM = cs.ControllerManager(observerManager,
                                 QRotorGeometricPD,
                                 None,
                                 trajM,
                                 depthM)
    env = cs.Environment(dynamics,
                         ctrlM,
                         observerManager,
                         obstacleM,
                         T=10)
    env.reset()

    print("BEGINING RUN...")
    anim = env.run(verbose=True)
    t1 = time.time()

    # extract and save
    trajectory = anim[0][0:3, :].T
    out_file = os.path.join(outd, f"trajectory_{idx:04d}.txt")
    header   = f"Execution time: {t1-t0:.4f} sec"
    np.savetxt(out_file, trajectory, delimiter=",", header=header)

if __name__ == "__main__":
    main()