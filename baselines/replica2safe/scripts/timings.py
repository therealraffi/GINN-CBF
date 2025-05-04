import json
import numpy as np

# === CONFIGURATION ===
scene_name = 'room_0'
method = 'ball-to-ellipsoid'
traj_file = f'../safer-splat/trajs/mix/{scene_name}_{method}.json'

# === Load Data
with open(traj_file, 'r') as f:
    data = json.load(f)

# === Lists to collect metrics ===
step_times = []
trial_times = []
cbf_times = []
qp_times = []
prune_times = []

traj_lengths = []
traj_lengths_success = []
traj_lengths_fail = []

trajectories_with_unsafe = 0
total_trajectories = 0
success_count = 0
fail_count = 0

for i, trial in enumerate(data['total_data']):
    total_trajectories += 1

    # Timing metrics
    if trial['total_time']:
        step_times.append(np.mean(trial['total_time']))
        trial_times.append(np.sum(trial['total_time']))

    if trial['cbf_solve_time']:
        cbf_times.append(np.mean(trial['cbf_solve_time']))

    if trial['qp_solve_time']:
        qp_times.append(np.mean(trial['qp_solve_time']))

    if trial['prune_time']:
        prune_times.append(np.mean(trial['prune_time']))

    # Trajectory length
    if trial['traj']:
        traj_len = len(trial['traj'])
        traj_lengths.append(traj_len)

    # Safety analysis
    safety_vals = np.array(trial['safety'], dtype=np.float32)
    num_unsafe = np.sum(safety_vals < 0)
    if num_unsafe > 0:
        trajectories_with_unsafe += 1

    # Success/fail (note typo in JSON: "sucess")
    success = trial['sucess'][0]
    if success:
        success_count += 1
        traj_lengths_success.append(traj_len)
    else:
        fail_count += 1
        traj_lengths_fail.append(traj_len)

# === Summary Output ===
print(f"\n--- {scene_name} Timing & Safety Summary ---")
print(f"Avg step time across all trials:     {np.mean(step_times):.6f} s")
print(f"Avg total time per trial:            {np.mean(trial_times):.2f} s")
print(f"Avg CBF call time per step:          {np.mean(cbf_times):.6f} s")
print(f"Avg QP solve time per step:          {np.mean(qp_times):.6f} s")
print(f"Avg prune time per step:             {np.mean(prune_times):.6f} s")

print(f"\n--- Trajectory Safety & Success ---")
print(f"Total trajectories:                  {total_trajectories}")
print(f"Trajectories with unsafe moments:    {trajectories_with_unsafe} / {total_trajectories}")
print(f"Successes:                           {success_count}")
print(f"Failures:                            {fail_count}")

if traj_lengths:
    print(f"\n--- Trajectory Lengths ---")
    print(f"Average trajectory length (all):     {np.mean(traj_lengths):.1f} steps")
    if traj_lengths_success:
        print(f"Average length (successes):          {np.mean(traj_lengths_success):.1f} steps")
    if traj_lengths_fail:
        print(f"Average length (failures):           {np.mean(traj_lengths_fail):.1f} steps")
