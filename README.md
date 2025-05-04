# GINN-CBF
Ensuring safe autonomous navigation in 3D environments reconstructed from vision remains a core challenge, especially when relying on learned scene geometry. We propose \textbf{GINN-CBF}, a method that integrates Geometry-Informed Neural Networks (GINNs) with Control Barrier Functions (CBFs) to enforce safety in real-time over signed distance fields (SDFs). Each GINN is a sine-activated implicit network trained on object-level point cloud clusters with a novel forward-invariance loss, allowing SDFs to directly encode control-theoretic safety. A differentiable masking strategy composes modular SDFs into a global field, enabling fast, query-time-safe control. To guarantee robustness under approximation error, we derive a Lipschitz-based bound and incorporate it into a conservatively approximated quadratic program (QP) that enforces forward invariance in real-time. We evaluate GINN-CBF on Replica indoor scenes under static and dynamic conditions. Compared to DepthCBF, NeRF-CBF, and SaferSplat, our method achieves perfect safety and significantly higher goal success across all test scenarios. GINN-CBF scales to complex, vision-derived scenes without online retraining, offering a flexible pipeline for learning-based safe control in real-world environments.

This is a copy of my GINN-CBF folder - re-structured folders and deleted or moved redundant ones.

## Table of Contents
- [ GINN-CBF training](#Training)
- [Experiments](#features)

## Training
Runs the modified `ginn` training code, with my CBF changes. Graphs to a WandDB terminal. Example usage:

#### GINN-CBF training

```bash
python run.py \
  --gpu_list 0 \
  --yml config_3dis.yml \
  --no_save True \
  --hp_dict "lambda_bound:0;interface_delta:0;dataset_dir:/scratch/rhm4nj/cral/cral-ginn/ginn/myvis/data_gen/S3D/Area_1/0_ceiling;model_save_path:/scratch/rhm4nj/cral/cral-ginn/ginn/all_runs/models/experiments/2025-02-21_08-51-11_Area_1/_0_ceiling;lambda_descent:0.0001"
```

| Argument      | Description                                                                                                                        |
|---------------|------------------------------------------------------------------------------------------------------------------------------------|
| `--gpu_list`  | Comma-separated list of GPU IDs to use (e.g. `0` for GPU 0).                                                                       |
| `--yml`       | Path to the YAML config file defining dataset paths, model settings, and other defaults.                                          |
| `--no_save`   | Skip saving outputs when set to `True`                                                    |
| `--hp_dict`   | Semicolon-delimited `key:value` pairs to override default hyperparameters:<br>- `interface_delta`: target value at surface (interface) of object (here `0`)<br>- `dataset_dir`: full path to the data directory used for training<br>- `model_save_path`: directory where model checkpoints and logs are written<br>- `lambda_descent`: weighting for forwared-invariance (descent) loss (here `0.0001`) |

#### Running in batches
To run mutiple scripts at once, `/scratch/rhm4nj/cral/cral-ginn-copy/slurm_scripts/schedule_ginn.ipynb` - runs individual `run.py` calls over several SLURM nodes, allowing for different hyper-paramters


## Experiments
To run an `GINN-CBF`, run `experiment_base.ipynb`. Includes visualizations at beginning and end of files for point cloud and results. To run multiple concurrent experiments, run `experiments_export.ipynb` which runs experiments across SLURM nodes.