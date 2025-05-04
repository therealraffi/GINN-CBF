# replica2safe

replica -> safersplat

---

### 1. Create Environment

```bash
module load cuda/11.8.0
conda create -n safer-splat python=3.8 -y
conda activate safer-splat
```

---

### 2. Install Dependencies

```bash
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
conda install -c nvidia/label/cuda-11.8.0 cuda-toolkit
pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
pip install nerfstudio clarabel
```

---

### 3. Clone Repo & Get Data

```bash
git clone https://github.com/chengine/safer-splat.git
mkdir -p data && cd data
wget https://huggingface.co/datasets/kxic/vMAP/resolve/main/vmap.zip
unzip vmap.zip && mv vmap/* replica_v2/
```

---

### 4. Process & Train

**4.1. Prepare COLMAP input**

Use `replica2colmap.py` to convert Replica-format folders into COLMAP-ready structures:
```bash
python replica2colmap.py
```

**4.2. Process with Nerfstudio**

Run `ns-process-data` to convert the COLMAP outputs into Nerfstudio-compatible format:
```bash
ns-process-data images \
  --data data/colmap/office_0 \
  --output-dir data/nerfstudio/office_0 \
  --sfm-tool colmap \
  --no-gpu
```

**4.3. Train with Splatfacto**

Train the Gaussian Splatting pipeline on the processed data:
```bash
ns-train splatfacto \
  --data data/nerfstudio/office_0 \
  --output-dir outputs/office_0
```

---

### 5. Align & Run

- Align scene manually using `dim.ipynb`
- Update `coords.py` with alignment values
- Add a path CSV at `/path/office_0.csv`

```bash
cd safer-splat
python run.py
```

---

Note: Be sure to update file paths and scene names in the Python scripts to match your own directory structure
