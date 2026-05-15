# OFC Planning Dynamics

Code for Hu and Wallis (2026).

---

## Installation

### Requirements

- Python 3.9 or higher
- [conda](https://docs.conda.io/en/latest/miniconda.html) (recommended) or a standard Python virtual environment

---

### 1. Clone the repository

```bash
git clone <repo-url>
cd PlanningDynamics
```

---

### 2. Create and activate a conda environment

```bash
conda create -n PlanningDynamics python=3.9
conda activate PlanningDynamics
```

Or with `venv`:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

---

### 3. Install PyTorch
PyTorch should be installed separately before the other packages so that the
correct build (CPU-only or CUDA) is selected for your hardware.

**CPU only:**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**CUDA 12.x (GPU):**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

See [pytorch.org/get-started](https://pytorch.org/get-started/locally/) for
other configurations.

---

### 4. Install the package and all dependencies

```bash
pip install -e .
```

This installs `PlanningDynamics` in editable mode along with all required
packages declared in `pyproject.toml`:

- `numpy`
- `pandas`
- `scipy`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `networkx`
- `pynwb`
- `pingouin`
- `tqdm`
- `torch`

---

### 5. Install Jupyter (for figures)

```bash
pip install jupyter
```

Then open any notebook under `figures/`:

```bash
jupyter notebook figures/fig1.ipynb
```

---

### 6. Data
To get data, please reach out to wallis@berkeley.edu. Raw NWB files should be placed at:

```
data/
  bart/
  london/
```

The path is configured in `PlanningDynamics/utils.py` (`get_filenames`).