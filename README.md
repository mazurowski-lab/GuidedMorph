# GuidedMorph: Two-Stage Deformable Registration for Breast MRI

**By [Yaqian Chen](https://scholar.google.com/citations?user=iegKFuQAAAAJ&hl=en), [Hanxue Gu](https://scholar.google.com/citations?user=aGjCpQUAAAAJ&hl=en&oi=ao), [Haoyu Dong](https://scholar.google.com/citations?user=eZVEUCIAAAAJ&hl=en&oi=ao), [Qihang Li](https://scholar.google.com/citations?user=Yw9_kMQAAAAJ&hl=en), [Yuwen Chen](https://scholar.google.com/citations?user=61s49p0AAAAJ&hl=en&oi=ao), [Nicholas Konz](https://scholar.google.com/citations?user=a9rXidMAAAAJ&hl=en), [Lin Li](https://scholar.google.com/citations?user=uRHrZUkAAAAJ&hl=zh-CN&authuser=1) and [Maciej Mazurowski](https://scholar.google.com/citations?user=HlxjJPQAAAAJ&hl=en&oi=ao)**
---
[![arXiv](https://img.shields.io/badge/arXiv-2502.09779-b31b1b.svg)](https://arxiv.org/abs/2505.13414)

Please check the [Google Drive](https://drive.google.com/drive/folders/1OZtwY_XnlwbKCATCZiuiFSLwuUW3XZnM?usp=sharing) for Guided-Trans-DT weights and the ISPY2 external test dataset.


## To-do

- [ ] update arxiv
- [ ] add feature extraction gpu version


This is the official code for our paper:  
**GuidedMorph: Two-Stage Deformable Registration for Breast MRI**
![Screenshot 2025-06-16 at 3 31 14 PM](https://github.com/user-attachments/assets/d638f384-269b-4a1f-8151-e7b2b92735b7)

---

## Table of Contents

- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Configuration](#configuration)
- [Training](#training)
- [Inference & Evaluation](#inference--evaluation)
- [File Structure](#file-structure)
- [Citation](#citation)
- [License](#license)

---

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/GuidedMorph.git
   cd GuidedMorph
   ```

2. **Install dependencies**
   - Recommended: Use [conda](https://docs.conda.io/en/latest/) for environment management.
   - Create environment:
     ```bash
     conda env create -f environment.yml
     conda activate guidedmorph
     ```
   - Or manually install requirements:
     ```bash
     pip install torch torchvision numpy matplotlib natsort
     ```

---

## Data Preparation

- **Input Format:** All data should be preprocessed and saved as `.pkl` files, where each file contains:
  - `x` (moving image), `y` (fixed image)
  - Followed by pairs of segmentation masks: `x_seg1, y_seg1, x_seg2, y_seg2, ...` (for each label)
- **Example directory structure:**
  ```
  demo/
    train/
      case1.pkl
      case2.pkl
      ...
    test/
      case1.pkl
      case2.pkl
      ...
  ```

- **Data loader will automatically handle any number of label pairs per case.**

---

## Configuration

All parameters are managed via JSON config files:

- **Training:** `config.json`
- **Inference:** `infer_config.json`

**Example (`config.json`):**
```json
{
  "GPU_iden": 0,
  "batch_size": 1,
  "train_dir": "demo/train/",
  "save_frequency": 2,
  "lr": 0.0005,
  "epoch_start": 0,
  "max_epoch": 15000,
  "img_size": [128, 256, 256],
  "cont_training": false,
  "weights": [1.0, 1.0, 0.08],
  "architecture": "UNet_Cbam_STN"
}
```

**Example (`infer_config.json`):**
```json
{
  "test_dir": "demo/test/",
  "img_size": [128, 256, 256],
  "weights": [1, 1, 0.06],
  "model_idx": -1,
  "model_type": "VxmDense_2",
  "model_folder_template": "vxm_2_mse_{0}_diffusion_{1}_{2}_2/"
}
```

---

## Training

To train the model:

```bash
python train_vxm.py
```

- All training parameters are controlled by `config.json`.
- Checkpoints and logs will be saved in the `experiments/` directory.

---

## Inference & Evaluation

To run inference and evaluate Dice for each label:

```bash
python infer.py
```

- All inference parameters are controlled by `infer_config.json`.
- The script will automatically compute Dice for each label in every test case and print the mean and standard deviation.

---

## File Structure

```
GuidedMorph/
├── data/
│   ├── data_utils.py
│   ├── datasets.py
│   └── ...
├── edge.py
├── feature_extract.py
├── infer.py
├── infer_config.json
├── losses.py
├── models.py
├── train_vxm.py
├── config.json
├── utils.py
├── experiments/
│   └── ... (checkpoints, logs)
├── demo/
│   ├── train/
│   └── test/
└── README.md
```

---

## Citation

If you use this code or our method in your research, please cite:

```bibtex
@article{chen2024guidedmorph,
  title={GuidedMorph: Two-Stage Deformable Registration for Breast MRI},
  author={Chen, Yaqian and Gu, Hanxue and Dong, Haoyu and Li, Qihang and Chen, Yuwen and Konz, Nicholas and Li, Lin and Mazurowski, Maciej},
  journal={arXiv preprint arXiv:2505.13414},
  year={2024}
}
```

---

## Contact

For questions or collaborations, please contact [Yaqian Chen](mailto:yaqian.chen@duke.edu) or open an issue on GitHub.
