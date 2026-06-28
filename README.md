# Mtb-Mate Model Code

This repository provides the model implementation and training scripts for **Mtb-Mate**, an AI-based platform for anti-tuberculosis small-molecule activity prediction.

Mtb-Mate integrates graph-based deep learning, molecular fingerprints/descriptors, traditional machine learning models, ensemble prediction, and uncertainty estimation to support anti-tuberculosis compound screening.

## Overview

The code supports two prediction tasks:

1. **Phenotypic anti-tuberculosis activity prediction**
   Regression prediction of pMIC values for small molecules against *Mycobacterium tuberculosis*.

2. **Target-specific activity prediction**
   Binary classification of small-molecule activity against tuberculosis-related protein targets.

The final prediction framework combines graph neural network predictions with traditional machine learning models to generate ensemble outputs.

## Repository Structure

```text
Mtb-Mate-model-code/
├── scripts/
│   ├── train.py              # Main training and evaluation script
│   ├── build_dataset.py      # Dataset construction script
│   ├── featurization.py      # Molecular feature construction
│   └── model.py              # Model architecture
├── environment.yml           # Conda environment file
├── .gitignore
└── README.md
```

## Main Functions

* Construct molecular graphs from SMILES strings.
* Generate molecular fingerprints and physicochemical descriptors.
* Train a multi-task graph neural network model.
* Perform pMIC regression for phenotypic anti-tuberculosis activity prediction.
* Perform target-specific binary classification.
* Integrate deep learning and machine learning predictions through ensemble modeling.
* Evaluate model performance using cross-validation.

## Installation

The code was developed using Python 3.9. The environment can be created using:

```bash
conda env create -f environment.yml
conda activate mtb_dl
```

Major dependencies include:

* PyTorch
* PyTorch Geometric
* RDKit
* NumPy
* Pandas
* Scikit-learn
* XGBoost
* SciPy
* Joblib

Please install PyTorch and PyTorch Geometric versions compatible with your CUDA environment if needed.

## Model Training

Before running the training script, please modify the paths in `scripts/train.py`, especially:

```python
config = {
    "ROOT_DIR": "/path/to/your/project",
    "OUT_DIR": "results_5fold",
}
```

Then run:

```bash
python scripts/train.py
```

The script performs five-fold cross-validation and saves trained model checkpoints, traditional machine learning models, fold indices, and summary metric tables.

## Output Files

The training process generates files such as:

```text
best_model_fold_*.pt
rf_fold_*.pkl
xgb_*_fold_*.pkl
global_mskf_indices.joblib
final_avg_regression.csv
final_avg_classification.csv
```

These files are saved in the output directory defined by `OUT_DIR`.

## Model Checkpoints

Large trained checkpoint files are not stored directly in this repository due to file size limitations.

The checkpoint archive is provided through **GitHub Releases**. After downloading the archive, extract it using:

```bash
tar -xzf results_5fold_20260510_for_release.tar.gz
```

## Web Platform

The online Mtb-Mate prediction platform is available at:

https://lmmd.ecust.edu.cn/mtb-mate/

The web platform supports SMILES-based single-molecule and batch prediction and provides ensemble pMIC predictions, target-specific activity probabilities, and uncertainty estimates.

## Data Availability

The cleaned datasets and trained model checkpoints may be provided separately depending on data-sharing restrictions and file size limitations.

## Citation

If you use this code or the Mtb-Mate platform, please cite the corresponding manuscript.

