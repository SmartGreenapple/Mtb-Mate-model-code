# Mtb-Mate Model Code

This repository contains the model implementation and training scripts for **Mtb-Mate**, an AI-based platform for anti-tuberculosis small-molecule activity prediction.

Mtb-Mate integrates graph-based deep learning, molecular fingerprints and descriptors, traditional machine learning models, ensemble prediction, and uncertainty estimation to support anti-tuberculosis compound screening.

## Overview

The code supports two major prediction tasks:

1. **Phenotypic anti-tuberculosis activity prediction**
   Regression prediction of pMIC values for small molecules against *Mycobacterium tuberculosis*.

2. **Target-specific activity prediction**
   Binary classification of small-molecule activity against tuberculosis-related protein targets.

The final prediction framework combines graph neural network outputs with traditional machine learning models to generate ensemble predictions.

## Repository structure

* `scripts/train.py`: Main training and evaluation script.
* `scripts/build_dataset.py`: Dataset construction script.
* `scripts/featurization.py`: Molecular feature construction.
* `scripts/model.py`: Model architecture definition.

## Main features

* Molecular graph construction from SMILES strings.
* Molecular fingerprint and descriptor-based feature extraction.
* Multi-task graph neural network modeling.
* pMIC regression for phenotypic anti-tuberculosis activity prediction.
* Target-specific binary classification.
* Ensemble prediction using deep learning and machine learning models.
* Cross-validation-based model evaluation.
* Output of regression and classification performance metrics.

## Model training

Before running the training script, please check and modify the paths in `scripts/train.py`, especially `ROOT_DIR` and `OUT_DIR`.

Example command:

`python scripts/train.py`

The training script performs five-fold cross-validation and saves model checkpoints, machine learning models, fold indices, and summary metric tables.

## Output files

The training process generates files such as:

* `best_model_fold_*.pt`
* `rf_fold_*.pkl`
* `xgb_*_fold_*.pkl`
* `global_mskf_indices.joblib`
* `final_avg_regression.csv`
* `final_avg_classification.csv`

These files are saved in the output directory defined by `OUT_DIR`.

## Model checkpoints

Large trained checkpoint files are not included directly in this repository due to GitHub file size limitations.

The checkpoint archive is provided via GitHub Releases. After downloading the checkpoint archive, extract it using:

`tar -xzf results_5fold_nopro_20260510_for_release.tar.gz`

## Web platform

The online Mtb-Mate prediction platform is available at:

https://lmmd.ecust.edu.cn/mtb-mate/

The web platform supports SMILES-based single-molecule and batch prediction and provides ensemble pMIC predictions, target-specific activity probabilities, and uncertainty estimates.

## Dependencies

The code was developed using Python 3.9. Major dependencies include:

* `torch`
* `torch-geometric`
* `rdkit`
* `numpy`
* `pandas`
* `scikit-learn`
* `xgboost`
* `scipy`
* `joblib`

Please install compatible versions of PyTorch and PyTorch Geometric according to your CUDA environment.

## Data availability

The cleaned datasets and trained model checkpoints may be provided separately depending on data-sharing restrictions and file size limitations.

## Citation

If you use this code or the Mtb-Mate platform, please cite the corresponding manuscript.

