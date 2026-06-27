import os
import random

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import RDLogger
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from torch.optim import Adam
from torch_geometric.loader import DataLoader
from xgboost import XGBClassifier

from build_dataset import build_dataset
from model import GAT_MultiTask_V3


RDLogger.DisableLog("rdApp.*")


config = {
    "SEED": 10,
    "ROOT_DIR": "/home/yuntang/data/data_mtb/multi_data",
    "OUT_DIR": "results_5fold",
    "DEVICE": "cuda:7" if torch.cuda.is_available() else "cpu",
    "EPOCHS": 200,
    "BATCH_SIZE": 32,
    "LR": 5e-5,
    "NUM_EXPERTS": 3,
    "N_SPLITS": 5,
    "SELECTED_TARGETS": [
        "DXR",
        "DNA gyrase",
        "InhA",
        "MptpA",
        "MptpB",
        "PknB",
        "TMPK",
        "DprE1",
    ],
}


def set_seed(seed=10):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)


set_seed(config["SEED"])

DEVICE = torch.device(config["DEVICE"])

os.makedirs(config["OUT_DIR"], exist_ok=True)


def get_pos_weight(dataset, num_targets):
    """
    Compute per-target positive-class weights.
    """
    y_targets = torch.stack([d.y_target for d in dataset])
    y_masks = torch.stack([d.y_mask for d in dataset])

    pos_weights = []

    for k in range(num_targets):
        mask = y_masks[:, k].bool()

        pos = (y_targets[mask, k] == 1).sum().float()
        neg = (y_targets[mask, k] == 0).sum().float()

        weight = neg / (pos + 1e-6)
        weight = torch.clamp(weight, min=0.1, max=10.0)

        pos_weights.append(weight)

    pos_weights = torch.stack(pos_weights)

    print("=== pos_weights per target ===")
    print(pos_weights.cpu().numpy())

    return pos_weights.to(DEVICE)


def weighted_focal_loss(logits, targets, masks, pos_weights, gamma=2.0):
    """
    Masked weighted focal loss for multi-target binary classification.
    """
    bce_loss = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="none",
        pos_weight=pos_weights,
    )

    probs = torch.sigmoid(logits)
    p_t = targets * probs + (1 - targets) * (1 - probs)

    focal_weight = (1 - p_t) ** gamma

    loss = focal_weight * bce_loss
    loss = (loss * masks).sum() / (masks.sum() + 1e-6)

    return loss


class AutomaticWeightedLoss(nn.Module):
    """
    Automatic weighted loss for balancing regression and classification tasks.
    """

    def __init__(self, num_tasks):
        super().__init__()
        self.log_vars = nn.Parameter(torch.ones(num_tasks) * 0.5)

    def forward(self, *losses):
        total_loss = 0

        for i, loss in enumerate(losses):
            log_var = torch.clamp(self.log_vars[i], min=-2.0, max=5.0)
            precision = torch.exp(-log_var)

            if i == 0:
                weighted = precision * loss + 0.5 * log_var
            else:
                weighted = precision * loss + log_var

            total_loss += weighted

        return total_loss


class EarlyStopping:
    """
    Early stopping based on:
    score = reg_weight * max(R2, 0) + auprc_weight * mean(AUPRC)
    """

    def __init__(
        self,
        patience=15,
        delta=0,
        reg_weight=0.5,
        auprc_weight=0.5,
    ):
        self.patience = patience
        self.delta = delta
        self.reg_weight = reg_weight
        self.auprc_weight = auprc_weight

        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, r2, auprc_list):
        valid_auprc = [
            res["AUPRC"]
            for res in auprc_list
            if "AUPRC" in res and not np.isnan(res["AUPRC"])
        ]

        mean_auprc = np.mean(valid_auprc) if valid_auprc else 0.0
        r2 = max(0.0, r2)

        score = self.reg_weight * r2 + self.auprc_weight * mean_auprc

        if self.best_score is None:
            self.best_score = score

        elif score < self.best_score + self.delta:
            self.counter += 1

            print(f"[EarlyStopping] No improvement: {self.counter}/{self.patience}")

            if self.counter >= self.patience:
                self.early_stop = True
                print("[EarlyStopping] STOP triggered")

        else:
            self.best_score = score
            self.counter = 0
            print("[EarlyStopping] Improved -> reset counter")

        return score, mean_auprc


def extract_ml_features(loader):
    """
    Extract molecular features and labels from PyG DataLoader
    for traditional ML models.
    """
    all_feats = []
    all_y_mic = []
    all_y_cls = []
    all_m_mic = []
    all_m_cls = []

    for data in loader:
        all_feats.append(data.mol_feature.numpy())
        all_y_mic.append(data.y_mic.numpy())
        all_y_cls.append(data.y_target.numpy())
        all_m_mic.append(data.mic_mask.numpy())
        all_m_cls.append(data.y_mask.numpy())

    return (
        np.concatenate(all_feats),
        np.concatenate(all_y_mic).flatten(),
        np.concatenate(all_y_cls),
        np.concatenate(all_m_mic).flatten(),
        np.concatenate(all_m_cls),
    )


def safe_pearson(y_true, y_pred):
    if len(y_true) < 2:
        return np.nan

    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return np.nan

    return pearsonr(y_true, y_pred)[0]


def safe_spearman(y_true, y_pred):
    if len(y_true) < 2:
        return np.nan

    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return np.nan

    return spearmanr(y_true, y_pred)[0]


def calculate_screening_metrics(y_true, y_pred, top_n=300):
    """
    Calculate Hit@N and EF@N.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if len(y_true) == 0:
        return 0, 0.0

    n = min(top_n, len(y_true))

    if n <= 0:
        return 0, 0.0

    top_indices = np.argsort(y_pred)[::-1][:n]

    true_threshold = np.partition(y_true, -n)[-n]
    true_active_indices = np.where(y_true >= true_threshold)[0]

    hits = len(set(top_indices) & set(true_active_indices))

    total_active_ratio = len(true_active_indices) / len(y_true)
    precision_at_n = hits / n

    ef_at_n = precision_at_n / total_active_ratio if total_active_ratio > 0 else 0.0

    return hits, ef_at_n


def evaluate_fold(model, loader, device, num_targets, idx2target, top_n=300):
    """
    Evaluate a trained model fold.

    Returns:
    1. classification metrics
    2. regression metrics
    3. raw GAT pMIC predictions
    4. raw GAT classification probabilities
    """
    model.eval()

    all_reg_preds = []
    all_reg_trues = []

    all_cls_probs = []
    all_cls_trues = []
    all_cls_masks = []

    all_gat_mic_raw = []
    all_gat_probs_raw = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)

            mic_pred, act_logit, _, _, _ = model(data)

            cls_probs = torch.sigmoid(act_logit)

            all_gat_mic_raw.append(mic_pred.cpu().numpy())
            all_gat_probs_raw.append(cls_probs.cpu().numpy())

            m_reg = data.mic_mask.bool().reshape(-1)

            if m_reg.any():
                all_reg_preds.append(mic_pred[m_reg].cpu())
                all_reg_trues.append(data.y_mic[m_reg].cpu())

            all_cls_probs.append(cls_probs.cpu())
            all_cls_trues.append(data.y_target.cpu())
            all_cls_masks.append(data.y_mask.cpu())

    gat_mic_raw = np.concatenate(all_gat_mic_raw, axis=0).flatten()
    gat_probs_raw = np.concatenate(all_gat_probs_raw, axis=0)

    reg_metrics = {
        "R2": -1.0,
        "RMSE": np.nan,
        "Pearson_r": np.nan,
        "Hit@300": 0,
        "EF@300": 0.0,
    }

    if all_reg_preds:
        y_rp = torch.cat(all_reg_preds).numpy().flatten()
        y_rt = torch.cat(all_reg_trues).numpy().flatten()

        reg_metrics["R2"] = r2_score(y_rt, y_rp)
        reg_metrics["RMSE"] = np.sqrt(mean_squared_error(y_rt, y_rp))
        reg_metrics["Pearson_r"] = safe_pearson(y_rt, y_rp)

        curr_n = min(top_n, len(y_rp))

        hits, ef = calculate_screening_metrics(
            y_rt,
            y_rp,
            top_n=curr_n,
        )

        reg_metrics["Hit@300"] = hits
        reg_metrics["EF@300"] = ef

    y_ct = torch.cat(all_cls_trues).numpy()
    y_cp = torch.cat(all_cls_probs).numpy()
    y_cm = torch.cat(all_cls_masks).numpy()

    cls_results = []

    for k in range(num_targets):
        mask = y_cm[:, k].astype(bool)

        if mask.any() and len(np.unique(y_ct[mask, k])) > 1:
            y_tk = y_ct[mask, k]
            y_pk = y_cp[mask, k]

            best_t = 0.5
            y_pred_cls = (y_pk >= best_t).astype(int)

            cls_results.append(
                {
                    "Target": idx2target[k],
                    "AUROC": roc_auc_score(y_tk, y_pk),
                    "AUPRC": average_precision_score(y_tk, y_pk),
                    "BACC": balanced_accuracy_score(y_tk, y_pred_cls),
                    "Precision": precision_score(
                        y_tk,
                        y_pred_cls,
                        zero_division=0,
                    ),
                    "Recall": recall_score(
                        y_tk,
                        y_pred_cls,
                        zero_division=0,
                    ),
                    "F1": f1_score(
                        y_tk,
                        y_pred_cls,
                        zero_division=0,
                    ),
                    "MCC": matthews_corrcoef(y_tk, y_pred_cls),
                    "Best_T": best_t,
                }
            )

        else:
            cls_results.append(
                {
                    "Target": idx2target[k],
                    "AUROC": np.nan,
                    "AUPRC": np.nan,
                    "BACC": np.nan,
                    "Precision": np.nan,
                    "Recall": np.nan,
                    "F1": np.nan,
                    "MCC": np.nan,
                    "Best_T": 0.5,
                }
            )

    return cls_results, reg_metrics, gat_mic_raw, gat_probs_raw


def find_best_weight(y_true, p_gat, p_ml, task_type="cls"):
    """
    Search the best ensemble weight.

    combined = weight * GAT + (1 - weight) * ML
    """
    ratios = np.linspace(0, 1.0, 21)

    best_r = 0.5
    best_score = -1e9

    for r in ratios:
        combined = r * p_gat + (1 - r) * p_ml

        try:
            if task_type == "cls":
                score = average_precision_score(y_true, combined)

            elif task_type == "reg":
                score = r2_score(y_true, combined)

            else:
                raise ValueError(f"Unsupported task_type: {task_type}")

        except Exception:
            score = -1e9

        if score > best_score:
            best_score = score
            best_r = r

    return best_r


def save_final_reports(cls_metrics, reg_metrics, out_dir):
    """
    Save final cross-validation summary reports.
    """
    df_cls = pd.DataFrame(cls_metrics)

    if not df_cls.empty:
        metrics_cols = [
            c for c in df_cls.columns
            if c not in ["Target", "Best_T"]
        ]

        cls_sum = (
            df_cls
            .groupby("Target")[metrics_cols]
            .agg(["mean", "std"])
            .reset_index()
        )

        cls_fmt = pd.DataFrame({"Target": cls_sum["Target"]})

        for metric in metrics_cols:
            cls_fmt[metric] = cls_sum.apply(
                lambda x: f"{x[(metric, 'mean')]:.4f} ± {x[(metric, 'std')]:.4f}",
                axis=1,
            )

        cls_path = os.path.join(out_dir, "final_avg_classification.csv")
        cls_fmt.to_csv(cls_path, index=False)

        print(f"Classification report saved: {cls_path}")

    df_reg = pd.DataFrame(reg_metrics)

    if not df_reg.empty:
        reg_mean = df_reg.mean(numeric_only=True)
        reg_std = df_reg.std(numeric_only=True)

        reg_sum = pd.DataFrame(
            {
                "Metric": reg_mean.index,
                "Mean ± Std": [
                    f"{m:.4f} ± {s:.4f}"
                    for m, s in zip(reg_mean, reg_std)
                ],
            }
        )

        reg_path = os.path.join(out_dir, "final_avg_regression.csv")
        reg_sum.to_csv(reg_path, index=False)

        print(f"Regression report saved: {reg_path}")


def train_cross_validation():
    dataset, target2idx, _ = build_dataset(
        config["ROOT_DIR"],
        "clean_data/antiTB_mtb_cleaned_standardized.csv",
        "clean_data/multi_cleaned_all20260424.csv",
        selected_targets=config["SELECTED_TARGETS"],
    )

    num_targets = len(target2idx)
    idx2target = {v: k for k, v in target2idx.items()}

    sample = dataset[0]

    atom_dim = sample.x.shape[1]
    fp_dim = sample.mol_feature.shape[-1]

    skf = StratifiedKFold(
        n_splits=config["N_SPLITS"],
        shuffle=True,
        random_state=config["SEED"],
    )

    indices = np.arange(len(dataset))

    y_strat = np.array(
        [
            1 if data.y_target.sum().item() > 0 else 0
            for data in dataset
        ]
    )

    all_fold_cls_metrics = []
    all_fold_reg_metrics = []
    all_fold_indices = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(indices, y_strat)):
        fold_id = fold + 1

        print(f"\n>>>> FOLD {fold_id} started")

        all_fold_indices.append(
            {
                "fold": fold_id,
                "train_idx": train_idx,
                "val_idx": val_idx,
            }
        )

        train_loader = DataLoader(
            [dataset[i] for i in train_idx],
            batch_size=config["BATCH_SIZE"],
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )

        val_loader = DataLoader(
            [dataset[i] for i in val_idx],
            batch_size=64,
            shuffle=False,
            num_workers=2,
        )

        model = GAT_MultiTask_V3(
            atom_dim=atom_dim,
            fp_dim=fp_dim,
            num_targets=num_targets,
            num_experts=config["NUM_EXPERTS"],
        ).to(DEVICE)

        awl = AutomaticWeightedLoss(num_tasks=2).to(DEVICE)

        optimizer = Adam(
            [
                {"params": model.parameters()},
                {"params": awl.parameters()},
            ],
            lr=config["LR"],
        )

        early_stopping = EarlyStopping(
            patience=20,
            reg_weight=0.5,
            auprc_weight=0.5,
        )

        train_subset = [dataset[i] for i in train_idx]

        pos_weights = get_pos_weight(
            train_subset,
            num_targets,
        ).to(DEVICE)

        best_val_r2 = -float("inf")

        best_model_path = os.path.join(
            config["OUT_DIR"],
            f"best_model_fold_{fold_id}.pt",
        )

        for epoch in range(1, config["EPOCHS"] + 1):
            model.train()

            epoch_reg_losses = []
            epoch_cls_losses = []

            for data in train_loader:
                data = data.to(DEVICE)

                optimizer.zero_grad()

                mic_pred, act_logits, _, _, _ = model(data)

                m_reg = data.mic_mask.bool().reshape(-1)

                if m_reg.any():
                    reg_loss = F.huber_loss(
                        mic_pred[m_reg].reshape(-1),
                        data.y_mic[m_reg].reshape(-1),
                    )

                    epoch_reg_losses.append(reg_loss.item())

                else:
                    reg_loss = torch.tensor(
                        0.0,
                        device=DEVICE,
                        requires_grad=True,
                    )

                cls_loss = weighted_focal_loss(
                    act_logits,
                    data.y_target,
                    data.y_mask,
                    pos_weights,
                    gamma=2.0,
                )

                epoch_cls_losses.append(cls_loss.item())

                loss = awl(reg_loss, cls_loss)

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    1.0,
                )

                optimizer.step()

            cls_res, val_reg, _, _ = evaluate_fold(
                model,
                val_loader,
                DEVICE,
                num_targets,
                idx2target,
            )

            val_r2 = val_reg["R2"]

            score, mean_auprc = early_stopping(
                val_r2,
                cls_res,
            )

            if val_r2 > best_val_r2:
                best_val_r2 = val_r2
                torch.save(model.state_dict(), best_model_path)

            if epoch == 1 or epoch % 10 == 0:
                mean_reg_loss = (
                    np.mean(epoch_reg_losses)
                    if epoch_reg_losses
                    else np.nan
                )

                mean_cls_loss = (
                    np.mean(epoch_cls_losses)
                    if epoch_cls_losses
                    else np.nan
                )

                print(
                    f"Epoch [{epoch:03d}/{config['EPOCHS']}] "
                    f"| Train Reg Loss: {mean_reg_loss:.4f} "
                    f"| Train Cls Loss: {mean_cls_loss:.4f} "
                    f"| Val R2: {val_r2:.4f} "
                    f"| Mean AUPRC: {mean_auprc:.4f} "
                    f"| Stop Score: {score:.4f}"
                )

            if early_stopping.early_stop:
                break

        model.load_state_dict(
            torch.load(
                best_model_path,
                weights_only=True,
                map_location=DEVICE,
            )
        )

        model.eval()

        train_feat_loader = DataLoader(
            [dataset[i] for i in train_idx],
            batch_size=config["BATCH_SIZE"],
            shuffle=False,
        )

        (
            train_feat,
            train_y_mic,
            train_y_cls,
            train_m_mic,
            train_m_cls,
        ) = extract_ml_features(train_feat_loader)

        (
            val_feat,
            val_y_mic,
            val_y_cls,
            val_m_mic,
            val_m_cls,
        ) = extract_ml_features(val_loader)

        _, _, gat_mic_val, gat_probs_val = evaluate_fold(
            model,
            val_loader,
            DEVICE,
            num_targets,
            idx2target,
        )

        # Regression ensemble: GAT + Random Forest
        rf_reg = RandomForestRegressor(
            n_estimators=200,
            n_jobs=-1,
            random_state=config["SEED"],
        )

        train_mic_mask = train_m_mic.astype(bool)
        val_mic_mask = val_m_mic.astype(bool)

        if train_mic_mask.any() and val_mic_mask.any():
            rf_reg.fit(
                train_feat[train_mic_mask],
                train_y_mic[train_mic_mask],
            )

            rf_p_val = rf_reg.predict(val_feat)

            y_true_reg = val_y_mic[val_mic_mask]

            if len(y_true_reg) > 10:
                reg_gat_w = find_best_weight(
                    y_true_reg,
                    gat_mic_val[val_mic_mask],
                    rf_p_val[val_mic_mask],
                    task_type="reg",
                )

                ensemble_mic_val = (
                    reg_gat_w * gat_mic_val[val_mic_mask]
                    + (1 - reg_gat_w) * rf_p_val[val_mic_mask]
                )

                curr_top_n = min(300, len(y_true_reg))

                hits_300, ef_300 = calculate_screening_metrics(
                    y_true_reg,
                    ensemble_mic_val,
                    top_n=curr_top_n,
                )

                reg_unc = np.abs(
                    gat_mic_val[val_mic_mask]
                    - rf_p_val[val_mic_mask]
                )

                all_fold_reg_metrics.append(
                    {
                        "R2": r2_score(
                            y_true_reg,
                            ensemble_mic_val,
                        ),
                        "RMSE": np.sqrt(
                            mean_squared_error(
                                y_true_reg,
                                ensemble_mic_val,
                            )
                        ),
                        "Pearson": safe_pearson(
                            y_true_reg,
                            ensemble_mic_val,
                        ),
                        "Spearman": safe_spearman(
                            y_true_reg,
                            ensemble_mic_val,
                        ),
                        "Hit@300": hits_300,
                        "EF@300": ef_300,
                        "Avg_Uncertainty": np.mean(reg_unc),
                        "Best_GAT_Weight": reg_gat_w,
                    }
                )

                joblib.dump(
                    rf_reg,
                    os.path.join(
                        config["OUT_DIR"],
                        f"rf_fold_{fold_id}.pkl",
                    ),
                )

                print(
                    f"Fold {fold_id} regression best GAT weight: "
                    f"{reg_gat_w:.2f}"
                )

        # Classification ensemble: GAT + XGBoost
        for i in range(num_targets):
            target_name = idx2target[i]

            train_cls_mask = train_m_cls[:, i].astype(bool)
            val_cls_mask = val_m_cls[:, i].astype(bool)

            if not train_cls_mask.any() or not val_cls_mask.any():
                continue

            if len(np.unique(train_y_cls[train_cls_mask, i])) < 2:
                continue

            if len(np.unique(val_y_cls[val_cls_mask, i])) < 2:
                continue

            xgb = XGBClassifier(
                n_estimators=100,
                n_jobs=-1,
                scale_pos_weight=pos_weights[i].item(),
                random_state=config["SEED"],
                eval_metric="logloss",
            )

            xgb.fit(
                train_feat[train_cls_mask],
                train_y_cls[train_cls_mask, i],
            )

            p_gat = gat_probs_val[val_cls_mask, i]

            p_ml = xgb.predict_proba(
                val_feat[val_cls_mask]
            )[:, 1]

            y_true_cls = val_y_cls[val_cls_mask, i]

            cls_w = find_best_weight(
                y_true_cls,
                p_gat,
                p_ml,
                task_type="cls",
            )

            y_score_cls = cls_w * p_gat + (1 - cls_w) * p_ml

            best_t = 0.5
            y_pred_cls = (y_score_cls >= best_t).astype(int)

            all_fold_cls_metrics.append(
                {
                    "Target": target_name,
                    "AUROC": roc_auc_score(
                        y_true_cls,
                        y_score_cls,
                    ),
                    "AUPRC": average_precision_score(
                        y_true_cls,
                        y_score_cls,
                    ),
                    "BACC": balanced_accuracy_score(
                        y_true_cls,
                        y_pred_cls,
                    ),
                    "Precision": precision_score(
                        y_true_cls,
                        y_pred_cls,
                        zero_division=0,
                    ),
                    "Recall": recall_score(
                        y_true_cls,
                        y_pred_cls,
                        zero_division=0,
                    ),
                    "F1": f1_score(
                        y_true_cls,
                        y_pred_cls,
                        zero_division=0,
                    ),
                    "MCC": matthews_corrcoef(
                        y_true_cls,
                        y_pred_cls,
                    ),
                    "Best_T": best_t,
                    "Best_GAT_Weight": cls_w,
                }
            )

            joblib.dump(
                xgb,
                os.path.join(
                    config["OUT_DIR"],
                    f"xgb_{target_name}_fold_{fold_id}.pkl",
                ),
            )

            print(
                f"Fold {fold_id} {target_name} best GAT weight: "
                f"{cls_w:.2f}"
            )

    index_save_path = os.path.join(
        config["OUT_DIR"],
        "global_mskf_indices.joblib",
    )

    joblib.dump(all_fold_indices, index_save_path)

    print(f"\nFold indices saved: {index_save_path}")

    save_final_reports(
        all_fold_cls_metrics,
        all_fold_reg_metrics,
        config["OUT_DIR"],
    )


if __name__ == "__main__":
    train_cross_validation()