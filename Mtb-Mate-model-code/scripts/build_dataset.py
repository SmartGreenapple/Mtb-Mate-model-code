import os
import torch
import pandas as pd
import numpy as np
from rdkit import Chem
from torch_geometric.data import Data
from featurization import (
    featurize_molecule, 
    ATOM_FDIM, 
    ECFP_DIM,
    DESC_CALCULATOR
)
from sklearn.preprocessing import StandardScaler


selected_targets = [
    "DprE1","DXR","DNA gyrase","InhA","MptpA","MptpB","PknB","TMPK"
]

ROOT_DIR = "/home/yuntang/data/data_mtb/multi_data"

PMIC_CSV_PATH = os.path.join(ROOT_DIR, "clean_data/antiTB_mtb_cleaned_standardized.csv")
CLS_CSV_PATH  = os.path.join(ROOT_DIR, "clean_data/multi_cleaned_all20260424.csv")

class MultiTaskData(Data):
    def __cat_dim__(self, key, value, *args, **kwargs):
        if key in ['mol_feature', 'y_target', 'y_mask']:
            return None
        return super().__cat_dim__(key, value, *args, **kwargs)

def canonicalize(smi):
    mol = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(mol) if mol else smi

def build_dataset(root_dir, pmic_csv, cls_csv, selected_targets=None):
    pmic_path = os.path.join(root_dir, pmic_csv)
    cls_path = os.path.join(root_dir, cls_csv)
    

    print("Pre-processing SMILES...")
    raw_pmic_df = pd.read_csv(pmic_path).dropna(subset=["std_smiles", "pMIC"])

    pmic_dict = {canonicalize(str(s)): float(p) for s, p in zip(raw_pmic_df["std_smiles"], raw_pmic_df["pMIC"])}

    raw_cls_df = pd.read_csv(cls_path).dropna(subset=["canon_smiles", "target", "label"])
    if selected_targets is None:
        selected_targets = sorted(raw_cls_df["target"].unique().tolist())
    target2idx = {t: i for i, t in enumerate(selected_targets)}
    K = len(target2idx)


    target_label_dict = {}
    for _, r in raw_cls_df.iterrows():
        t = r["target"]
        if t in target2idx:

            smi = canonicalize(str(r["canon_smiles"])) 
            target_label_dict.setdefault(smi, {})[t] = float(r["label"])

    all_smiles = sorted(set(pmic_dict.keys()) | set(target_label_dict.keys()))
    

    temp_storage = []
    all_raw_descs = []
    
    print(f"Extracting features for {len(all_smiles)} molecules...")
    for smi in all_smiles:
        res = featurize_molecule(smi, scaler=None, augment=False)
        if res is not None:
            graph_data, mol_feature = res
            raw_desc = mol_feature[-20:].numpy()
            temp_storage.append((smi, graph_data, mol_feature[:2215])) 
            all_raw_descs.append(raw_desc)
    
    all_raw_descs = np.array(all_raw_descs)
    scaler = StandardScaler()
    scaler.fit(all_raw_descs)
    

    d_mean = torch.tensor(scaler.mean_, dtype=torch.float)
    d_std = torch.tensor(np.sqrt(scaler.var_) + 1e-6, dtype=torch.float)
    
    dataset = []
    for i, (smi, graph_data, fp_part) in enumerate(temp_storage):
        data = MultiTaskData()
        data.smiles = smi
        data.x = graph_data.x
        data.edge_index = graph_data.edge_index
        data.edge_attr = graph_data.edge_attr

    
        data.y_mic = torch.tensor([pmic_dict.get(smi, 0.0)], dtype=torch.float)
        data.mic_mask = torch.tensor([1 if smi in pmic_dict else 0], dtype=torch.float)
        
        y_cls = torch.zeros(K, dtype=torch.float)
        y_m = torch.zeros(K, dtype=torch.float)
        if smi in target_label_dict:
            for t_name, label in target_label_dict[smi].items():
                idx = target2idx[t_name]
                y_cls[idx] = label
                y_m[idx] = 1.0
        data.y_target = y_cls
        data.y_mask = y_m


        raw_desc_tensor = torch.from_numpy(all_raw_descs[i])
        norm_desc = (raw_desc_tensor - d_mean) / d_std
        
  
        data.mol_feature = torch.cat([fp_part, norm_desc], dim=0)
        
        dataset.append(data)

    print(f"Dataset built. Samples: {len(dataset)}")
    return dataset, target2idx, (d_mean, d_std)