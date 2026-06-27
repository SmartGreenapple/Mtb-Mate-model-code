import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Descriptors, MACCSkeys
from rdkit.ML.Descriptors import MoleculeDescriptors
from torch_geometric.data import Data
import joblib
from sklearn.preprocessing import StandardScaler


ECFP_DIM = 2048               
EDGE_DIM = 0                 

# ======================================================
# 1. 核心描述符定义与计算器初始化
# ======================================================
DESC_NAMES = [
    'MolLogP', 'MolWt', 'TPSA', 'NumHAcceptors', 'NumHDonors',
    'NumRotatableBonds', 'FractionCSP3', 'LabuteASA', 'HallKierAlpha',
    'HeavyAtomCount', 'NHOHCount', 'NOCount', 'RingCount',
    'MaxAbsPartialCharge', 'MinAbsPartialCharge', 'MolMR',
    'NumAliphaticRings', 'NumAromaticRings', 'NumSaturatedRings', 'BertzCT'
]

DESC_CALCULATOR = MoleculeDescriptors.MolecularDescriptorCalculator(DESC_NAMES)


def fit_descriptors_scaler(all_smiles_list):
    all_descs = []
    for s in all_smiles_list:
        mol = Chem.MolFromSmiles(s)
        if mol:
            ds = list(DESC_CALCULATOR.CalcDescriptors(mol))
            all_descs.append(ds)
    
    all_descs = np.array(all_descs, dtype=np.float32)

    all_descs = np.nan_to_num(all_descs, nan=0.0, posinf=0.0, neginf=0.0)
    
    scaler = StandardScaler()
    scaler.fit(all_descs)
    
    joblib.dump(scaler, 'descriptor_scaler.pkl')
    print("Scaler fitted and saved.")
    return scaler



class FeaturizationParameters:
    def __init__(self):
        self.COMMON_ATOMS = [1, 6, 7, 8, 9, 15, 16, 17, 35, 53] 
        self.ATOM_FEATURES = {
            'atomic_num': self.COMMON_ATOMS, 
            'degree': [0, 1, 2, 3, 4, 5],    
            'formal_charge': [-1, -2, 1, 2, 0],  
            'chiral_tag': [0, 1, 2, 3],          
            'num_Hs': [0, 1, 2, 3, 4],       
            'hybridization': [               
                Chem.rdchem.HybridizationType.SP,
                Chem.rdchem.HybridizationType.SP2,
                Chem.rdchem.HybridizationType.SP3,
                Chem.rdchem.HybridizationType.SP3D,
                Chem.rdchem.HybridizationType.SP3D2
            ],
        }

        self.ATOM_FDIM = sum(len(v) + 1 for v in self.ATOM_FEATURES.values()) + 2

PARAMS = FeaturizationParameters()
ATOM_FDIM = PARAMS.ATOM_FDIM

def onek_encoding_unk(value, choices):
    vec = np.zeros(len(choices) + 1, dtype=np.float32)
    try:
        idx = choices.index(value)
    except ValueError:
        idx = -1
    vec[idx] = 1.0
    return vec


def atom_features(atom):
    feats = [
        onek_encoding_unk(atom.GetAtomicNum(), PARAMS.ATOM_FEATURES['atomic_num']), 
        onek_encoding_unk(atom.GetTotalDegree(), PARAMS.ATOM_FEATURES['degree']),
        onek_encoding_unk(atom.GetFormalCharge(), PARAMS.ATOM_FEATURES['formal_charge']),
        onek_encoding_unk(int(atom.GetChiralTag()), PARAMS.ATOM_FEATURES['chiral_tag']),
        onek_encoding_unk(int(atom.GetTotalNumHs()), PARAMS.ATOM_FEATURES['num_Hs']),
        onek_encoding_unk(atom.GetHybridization(), PARAMS.ATOM_FEATURES['hybridization']),

        np.array([1.0 if atom.GetIsAromatic() else 0.0], dtype=np.float32),
        np.array([atom.GetMass() * 0.01], dtype=np.float32)

    ]

    return np.concatenate(feats)



def featurize_molecule(smiles, scaler=None, radius=2, num_bits=ECFP_DIM, augment=False):

    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None

    if augment:
        aug_smiles = Chem.MolToSmiles(mol, doRandom=True, canonical=False)
        mol = Chem.MolFromSmiles(aug_smiles) or mol

    x = np.array([atom_features(a) for a in mol.GetAtoms()], dtype=np.float32)
    
    edge_index = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_index.extend([[i, j], [j, i]])

    if len(edge_index) == 0:
        edge_index_ts = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index_ts = torch.from_numpy(np.array(edge_index)).t().contiguous()
    

    edge_attr_ts = torch.empty((edge_index_ts.size(1), 0), dtype=torch.float)

    graph_data = Data(x=torch.from_numpy(x), edge_index=edge_index_ts, edge_attr=edge_attr_ts)


    fp_bit = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=num_bits)
    fp_arr = np.zeros(num_bits, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp_bit, fp_arr)

    maccs_bit = MACCSkeys.GenMACCSKeys(mol)
    maccs_arr = np.zeros(maccs_bit.GetNumBits(), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(maccs_bit, maccs_arr) 
    
    combined_fp = np.concatenate([fp_arr, maccs_arr]) 

    ds = list(DESC_CALCULATOR.CalcDescriptors(mol))
    ds = np.array(ds, dtype=np.float32).reshape(1, -1)
    ds = np.nan_to_num(ds, nan=0.0, posinf=0.0, neginf=0.0)

    if scaler is not None:
        ds = scaler.transform(ds) 
    
    desc_tensor = torch.from_numpy(ds.flatten())
    fp_tensor = torch.from_numpy(combined_fp)

    mol_feature = torch.cat([fp_tensor, desc_tensor], dim=0)

    return graph_data, mol_feature