from __future__ import annotations
from pathlib import Path
import random, yaml, numpy as np, torch
from torch.utils.data import DataLoader
from .datasets import UdacityCSV, ImageTransform

def set_seeds(seed:int=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False

def load_preset(path_yaml: Path, name: str):
    with open(path_yaml,'r',encoding='utf-8') as f: presets=yaml.safe_load(f)
    assert name in presets, f'Preset no encontrado: {name} en {path_yaml}'
    return presets[name]

def make_loaders_from_csvs(base_dir: Path, train_csv: Path, val_csv: Path, test_csv: Path, batch_size:int, encoder:str, T:int, gain:float, tfm:ImageTransform):
    tr=UdacityCSV(train_csv, base_dir, encoder=encoder, T=T, gain=gain, tfm=tfm)
    va=UdacityCSV(val_csv, base_dir, encoder=encoder, T=T, gain=gain, tfm=tfm)
    te=UdacityCSV(test_csv, base_dir, encoder=encoder, T=T, gain=gain, tfm=tfm)
    return (DataLoader(tr,batch_size,True, num_workers=2,pin_memory=True),
            DataLoader(va,batch_size,False,num_workers=2,pin_memory=True),
            DataLoader(te,batch_size,False,num_workers=2,pin_memory=True))
