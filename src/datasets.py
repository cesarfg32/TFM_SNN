from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
from pathlib import Path
import cv2, numpy as np, pandas as pd, torch
from torch.utils.data import Dataset
from .encoders import rate_encode, latency_encode

@dataclass
class ImageTransform:
    width:int=160; height:int=80; grayscale:bool=True
    crop: Optional[Tuple[int,int,int,int]] = None

def _load_image(path: Path, tfm: ImageTransform) -> torch.Tensor:
    img=cv2.imread(str(path)); assert img is not None, f"No se pudo leer: {path}"
    if tfm.crop is not None:
        y0,y1,x0,x1=tfm.crop; img=img[y0:y1,x0:x1]
    if tfm.grayscale:
        img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img=cv2.resize(img,(tfm.width,tfm.height),interpolation=cv2.INTER_AREA)
        return torch.from_numpy(img.astype(np.float32)/255.0).unsqueeze(0)
    else:
        img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img=cv2.resize(img,(tfm.width,tfm.height),interpolation=cv2.INTER_AREA)
        arr=img.astype(np.float32)/255.0
        return torch.from_numpy(np.transpose(arr,(2,0,1)))

class UdacityCSV(Dataset):
    def __init__(self, csv_path: Path, base_dir: Path, encoder='rate', T=20, gain=0.5, tfm:ImageTransform=None, camera='center'):
        self.csv=pd.read_csv(csv_path); self.base_dir=Path(base_dir)
        self.encoder=encoder; self.T=int(T); self.gain=float(gain)
        self.tfm=tfm or ImageTransform(); assert camera in ['center','left','right']; self.camera=camera
    def __len__(self): return len(self.csv)
    def __getitem__(self, idx:int):
        row=self.csv.iloc[idx]
        img01=_load_image((self.base_dir/row[self.camera]).resolve(), self.tfm)
        if self.encoder=='rate': x=rate_encode(img01,self.T,self.gain)
        elif self.encoder=='latency': x=latency_encode(img01,self.T)
        else: raise ValueError(f"Encoder no reconocido: {self.encoder}")
        y=torch.tensor([float(row['steering'])], dtype=torch.float32)
        return x,y

class HDF5SpikesDataset(Dataset):
    def __init__(self, h5_path: Path):
        import h5py
        self.h5=h5py.File(h5_path,'r'); self.grp=self.h5['samples']; self.keys=list(self.grp.keys()); self.y=self.h5['steering']
        self.T=int(self.h5.attrs.get('T',0))
    def __len__(self): return len(self.keys)
    def __getitem__(self, idx:int):
        import torch
        k=self.keys[idx]; x=torch.from_numpy(self.grp[k][()]).float().unsqueeze(1)
        y=torch.tensor([float(self.y[idx])], dtype=torch.float32); return x,y
    def __del__(self):
        try: self.h5.close()
        except Exception: pass
