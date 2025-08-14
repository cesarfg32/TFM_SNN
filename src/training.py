from dataclasses import dataclass
from typing import Dict, List
from pathlib import Path
import time, json, torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

@dataclass
class TrainConfig:
    epochs:int=2; batch_size:int=8; lr:float=1e-3; amp:bool=True
def _device(): return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def _save_manifest(out_dir: Path, manifest: Dict):
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir/'manifest.json','w',encoding='utf-8') as f: json.dump(manifest,f,indent=2,ensure_ascii=False)

def train_supervised(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, loss_fn: nn.Module, cfg: TrainConfig, out_dir: Path, method=None) -> Dict:
    device=_device(); model=model.to(device)
    opt=torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scaler=torch.cuda.amp.GradScaler(enabled=cfg.amp)
    hist={'train_mae':[],'val_mae':[],'train_mse':[],'val_mse':[]}
    start=time.time()
    for ep in range(cfg.epochs):
        model.train(True); pbar=tqdm(train_loader, desc=f'Epoch {ep+1}/{cfg.epochs}')
        for x,y in pbar:
            x=x.to(device); y=y.to(device); opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=cfg.amp):
                y_hat=model(x); loss=loss_fn(y_hat,y)
                if method is not None: loss = loss + method.penalty()
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            pbar.set_postfix(loss=float(loss.detach().cpu()))
        def eval_loader(loader):
            mae_sum=mse_sum=n=0.0
            for x,y in loader:
                x=x.to(device); y=y.to(device)
                with torch.no_grad(): y_hat=model(x)
                mae_sum += torch.mean(torch.abs(y_hat-y)).item()*len(y)
                mse_sum += torch.mean((y_hat-y)**2).item()*len(y)
                n += len(y)
            return mae_sum/n, mse_sum/n
        tr_mae,tr_mse=eval_loader(train_loader); va_mae,va_mse=eval_loader(val_loader)
        hist['train_mae'].append(tr_mae); hist['train_mse'].append(tr_mse)
        hist['val_mae'].append(va_mae);   hist['val_mse'].append(va_mse)
    elapsed=time.time()-start
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir/'metrics.json','w',encoding='utf-8') as f: json.dump(hist,f,indent=2)
    _save_manifest(out_dir, {'mode':'supervised','epochs':cfg.epochs,'batch_size':cfg.batch_size,'lr':cfg.lr,'amp':cfg.amp,'elapsed_sec':elapsed,'device':str(device)})
    return {'history':hist,'elapsed_sec':elapsed}

def train_continual(model: nn.Module, tasks: List[Dict], make_loader_fn, loss_fn: nn.Module, cfg: TrainConfig, out_dir: Path, method=None) -> Dict:
    device=_device(); model=model.to(device); results={}; seen=[]
    for ti,task in enumerate(tasks):
        name=task['name']; t_out=out_dir/f'task_{ti+1}_{name}'; t_out.mkdir(parents=True, exist_ok=True)
        train_loader,val_loader,test_loader=make_loader_fn(task, cfg.batch_size)
        if method is not None and hasattr(method,'begin_task'): method.begin_task()
        _ = train_supervised(model, train_loader, val_loader, loss_fn, cfg, t_out, method=method)
        if method is not None and hasattr(method,'end_task'): method.end_task()
        def eval_loader(loader):
            mae_sum=mse_sum=n=0.0
            for x,y in loader:
                x=x.to(device); y=y.to(device)
                with torch.no_grad(): y_hat=model(x)
                mae_sum += torch.mean(torch.abs(y_hat-y)).item()*len(y)
                mse_sum += torch.mean((y_hat-y)**2).item()*len(y)
                n += len(y)
            return mae_sum/n, mse_sum/n
        te_mae,te_mse=eval_loader(test_loader); results[name]={'test_mae':te_mae,'test_mse':te_mse}
        seen.append((name,test_loader))
        for pname,p_loader in seen[:-1]:
            p_mae,p_mse=eval_loader(p_loader)
            results[pname][f'after_{name}_mae']=p_mae; results[pname][f'after_{name}_mse']=p_mse
    with open(out_dir/'continual_results.json','w',encoding='utf-8') as f: json.dump(results,f,indent=2)
    return results
