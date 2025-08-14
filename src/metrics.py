import torch, numpy as np
def mae(y_hat,y): return float(torch.mean(torch.abs(y_hat-y)).detach().cpu())
def mse(y_hat,y): return float(torch.mean((y_hat-y)**2).detach().cpu())
def backward_transfer(before, after):
    b=np.array(before,dtype=float); a=np.array(after,dtype=float)
    if len(b)==0: return 0.0
    return float((b-a).mean())
