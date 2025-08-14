import torch
def rate_encode(img01, T, gain=0.5):
    if img01.dim()==2: img01=img01.unsqueeze(0)
    probs=torch.clamp(img01*gain,0,1)
    rand=torch.rand((T,*probs.shape), device=img01.device)
    return (rand<probs).to(img01.dtype)
def latency_encode(img01, T):
    if img01.dim()==2: img01=img01.unsqueeze(0)
    x=torch.clamp(img01,1e-5,1-1e-5)
    t_star=torch.floor((1-x)*(T-1)).long()
    spikes=torch.zeros((T,*x.shape), device=img01.device, dtype=img01.dtype)
    for t in range(T):
        m=(t_star==t)
        if m.any(): spikes[t][m]=1.0
    return spikes
