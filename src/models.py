import torch, torch.nn as nn, snntorch as snn
from snntorch import surrogate
class SNNVisionRegressor(nn.Module):
    def __init__(self,in_channels=1,lif_beta=0.95):
        super().__init__()
        self.features=nn.Sequential(
            nn.Conv2d(in_channels,16,5,2,2), nn.ReLU(inplace=True),
            nn.Conv2d(16,32,3,2,1), nn.ReLU(inplace=True),
            nn.Conv2d(32,64,3,2,1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((5,10)),
        )
        self.flat_dim=64*5*10
        self.lif=snn.Leaky(beta=lif_beta, surrogate_function=surrogate.fast_sigmoid())
        self.fc=nn.Linear(self.flat_dim,128)
        self.readout=nn.Linear(128,1)
    def forward(self,x):
        T,B,C,H,W=x.shape; mem=self.lif.init_leaky(); preds=[]
        for t in range(T):
            ft=self.features(x[t]).flatten(1)
            spk,mem=self.lif(self.fc(ft),mem)
            preds.append(self.readout(mem))
        return torch.stack(preds,0).mean(0)
