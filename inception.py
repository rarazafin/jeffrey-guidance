import torch
import torch.nn.functional as F
from pytorch_fid.inception import InceptionV3
from torch.nn.functional import adaptive_avg_pool2d

class InceptionEmbedder(torch.nn.Module):
    """
    Usage:
        f_inception = InceptionEmbedder(device="cuda", implementation="fid-pytorch", inceptionv3=True)
        emb = f_inception(samples)  # [N, 2048]
    Input: samples shape [N,3,256,256] values in [0,1]
    """
    def __init__(self, device="cuda", implementation="fid-pytorch", inceptionv3=True):
        super().__init__()
        self.device = torch.device(device)
        self.implementation = implementation
        self.inceptionv3 = inceptionv3

 
        if implementation == "fid-pytorch":
            block_idx = InceptionV3(use_fid_inception=True).BLOCK_INDEX_BY_DIM[2048]
            self.model = InceptionV3([block_idx]).to(self.device).eval()
            for p in self.model.parameters():
                p.requires_grad_(False)
        else:
            raise NotImplementedError(f"Unknown implementation: {implementation}")

    def forward(self, samples: torch.Tensor) -> torch.Tensor:


        if self.implementation == "fid-pytorch":
            samples = samples.to(self.device, non_blocking=True)
            feats = self.model(samples)[0]  # [N,2048,h,w]
            return adaptive_avg_pool2d(feats, output_size=(1, 1)).squeeze(-1).squeeze(-1)  # [N,2048]
        else:
            raise NotImplementedError