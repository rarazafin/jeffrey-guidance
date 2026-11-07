
import clip
import torch
import torch.nn as nn
from torch.nn import functional as F
from functools import partial


class ClipEmbedder(nn.Module):
    def __init__(self, task, device):
        super(ClipEmbedder, self).__init__()
        self.model, _ = clip.load("ViT-B/32", device=device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.register_buffer("mean", torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1,3,1,1))
        self.register_buffer("std", torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1,3,1,1))
        self.device = device
        if task.lower() == "fid":
            texts = [
                'a blurry photo.',
                'a low contrast photo.',
                'a high contrast photo.',
                'a bad photo.',
                'a good photo.',
                'a high quality photo.',
                'a low quality photo.'
            ]
        else:
            raise NotImplementedError(f"Unknown task: {task}")
        with torch.no_grad():
            text_tokens = clip.tokenize(texts).to(self.device)
            text_features = self.model.encode_text(text_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        self.register_buffer("text_features", text_features)
        
    def preprocess_clip_tensor(self, x):

        # x : (B,3,H,W) in [0,1]

        # resize
        x = F.interpolate(x, size=224, mode="bicubic", align_corners=False)

        # center crop
        h, w = x.shape[-2:]
        top = (h - 224)//2
        left = (w - 224)//2
        x = x[:,:,top:top+224,left:left+224]

        # normalize
        x = (x - self.mean) / self.std

        return x

    def forward(self, x): # x : (B,3,H,W) in [0,1]
        image_features = self.model.encode_image(self.preprocess_clip_tensor(x))
        image_features = image_features/image_features.norm(dim=-1, keepdim=True)
        clip_values = image_features @ self.text_features.T
        return clip_values