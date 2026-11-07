import torch
import torchvision
import numpy as np
import torch.nn as nn
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from PIL import Image
import matplotlib.animation as animation
from IPython.display import HTML
from IPython.display import Image as IPImage
from IPython.display import display
from torchvision.utils import make_grid
from torchvision.transforms.functional import to_pil_image


def pilimg_to_tensor(pil_img):
  t = torchvision.transforms.ToTensor()(pil_img)
  t = 2*t-1 # [0,1]->[-1,1]
  t = t.unsqueeze(0)
  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  t = t.to(device)
  return(t)

def display_as_pilimg(t,disp=True):
  t = 0.5+0.5*t.to('cpu')
  t = t.squeeze()
  t = t.clamp(0.,1.)
  pil_img = torchvision.transforms.ToPILImage()(t)
  if disp: display(pil_img)
  return(pil_img)

def animation(images,fps=15,name_file='animation.gif',type = "image"):
    'images: list of pilimg images'

    
    fig, ax = plt.subplots()
    if type == "image":
        im = ax.imshow(images[0])
        
        # update the image
        def update(frame):
            im.set_data(images[frame])
            return [im]
            
    if type == "tabular":
        scatter = ax.scatter(images[0][:,0].cpu().numpy(), images[0][:,1].cpu().numpy(), s=2.0)  # Affichage des points avec taille de marqueur s=1

        # update the image
        def update(frame):
            scatter.set_offsets(images[frame].cpu().numpy())
            return [scatter]

    ani = FuncAnimation(fig, update, frames=len(images), blit=True)
    ani.save(name_file, writer='imagemagick', fps=fps)
    display(IPImage(name_file))