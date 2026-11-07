import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import os
from . import displayer
import math

from tqdm import tqdm


from sklearn import preprocessing, datasets
import dataclasses
from typing import Sequence
import pytorch_warmup as warmup


class DDIM:
  def __init__(self, model, num_diffusion_timesteps = 1000, imgshape = (1,3,256,256), dataset_name="ffhq", beta_start=0.0001, beta_end = 0.02, use_fp16=False, beta_schedule="linear", device="cuda"):
    self.num_diffusion_timesteps = num_diffusion_timesteps
    self.reversed_time_steps = np.arange(self.num_diffusion_timesteps)[::-1]
    #self.betas = torch.linspace(beta_start, beta_end, steps=self.num_diffusion_timesteps)
    if beta_schedule == "cosine":
      self.betas = self.cosine_betas(self.num_diffusion_timesteps)
    elif beta_schedule == "linear" or beta_schedule is None:
      self.betas = torch.linspace(beta_start, beta_end, steps=self.num_diffusion_timesteps)
    else:
      raise ValueError(f"Unknown beta_schedule={beta_schedule}. Use 'linear' or 'cosine'.")
        
    self.alphas = 1.0 - self.betas
    self.alphas_cumprod = torch.tensor(np.cumprod(self.alphas, axis=0))
    self.alphas_cumprod_prev = torch.tensor(np.append(1.0, self.alphas_cumprod[:-1]))
    self.model = model
    self.imgshape = imgshape
    self.use_fp16 = use_fp16
    self.device = device
    self.dataset_name = dataset_name
  @staticmethod
  def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    betas = []
    for i in range(num_diffusion_timesteps):
      t1 = i / num_diffusion_timesteps
      t2 = (i + 1) / num_diffusion_timesteps
      beta = 1 - alpha_bar(t2) / alpha_bar(t1)
      betas.append(min(beta, max_beta))
    return torch.tensor(betas, dtype=torch.float32)

  @staticmethod
  def cosine_betas(num_diffusion_timesteps):
    return DDIM.betas_for_alpha_bar(
      num_diffusion_timesteps,
      lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
    )
    
  def get_alpha_bar(self, t):
    return self.alphas_cumprod[t]
  
  def instant_blur(self, x0, t):
    zt = torch.randn(self.imgshape, device=x0.device)
    xt = torch.sqrt(self.alphas_cumprod[t]) * x0 + torch.sqrt(1 - self.alphas_cumprod[t]) * zt
    return xt
  
  def ddim_sigma_t(self, t, next_t, eta):
    """
    DDIM sigma_t (for stochasticity). Uses alpha_bar values.
    """
    alpha_t = self.alphas_cumprod[t].to(self.device)
    alpha_next = self.alphas_cumprod[next_t].to(self.device)
    return eta * torch.sqrt((1 - alpha_t / alpha_next) * (1 - alpha_next) / (1 - alpha_t))
  
  def get_eps_from_UNET(self, x, t, **kwargs):
    # the model outputs:
    # - an estimation of the noise eps (chanels 0 to 2)
    # - learnt variances for the posterior  (chanels 3 to 5)
    # (see Improved Denoising Diffusion Probabilistic Models
    # by Alex Nichol, Prafulla Dhariwal
    # for the parameterization)
    # We discard the second part of the output for this practice session.
    model = kwargs.get("model",self.model)
    #print(model)
    mask_seed = kwargs.get("mask_seed",None)
    if mask_seed is not None:
      model.module.update_dropout_seed(mask_seed)
 
    if self.dataset_name == "cifar10": #cifar10
      return model(x, torch.tensor(t,device=x.device).unsqueeze(0).repeat(self.imgshape[0]))
    elif self.dataset_name == "ffhq" or self.dataset_name == "celeba_p2": #ffhq and imagenet
      return model(x, torch.tensor(t,device=x.device).unsqueeze(0).repeat(x.shape[0]))[:,:3,:,:]
      #return model(x, torch.tensor(t,device=x.device).unsqueeze(0))[:,:3,:,:]
        # we repeat batch_size times, else dataparallel does not gather the results from each gpus
    elif self.dataset_name == "celeba": #celeba
      return model(x, torch.tensor(t,device=x.device).unsqueeze(0).repeat(self.imgshape[0]))#.sample
      # because nn.DataParallel
    else:
      raise NotImplementedError

  def dynamic_threshold(self, x0, q):
    """
    Imagen-style dynamic thresholding on x0.
    If q is None: simple clamp to [-1,1].
    """
    if q is None:
      return x0.clamp(-1., 1.)
    B = x0.shape[0]
    flat = x0.abs().reshape(B, -1)
    s = torch.quantile(flat, q, dim=1)  # [B]
    s = torch.maximum(s, torch.ones_like(s))
    s = s.view(B, 1, 1, 1)
    x0 = x0 / s
    return x0.clamp(-1., 1.)
  
  def x0_hat_from_eps(self, x_t, eps_t, t, clamp=False, dt_q=None):
    """
    \hat{x}_0(x_t) from epsilon-parameterization:
      x0_hat = 1/sqrt(alpha_bar_t) * x_t - sqrt(1/alpha_bar_t - 1) * eps
    Optionally apply dynamic thresholding (dt_q) and/or clamp.
    """
    alpha_bar_t = self.alphas_cumprod[t].to(x_t.device)
    x0_hat = torch.sqrt(1.0 / alpha_bar_t) * x_t - torch.sqrt(1.0 / alpha_bar_t - 1.0) * eps_t

    if dt_q is not None:
      x0_hat = self.dynamic_threshold(x0_hat, dt_q)
    elif clamp:
      x0_hat = x0_hat.clamp(-1., 1.)

    return x0_hat

  def guided_eps_from_grad_xt(self, eps_t_model, grad_xt, t, guidance_scale):
    """
    Apply classifier-guidance-style update in epsilon space:
      eps_guided = eps - lambda * sqrt(1 - alpha_bar_t) * grad_xt
    """
    if guidance_scale <= 0.0:
      return eps_t_model
    alpha_bar_t = self.alphas_cumprod[t].to(eps_t_model.device)
    return eps_t_model - guidance_scale * torch.sqrt(1.0 - alpha_bar_t) * grad_xt
  
  def guidance_grad_xt(
      self,
      x_t,
      t,
      *,
      log_r_fn,
      f_embedder,
      dt_q=0.995
    ):

    if log_r_fn is None:
      raise ValueError("log_r_fn must be provided for guidance.")

    with torch.enable_grad():
      x_t_req = x_t.detach().requires_grad_(True)

      eps = self.get_eps_from_UNET(x_t_req, t)  # <-- no detach, so grad flows through eps(x_t)
      x0_hat_dt = self.x0_hat_from_eps(x_t_req, eps, t, clamp=False, dt_q=None)
      #x0_hat_dt = self.dynamic_threshold(x0_hat, dt_q)
      if f_embedder is not None:
        x0_hat_dt = 0.5 * x0_hat_dt + 0.5  # (-1,1) to (0,1)
        emb = f_embedder(x0_hat_dt)
        log_r = log_r_fn(emb)             # [B, D]
      else:
        log_r = log_r_fn(x0_hat_dt)               # [B]
      grad_xt = torch.autograd.grad(log_r.sum(), x_t_req, create_graph=False)[0]

    return grad_xt.detach(), x0_hat_dt.detach(), log_r.detach()
  
  def sample_jeffrey(self, show_steps=True, save_images=True, **kwargs):
    images = []
    seed = kwargs.get("seed", None)
    gen = torch.Generator(device=self.device)
    if seed is not None:
      gen.manual_seed(seed)

    eta = kwargs.get("eta", 0.0)
    
    # Guidance / DT kwargs
    guidance_scale = float(kwargs.get("guidance_scale", 0.0))
    log_r_fn = kwargs.get("log_r_fn", None)
    f_embedder = kwargs.get("f_embedder", None)
    dt_q = kwargs.get("dt_q", 0.995)     # None to disable
    
    guidance_mode = kwargs.get("guidance_mode", "none")  # or "stop_unet"

    # init x_T
    xt = torch.randn(self.imgshape, device=self.device, generator=gen)

    skip_type = kwargs.get("skip_type", "none")
    seq = self.reversed_time_steps
    if skip_type == "uniform":
      S = kwargs.get("num_diffusion_timesteps", 100)
      seq = np.linspace(0, 1, S) * (self.num_diffusion_timesteps - 1)
      seq = [int(s) for s in seq][::-1]
    if skip_type == "quad":
      S = kwargs.get("num_diffusion_timesteps", 100)
      seq = (np.linspace(0, np.sqrt(self.num_diffusion_timesteps * 0.8), S) ** 2)
      seq = [int(s) for s in seq][::-1]

    delta = float(kwargs.get("delta", 1001))
    next_seq = seq[1:] + [0]
    seq_iterator = seq if show_steps else tqdm(seq, desc="Sampling Progress")

    for i, t in enumerate(seq_iterator):

      next_t = next_seq[i] if skip_type in ["uniform", "quad"] else (t - 1)
      z = torch.randn_like(xt, device=self.device)

      alpha_t = self.alphas_cumprod[t].to(z.device)
      alpha_t_next = self.alphas_cumprod[next_t].to(z.device)
      sigma_t = self.ddim_sigma_t(t, next_t, eta)

      # UNet prediction (no grad wrt params)
      with torch.no_grad():
        eps_t_model = self.get_eps_from_UNET(xt, t)
      
      # Guidance
      eps_t = eps_t_model
      if guidance_scale > 0.0 and t > 0 and t <= delta and log_r_fn is not None:
        grad_xt, _, _ = self.guidance_grad_xt(
          xt, t, log_r_fn=log_r_fn, f_embedder=f_embedder, dt_q=dt_q
        )
        eps_t = self.guided_eps_from_grad_xt(eps_t_model, grad_xt, t, guidance_scale)

      # x0 prediction from (possibly guided) eps, then dynamic thresholding
      x0_pred = self.x0_hat_from_eps(xt, eps_t, t, clamp=False, dt_q=None)
      #x0_pred = self.dynamic_threshold(x0_pred, dt_q)

      # DDIM update
      if t > 1:
        mean = torch.sqrt(alpha_t_next) * x0_pred + torch.sqrt((1 - alpha_t_next) - sigma_t ** 2) * eps_t
        noise = sigma_t * z
      else:
        mean = x0_pred
        noise = torch.zeros_like(xt)

      xt = mean + noise

      if show_steps and (i % (max(1, len(seq) // 10)) == 0 or i == len(seq) - 1):
        print("Iteration :", i + 1, "Timestep :", t)
        pilimg = displayer.display_as_pilimg(torch.cat((xt, x0_pred), dim=3))
      if save_images and i % 10 == 0:
        pilimg = displayer.display_as_pilimg(torch.cat((xt, x0_pred), dim=3), disp=False)
        images.append(pilimg)

    return xt, images

  def sample_jeffrey_ddpm(self, show_steps=True, save_images=True, **kwargs):
    """
    Ancestral DDPM sampling (Ho et al., 2020), with the same eps-space
    guidance / dynamic-thresholding hooks as sample_jeffrey (DDIM).
    """
    images = []
    seed = kwargs.get("seed", None)
    gen = torch.Generator(device=self.device)
    if seed is not None:
      gen.manual_seed(seed)

    # Guidance / DT kwargs
    guidance_scale = float(kwargs.get("guidance_scale", 0.0))
    log_r_fn = kwargs.get("log_r_fn", None)
    f_embedder = kwargs.get("f_embedder", None)
    dt_q = kwargs.get("dt_q", 0.995)     # None to disable
    delta = float(kwargs.get("delta", 1001))

    # init x_T
    xt = torch.randn(self.imgshape, device=self.device, generator=gen)

    n_steps = len(self.reversed_time_steps)
    seq_iterator = self.reversed_time_steps if show_steps else tqdm(self.reversed_time_steps, desc="Sampling Progress")

    for i, t in enumerate(seq_iterator):
      z = torch.randn(self.imgshape, device=self.device, generator=gen) if t > 1 else torch.zeros_like(xt)

      alpha_t = self.alphas[t].to(self.device)
      alpha_bar_t = self.alphas_cumprod[t].to(self.device)
      sigma_t = torch.sqrt(self.betas[t]).to(self.device)

      # UNet prediction (no grad wrt params)
      with torch.no_grad():
        eps_t_model = self.get_eps_from_UNET(xt, t)

      # Guidance
      eps_t = eps_t_model
      if guidance_scale > 0.0 and t > 0 and t <= delta and log_r_fn is not None:
        grad_xt, _, _ = self.guidance_grad_xt(
          xt, t, log_r_fn=log_r_fn, f_embedder=f_embedder, dt_q=dt_q
        )
        eps_t = self.guided_eps_from_grad_xt(eps_t_model, grad_xt, t, guidance_scale)

      # x0 prediction from the pre-update xt (still at noise level t), for display/DT only
      x0_pred = self.x0_hat_from_eps(xt, eps_t, t, clamp=False, dt_q=None)
      #x0_pred = self.dynamic_threshold(x0_pred, dt_q)

      # DDPM ancestral update
      xt = 1 / torch.sqrt(alpha_t) * (xt - (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t) * eps_t) + sigma_t * z

      if show_steps and (i % (max(1, n_steps // 10)) == 0 or i == n_steps - 1):
        print("Iteration :", i + 1, "Timestep :", t)
        pilimg = displayer.display_as_pilimg(torch.cat((xt, x0_pred), dim=3))
      if save_images and i % 10 == 0:
        pilimg = displayer.display_as_pilimg(torch.cat((xt, x0_pred), dim=3), disp=False)
        images.append(pilimg)

    return xt, images
