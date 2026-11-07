from absl import app
from absl import flags
from ml_collections.config_flags import config_flags
import logging

import gc
import io
import os
import time
import yaml

import torch
import torch.nn as nn
import tensorflow as tf

import kagglehub
from torchvision import datasets, transforms
from dataset import Dataset
from torchvision.datasets import CIFAR10

import torch
from models.ffhq.unet import create_model
from models.cifar.model import UNet

from scheduler import GradualWarmupScheduler

from tqdm import tqdm

from diffusion.Diffusion import DDIM
import matplotlib.pyplot as plt

FLAGS = flags.FLAGS

flags.DEFINE_string("config", None, "config file name")
flags.DEFINE_string("workdir", None, "Work directory.")
flags.DEFINE_string("checkpoint", None, "Checkpoint file name")
flags.DEFINE_boolean("scheduler", True, "Use scheduler during training")
flags.DEFINE_integer("batch_size", 64, "Sample batch size")
flags.DEFINE_integer("model_idx",None,"Model index for deep ensembling")
flags.DEFINE_integer("num_epochs", None, "Number of training epochs")
flags.DEFINE_float("lr", 2e-4, "Learning rate")
flags.DEFINE_enum("ensemble_type", None, ["none", "mc_dropout", "deep_ensemble"], "Type of ensemble")
flags.mark_flags_as_required(["workdir", "config", "batch_size", "num_epochs"])

def main(argv):
  
  tf.io.gfile.makedirs(FLAGS.workdir)
  # Set logger so that it outputs to both console and file
  # Make logging work for both disk and Google Cloud Storage
  workdir = FLAGS.workdir
  
  gfile_stream = open(os.path.join(FLAGS.workdir, 'stdout.txt'), 'w')
  handler = logging.StreamHandler(gfile_stream)
  formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
  handler.setFormatter(formatter)
  logger = logging.getLogger()
  logger.addHandler(handler)
  logger.setLevel('INFO')
  
  model_config_name = FLAGS.config
  with open(os.path.join(workdir, model_config_name), "r") as file:
      model_config = yaml.safe_load(file)
      
  
  batch_size = FLAGS.batch_size
  num_epochs = FLAGS.num_epochs
  ckpt_file = FLAGS.checkpoint
  scheduler = FLAGS.scheduler
  logging.info("Loading the dataset")
  if model_config_name == "configs/cifar10.yaml":
    path = './cifar10'
    dataset = CIFAR10(
        root=path, train=True, download=True,
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]))
    logging.info("Path to images: {}".format(path))
    scaler = None
    inverse_scaler = lambda t: 0.5 + 0.5 * t
  if model_config_name == "configs/ffhq.yaml":
    path = kagglehub.dataset_download("denislukovnikov/ffhq256-images-only")
    logging.info("Path to images: {}".format(path))
    dataset = Dataset(path,model_config,"train")
    scaler = dataset.get_data_scaler
    inverse_scaler = dataset.get_data_inverse_scaler
    dataset = dataset.get_dataset()
  dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=0, drop_last=True, pin_memory=True)
  
  
  # Load model
  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  if model_config_name == "configs/ffhq.yaml":
    model = create_model(**model_config)
    lr = 2e-5
  if model_config_name == "configs/cifar10.yaml":
    model = UNet(T=model_config["T"], ch=model_config["channel"], ch_mult=model_config["channel_mult"], 
                 attn=model_config["attn"], num_res_blocks=model_config["num_res_blocks"],
                 dropout=model_config["dropout"])
    lr = 1e-4
  if torch.cuda.device_count() > 1:
    logging.info(f"Using {torch.cuda.device_count()} GPUs!")
    model = nn.DataParallel(model)
  model.to(device)
  model.train()
  logging.info("Number of parameters: {}".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
  
  checkpoint_dir = os.path.join(workdir, "checkpoints")
  tf.io.gfile.makedirs(checkpoint_dir)

  optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
  criterion = nn.MSELoss()
  if scheduler:
    cosineScheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
          optimizer=optimizer, T_max=num_epochs, eta_min=0, last_epoch=-1)
  
    warmUpScheduler = GradualWarmupScheduler(
          optimizer=optimizer, multiplier=2., warm_epoch=num_epochs // 10, after_scheduler=cosineScheduler)
    
  start_epoch = 0
  if ckpt_file is not None:
    # Charger le checkpoint
    logging.info("loading checkpoint {}".format(ckpt_file))
    checkpoint = torch.load(os.path.join(checkpoint_dir, ckpt_file), map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and checkpoint['scheduler_state_dict'] is not None:
        warmUpScheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    
    
  if model_config_name == "configs/ffhq.yaml":
    model_fn = lambda x,t: model(x,t)[:,:3,:,:]
    model_fn_sampling = lambda x,t: model(x,torch.tensor(t,device=device).unsqueeze(0))[:,:3,:,:]
  if model_config_name == "configs/cifar10.yaml":
    model_fn = model
    model_fn_sampling = lambda x,t: model(x,x.new_ones([x.shape[0], ], dtype=type) * t)
  
  ddim = DDIM(
    model,
    imgshape=(batch_size,3,model_config["image_size"],model_config["image_size"])
  ) 
  samples_dir = os.path.join(workdir,"samples")
  os.makedirs(samples_dir, exist_ok=True)
  sample_first = False
  if sample_first and (ckpt_file is not None):
    with torch.no_grad():
      ddim.model = model_fn_sampling
      ddim.imgshape = (4,3,model_config["image_size"],model_config["image_size"])
      sample,_ = ddim.sample_image(show_steps=False,save_images=False,eta=1.0,device=device)
      fig, axes = plt.subplots(1, 4, figsize=(20, 10))
      for i, ax in enumerate(axes.flat):
        img = 0.5 + 0.5 * sample[i].to('cpu')
        img = img.permute(1, 2, 0).clamp(0, 1).numpy()
        ax.imshow(img)
        ax.axis("off")

      plt.tight_layout()
      save_path = os.path.join(samples_dir, f"test.png")
      plt.savefig(save_path)
      ddim.imgshape = (batch_size,3,model_config["image_size"],model_config["image_size"])
      ddim.model = model_fn
  logging.info("Training ?: {}".format(model.training))
  sampling_interval = 1
  #with torch.no_grad():
  #  logging.info(f"First loss: {ddim.loss(dataset[0][0].to(device))}")
  for epoch in range(start_epoch, num_epochs):
    with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
      for batch_image,_ in tqdmDataLoader:
        # train

        x0 = batch_image.to(device)
        loss = ddim.loss(x0,criterion)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), 1.)
        optimizer.step()
        tqdmDataLoader.set_postfix(ordered_dict={
                    "epoch": epoch,
                    "loss: ": loss.item(),
                    "batch size: ": x0.shape[0],
                    "LR": optimizer.state_dict()['param_groups'][0]["lr"]
         })
        if scheduler: warmUpScheduler.step()
        torch.save({
          'model_state_dict': model.state_dict(),
          'optimizer_state_dict': optimizer.state_dict(),
          'scheduler_state_dict': warmUpScheduler.state_dict() if scheduler else None,
          'epoch': epoch,
        }, os.path.join(checkpoint_dir, 'cifar10_ckpt_' + str(epoch) + "_.pt"))
    
    if epoch % sampling_interval == 0:
      with torch.no_grad():
        ddim.model = model_fn_sampling
        ddim.imgshape = (4,3,model_config["image_size"],model_config["image_size"])
        sample,_ = ddim.sample_image(show_steps=False,save_images=False,eta=1.0)
        fig, axes = plt.subplots(1, 4, figsize=(20, 10))
        for i, ax in enumerate(axes.flat):
          img = 0.5 + 0.5 * sample[i].to('cpu')
          img = img.permute(1, 2, 0).clamp(0, 1).numpy()
          ax.imshow(img)
          ax.axis("off")

        plt.tight_layout()
        save_path = os.path.join(samples_dir, f"epoch_{epoch}.png")
        plt.savefig(save_path)
        ddim.imgshape = (batch_size,3,model_config["image_size"],model_config["image_size"])
        ddim.model = model_fn

if __name__ == "__main__":
  app.run(main)
