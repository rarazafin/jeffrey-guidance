from absl import app
from absl import flags
from ml_collections.config_flags import config_flags
import logging

import gc
import io
import os
import time
import sys

import numpy as np
import yaml
#import tensorflow as tf
#import tensorflow_gan as tfgan

# Keep the import below for registering all model definitions
from models.cifar.model import UNet
from models.ffhq.unet import create_model
from models.ffhq import dist_util

import tensorflow as tf
import tensorflow_gan as tfgan
import evaluation
import torch
from torch.utils import tensorboard
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
from torch.distributions.bernoulli import Bernoulli

import kagglehub
from torchvision import datasets, transforms
from dataset import Dataset
from torchvision.datasets import CIFAR10

from diffusion.Diffusion import DDIM
from utils import enable_dropout

from itertools import combinations

import pickle

FLAGS = flags.FLAGS

flags.DEFINE_string("config", None, "config file name")
flags.DEFINE_string("workdir", None, "Work directory.")
flags.DEFINE_string("ckpt",None,"Checkpoint")
flags.DEFINE_string("eval_folder", "eval",
                    "The folder name for storing evaluation results")
flags.DEFINE_string("loss_folder", None, "This folder name for storing loss results")
flags.DEFINE_integer("n_ensembles", 0, "Number of ensembles")
flags.DEFINE_integer("batch_size", 1024, "Sample batch size")
flags.DEFINE_enum("ensemble_type", "none", ["none", "mc_dropout", "deep_ensemble", "laplace"], "Type of ensemble")
flags.DEFINE_string("agg", "arithmetic", "Type of aggregation")
flags.DEFINE_boolean("consistent_dropout",False,"Decide if Consistent MC dropout or not")
flags.DEFINE_boolean("force_dropout", False, "True to force dropout during inference")
flags.DEFINE_integer("num_sampling_timesteps", 100, "Number of sampling steps")
flags.DEFINE_integer("num_samples", 50000, "Number of samples")
flags.DEFINE_float("eta", 0., "eta")
flags.DEFINE_integer("seed", 0, "seed for ensemble of size 2")
flags.DEFINE_boolean("resume", False, "Resume loss or not")
flags.mark_flags_as_required(["workdir", "config", "batch_size"])

def main(argv):
  
  #torch.use_deterministic_algorithms(True)
  
  ensemble_type = FLAGS.ensemble_type
  n_ensembles = FLAGS.n_ensembles
  agg = FLAGS.agg
  if (n_ensembles == 0 and ensemble_type != "none"):
      raise ValueError(f"For {ensemble_type}, n_ensembles must be a positive integer.")
  
  workdir = FLAGS.workdir
  eval_folder = FLAGS.eval_folder
  eval_dir = os.path.join(workdir, eval_folder)
  
  model_config_name = FLAGS.config
  with open(os.path.join(workdir, model_config_name), "r") as file:
      model_config = yaml.safe_load(file)
  
  num_samples = FLAGS.num_samples
  ckpt = FLAGS.ckpt
  checkpoint_dir = os.path.join(workdir, "checkpoints")
  
  torch.manual_seed(0)
  tf.io.gfile.makedirs(eval_dir)

  batch_size = FLAGS.batch_size

  num_sampling_timesteps = FLAGS.num_sampling_timesteps
  
  force_dropout = FLAGS.force_dropout
  consistent_dropout = FLAGS.consistent_dropout
  
  seed = FLAGS.seed

  resume = FLAGS.resume

  device = "cuda"
  
  yaml_name = os.path.basename(model_config_name)
  dataset_name = os.path.splitext(yaml_name)[0]    
  if ensemble_type != "deep_ensemble":
    checkpoint_file = f"{ckpt}.pt"
    if dataset_name == "cifar10":
      model = UNet(T=model_config["T"], ch=model_config["channel"], ch_mult=model_config["channel_mult"], attn=model_config["attn"],
                num_res_blocks=model_config["num_res_blocks"], dropout=model_config["dropout"])
      model = model.to(device)
      if torch.cuda.device_count() > 1:
        logging.info(f"Using {torch.cuda.device_count()} GPUs!")
        model = torch.nn.DataParallel(model)
         
      logging.info(f"Loading {checkpoint_file} ...")
        
      model.load_state_dict(torch.load(os.path.join(
            checkpoint_dir, checkpoint_file), map_location=device))
    elif dataset_name == "ffhq":
      logging.info(f"Loading {checkpoint_file} ...")
      model_config["model_path"] = os.path.join(checkpoint_dir,checkpoint_file) # load checkpoint ffhq
      #model_config["dropout"] = 0.1
      model = create_model(**model_config)
      model = model.to(device)
      if torch.cuda.device_count() > 1:
        logging.info(f"Using {torch.cuda.device_count()} GPUs!")
        model = torch.nn.DataParallel(model)
    else:
      raise NotImplementedError
    
    
    model.eval();
    if ensemble_type == "mc_dropout": enable_dropout(model)
    
    model_fn = model
    
  else:
    model_fn = []
    
    generator = torch.Generator()
    generator.manual_seed(seed)

    k_list = torch.randperm(4, generator=generator)[:n_ensembles] + 1
    logging.info(f"Ensemble list: {k_list}")

    
    for k in range(n_ensembles):

      if FLAGS.ckpt is not None:
        checkpoint_file = f"{FLAGS.ckpt}.pt" # if ckpt in argument
      else:
        checkpoint_file = f"ckpt_{dataset_name}_num_{k_list[k]}.pt" # if no ckpt in argument
      ckpt = "cifar_ddim_ckpt" if dataset_name == "cifar10" else "ffhq_ddim_ckpt"
        
      
      if dataset_name == "cifar10":
        model = UNet(T=model_config["T"], ch=model_config["channel"], ch_mult=model_config["channel_mult"], attn=model_config["attn"],
                num_res_blocks=model_config["num_res_blocks"], dropout=model_config["dropout"])
        model = model.to(device)
        if torch.cuda.device_count() > 1:
          logging.info(f"Using {torch.cuda.device_count()} GPUs!")
          model = torch.nn.DataParallel(model)
          
        
        if ensemble_type == "deep_ensemble":
          logging.info(f"Loading {checkpoint_file} ...")
          model.load_state_dict(torch.load(os.path.join(
            checkpoint_dir, checkpoint_file), map_location=device))
        
      elif dataset_name == "ffhq":
        logging.info(f"Loading {checkpoint_file} ...")
        model_config["model_path"] = os.path.join(checkpoint_dir,checkpoint_file)
        model_config["dropout"] = 0.
        model = create_model(**model_config)
        model = model.to(device)
        
        if torch.cuda.device_count() > 1:
          logging.info(f"Using {torch.cuda.device_count()} GPUs!")
          model = torch.nn.DataParallel(model)
         
      else:
        raise NotImplementedError
      
      model.eval()
      model_fn.append(model)
      '''
      if dataset_name == "cifar10":
        model_fn.append(lambda x, t: model(x,x.new_ones([x.shape[0], ], dtype=torch.long,device=device) * t))
      elif dataset_name == "ffhq":
        model_fn.append(lambda x, t: model(x, torch.tensor(t,device=device).unsqueeze(0).repeat(batch_size))[:,:3,:,:])
        # we repeat batch_size times, else dataparallel does not gather the results from each gpus
      else:
        raise NotImplementedError
      '''    
  

  sampling_shape = (batch_size,
                      3,
                      model_config["image_size"], model_config["image_size"])
  
  
  ddim = DDIM(model_fn,imgshape=sampling_shape)
  
  if ensemble_type == "mc_dropout" or force_dropout:
    dropout_rate = model_config["dropout"]
    logging.info("Dropout p={} enabled during inference.".format(dropout_rate))
    logging.info("Number of ensembles = {}.".format(n_ensembles))
  elif ensemble_type == "deep_ensemble":
    assert n_ensembles == len(model_fn)
    logging.info("Deep ensemble.")
    logging.info("Number of ensembles = {}.".format(n_ensembles))
  else:
    logging.info("No ensemble and perhaps no dropout.")

  if ensemble_type == "none":
    losses_dir = os.path.join(eval_dir, f"checkpoint_{ckpt}_size_{num_sampling_timesteps}_seed_{seed}")
  else:
    losses_dir = os.path.join(eval_dir, f"checkpoint_{ckpt}_{ensemble_type}_{n_ensembles}_size_{num_sampling_timesteps}_{agg}_seed_{seed}") 
  if FLAGS.loss_folder is not None:
    losses_dir = os.path.join(eval_dir, FLAGS.loss_folder)

  tf.io.gfile.makedirs(losses_dir)
  logging.info(f"Eval dir: {losses_dir}")

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
    dataset = Dataset(path,model_config,train="test")
    scaler = dataset.get_data_scaler
    inverse_scaler = dataset.get_data_inverse_scaler
    dataset = dataset.get_dataset()
  dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=0, drop_last=True, pin_memory=True)
  
  total_loss = 0.
  with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
    for batch_image,_ in tqdmDataLoader:

        x0 = batch_image.to(device)
        loss = ddim.loss(x0,K=n_ensembles,agg=agg)

        tqdmDataLoader.set_postfix(ordered_dict={
                    "loss: ": loss.item()
         })
        total_loss += loss.item()
        total_samples += x0.numel()

    mean_loss = total_loss/total_samples

  logging.info(f"Mean loss: {mean_loss:.4f}")
  
  tf.io.gfile.makedirs(losses_dir)
  with tf.io.gfile.GFile(os.path.join(losses_dir, f"loss_seed_{seed}.npz"), "wb") as fout:
    io_buffer = io.BytesIO()
    np.savez_compressed(io_buffer, mean_loss=mean_loss)
    fout.write(io_buffer.getvalue())

   



if __name__ == "__main__":
  app.run(main)
