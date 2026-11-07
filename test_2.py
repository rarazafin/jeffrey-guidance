print("START")
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
from models.cifar.diffusion import Model
from models.cifar.ema import EMAHelper
from models.guided_diffusion.unet import create_model
from models.guided_diffusion import dist_util

import clip
from diffusers import UNet2DModel
from models.celeba.wrapper import Wrapper
import tensorflow as tf
import tensorflow_gan as tfgan
import evaluation
import torch
import torch.nn.functional as F
from torch.utils import tensorboard
import torchvision
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
from torch.distributions.bernoulli import Bernoulli

from diffusion.Diffusion import DDIM
from utils import enable_dropout

from functions.ckpt_util import get_ckpt_path
import yaml
from types import SimpleNamespace
from functools import partial

from itertools import combinations

import pickle

from inception import InceptionEmbedder
from ratio import MLP, get_classifier, log_ratio_from_attribute_classifier, log_ratio_joint_three_from_attribute_classifier, log_ratio_logsumexp_from_attribute_classifier, log_ratio_joint_from_attribute_classifier, log_ratio_match_from_attribute_classifier, log_ratio_balance_from_attribute_classifier, get_attribute_classifier
from ratio import log_ratio_balance_baseline_from_attribute_classifier
from clip_embedder import ClipEmbedder

FLAGS = flags.FLAGS

flags.DEFINE_string("config", None, "config file name")
flags.DEFINE_string("workdir", None, "Work directory.")
flags.DEFINE_string("ckpt",None,"Checkpoint")
flags.DEFINE_string("eval_folder", "eval",
                    "The folder name for storing evaluation results")
flags.DEFINE_string("fid_folder", None, "This folder name for storing FID results")
flags.DEFINE_integer("batch_size", 1024, "Sample batch size")
flags.DEFINE_integer("num_sampling_timesteps", 100, "Number of sampling steps")
flags.DEFINE_integer("num_samples", 50000, "Number of samples")
flags.DEFINE_float("eta", 0., "eta")
flags.DEFINE_integer("seed", 0, "seed for ensemble of size 2")
flags.DEFINE_float("guidance_scale", 0.0, "guidance scale")
flags.DEFINE_integer("delta", 1000, "delta for guidance")
flags.DEFINE_string("embedder", None, "embedding method")
flags.DEFINE_boolean("resume", False, "Resume sampling or not")
flags.DEFINE_boolean("fid_test", False, "Compute fid on test set")
flags.DEFINE_string("fairness", "base", "Type of fairness constraint: 'base', 'joint' or 'match'")
flags.DEFINE_string("classifier_ckpt", None, "Path to the attribute classifier checkpoint. If None, use the default one.")
flags.DEFINE_string("rho_table", None, "Path to the rho table pickle. If None, use rho_table_{dataset}_model_{fairness}.pkl.")
flags.mark_flags_as_required(["workdir", "config", "batch_size"])

fairness_log_ratio ={
  "base": log_ratio_logsumexp_from_attribute_classifier,
  "joint": log_ratio_joint_from_attribute_classifier,
  "joint_linear_probe": log_ratio_joint_from_attribute_classifier,
  "joint_makeup_lipstick": log_ratio_joint_from_attribute_classifier,
  "joint_three": log_ratio_joint_three_from_attribute_classifier,
  "match": log_ratio_match_from_attribute_classifier,
  "balance": log_ratio_balance_from_attribute_classifier,
  "balance baseline": log_ratio_balance_baseline_from_attribute_classifier,
}
def main(argv):
  
  #torch.use_deterministic_algorithms(True)
  
  workdir = FLAGS.workdir
  eval_folder = FLAGS.eval_folder
  eval_dir = os.path.join(workdir, eval_folder)
 
  def ns(d): return SimpleNamespace(**{k: ns(v) if isinstance(v, dict) else v for k,v in d.items()})
  model_config_name = FLAGS.config
  with open(os.path.join(workdir, model_config_name), "r") as file:
      model_config = yaml.safe_load(file)
  
  num_samples = FLAGS.num_samples
  ckpt = FLAGS.ckpt
  checkpoint_dir = os.path.join(workdir, "checkpoints")

  classifier_ckpt = FLAGS.classifier_ckpt

  seed = FLAGS.seed
  torch.manual_seed(seed)
  tf.io.gfile.makedirs(eval_dir)

  embedder = FLAGS.embedder

  batch_size = FLAGS.batch_size
  num_sampling_timesteps = FLAGS.num_sampling_timesteps
  skip_type = "uniform" if num_sampling_timesteps < 1000 else None

  
  eta = FLAGS.eta
  guidance_scale = FLAGS.guidance_scale
  delta = FLAGS.delta
  
  resume = FLAGS.resume

  fairness = FLAGS.fairness
  logging.info(f"Fairness setting: {fairness}")
  fairness_log_ratio_from_classifier = fairness_log_ratio[fairness]
  logging.info(fairness_log_ratio_from_classifier)
  device = "cuda"
  
  yaml_name = os.path.basename(model_config_name)
  dataset_name = os.path.splitext(yaml_name)[0]    

  checkpoint_file = f"{ckpt}.pt"
  if dataset_name == "cifar10":
    ckpt = get_ckpt_path(f"ema_cifar10")
    model = Model(ns(model_config))
    logging.info(f"Loading {ckpt}")
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model = model.to(device)
    if torch.cuda.device_count() > 1:
      logging.info(f"Using {torch.cuda.device_count()} GPUs!")
      model = torch.nn.DataParallel(model)  
    
  elif dataset_name == "ffhq":
    logging.info(f"Loading {checkpoint_file} ...")
    model_config["model_path"] = os.path.join(checkpoint_dir,checkpoint_file) # load checkpoint ffhq
    #model_config["dropout"] = 0.1
    model = create_model(**model_config)
    model = model.to(device)
    if torch.cuda.device_count() > 1:
      logging.info(f"Using {torch.cuda.device_count()} GPUs!")
      model = torch.nn.DataParallel(model)
  elif dataset_name == "celeba":
    logging.info(f"Loading google/ddpm-celebahq-256 ...")
    model = UNet2DModel.from_pretrained("google/ddpm-celebahq-256")
    model = model.to(device)
    model = Wrapper(model)
    ''''
    if torch.cuda.device_count() > 1:
      logging.info(f"Using {torch.cuda.device_count()} GPUs!")
      model = torch.nn.DataParallel(model)
    '''
  elif dataset_name == "celeba_p2":
    logging.info(f"Loading {checkpoint_file} ...")
    model_config["model_path"] = os.path.join(checkpoint_dir,checkpoint_file) # load checkpoint ffhq
    model = create_model(**model_config)
    model = model.to(device)
    '''
    if torch.cuda.device_count() > 1:
      logging.info(f"Using {torch.cuda.device_count()} GPUs!")
      model = torch.nn.DataParallel(model)
    '''
  else:
    raise NotImplementedError


  model.eval()
    
  model_fn = model
  
  def print_gpu_memory():
    allocated = torch.cuda.memory_allocated() / 1024**2  # MB
    reserved = torch.cuda.memory_reserved() / 1024**2  # MB
    logging.info(f"Mémoire allouée : {allocated:.2f} MB, Mémoire réservée : {reserved:.2f} MB")

  print_gpu_memory()

  sampling_shape = (batch_size,
                      3,
                      model_config["image_size"], model_config["image_size"])
  
  
  ddim = DDIM(model_fn,dataset_name=dataset_name,imgshape=sampling_shape,device=device)
  
  this_sample_dir = os.path.join(eval_dir, f"checkpoint_{ckpt}_size_{num_sampling_timesteps}")

  if FLAGS.fid_folder is not None:
    this_sample_dir = os.path.join(eval_dir, FLAGS.fid_folder)

  tf.io.gfile.makedirs(this_sample_dir)
  logging.info(f"Eval dir: {this_sample_dir}")
  # Use inceptionV3 for images with resolution higher than 256.


  log_r_fn = None
  if guidance_scale > 0.0:

    # CLASSIFIER LOADING
    if classifier_ckpt is None:
      if fairness in ["joint_three", "joint_makeup_lipstick", "joint_linear_probe"]:
        raise ValueError(f"Fairness setting '{fairness}' requires a classifier checkpoint. Please provide one using the --classifier_ckpt flag.")
      classifier_model, alpha = get_attribute_classifier(f"celeba_young_male_{fairness}_classifier_with_alpha.pth", fairness, device) # ICI JOINT ET MATCH C PAREIL
    else:
      logging.info(f"Loaded classifier from command: {classifier_ckpt}")
      classifier_model, alpha = get_attribute_classifier(classifier_ckpt, fairness, device)
    classifier_model.eval()
    logging.info(f"Loaded alpha={alpha:.4f}")
    classifier_fn = lambda x: alpha * classifier_model(
      F.interpolate(x, size=(224,224), mode='bilinear', align_corners=False)
    )
    # RHO TABLE LOADING
    if fairness != "balance baseline":
      rho_table_path = FLAGS.rho_table or f"rho_table_{dataset_name}_model_{fairness}.pkl"
      logging.info(f"Loaded rho table from: {rho_table_path}")
      with open(rho_table_path, "rb") as f:
        rho_table = pickle.load(f)
      logging.info(rho_table)
      log_r_fn = partial(
            fairness_log_ratio_from_classifier,
            classifier_fn=classifier_fn,
            rho_table=rho_table
      )
  else:
    logging.info("guidance_scale=0: no classifier/log_r_fn loaded, sampling unconditionally.")

  inceptionv3 = model_config["image_size"] >= 256
  #inception_model = evaluation.get_inception_model(inceptionv3=inceptionv3)

  num_sampling_rounds = num_samples // batch_size + 1

  def compute_statistics(dataset_name="cifar10"):
    # Compute inception scores, FIDs and KIDs.
    # Load all statistics that have been previously computed and saved for each host
    all_logits = []
    all_pools = []

    logging.info("starting to compute statistics.")
    this_sample_dir = os.path.join(eval_dir, f"checkpoint_{ckpt}_size_{num_sampling_timesteps}")
    if FLAGS.fid_folder is not None:
      this_sample_dir = os.path.join(eval_dir, FLAGS.fid_folder)
    stats = tf.io.gfile.glob(os.path.join(this_sample_dir, "statistics_*.npz"))
 
    
    for stat_file in stats:
      with tf.io.gfile.GFile(stat_file, "rb") as fin:
        stat = np.load(fin)
        if not inceptionv3:
          all_logits.append(stat["logits"])
        all_pools.append(stat["pool_3"])
    if not inceptionv3:
      all_logits = np.concatenate(all_logits, axis=0)[:num_samples] # 50k by default
    all_pools = np.concatenate(all_pools, axis=0)[:num_samples] # 50k by default
    logging.info(f"Number of samples: {all_pools.shape[0]}")
    # Load pre-computed dataset statistics.
    data_stats = evaluation.load_dataset_stats(dataset_name)
    data_pools = data_stats["pool_3"][:50000]

    # Compute FID/KID/IS on all samples together.
    logging.info("computing IS.")
    if not inceptionv3:
      inception_score = tfgan.eval.classifier_score_from_logits(all_logits)
    else:
      inception_score = -1
    logging.info("computing FID.")
    fid = tfgan.eval.frechet_classifier_distance_from_activations(data_pools, all_pools)
    # Hack to get tfgan KID work for eager execution.
    tf_data_pools = tf.convert_to_tensor(data_pools)
    tf_all_pools = tf.convert_to_tensor(all_pools)
    logging.info("computing KID.")
    kid = tfgan.eval.kernel_classifier_distance_from_activations(
        tf_data_pools, tf_all_pools).numpy()
    del tf_data_pools, tf_all_pools

    logging.info("ckpt-%s --- inception_score: %.6e, FID: %.6e, KID: %.6e" % (
          ckpt, inception_score, fid, kid))


    report_file = os.path.join(this_sample_dir, f"report_{num_samples}.npz")
  
    with tf.io.gfile.GFile(report_file,"wb") as f:
      io_buffer = io.BytesIO()
      np.savez_compressed(io_buffer, IS=inception_score, fid=fid, kid=kid)
      f.write(io_buffer.getvalue())

  def compute_FID_pytorch(this_sample_dir):
    f_inception = InceptionEmbedder(implementation="fid-pytorch",device=device)
    f_inception.eval()
    images = tf.io.gfile.glob(os.path.join(this_sample_dir, "samples_*.npz"))
    pools_f = []
    micro_bs = 1 if embedder == "clip" else 64  # pil_imageS does not support micro-batching, so we set micro_bs=1 for CLIP to avoid errors. Else:  16/32/64.
    logging.info("starting to compute statistics.")
    with torch.no_grad():
      for image_file in tqdm(images):
        arr = np.load(image_file)["samples"]          # (N, 32, 32, 3) uint8
        img = torch.from_numpy(arr).permute(0, 3, 1, 2).float().div_(255.0)  # CPU float32

        
        # micro-batching -> GPU
        for i in range(0, img.size(0), micro_bs):
            sub = img[i:i+micro_bs].to(device, non_blocking=True)
            emb = f_inception(sub).detach().cpu().numpy()
            #emb = f_embedder(preprocess(pil_images(sub.squeeze(0))).unsqueeze(0).to(device)).detach().cpu().numpy()
            pools_f.append(emb)

        del img, arr
        torch.cuda.empty_cache()
    
    pools_f = np.concatenate(pools_f, axis=0)[:num_samples]
    logging.info(f"Shape of pools_f {pools_f.shape}.")
    np.save(os.path.join(this_sample_dir, f"pools_f_{dataset_name}_final.npy"), pools_f)
    test = ""
    if FLAGS.fid_test:
      logging.info("fid is computed on test set")
      test = "_test"

    if num_samples != 5000:
      logging.info(f"stat file: assets/stats/pools_r_{dataset_name}{test}_final.npy")
      pools_r = np.load(f"assets/stats/pools_r_{dataset_name}{test}_final.npy")#[:num_samples]
    if num_samples == 5000:
      logging.info(f"stat file: assets/stats/pools_r_{dataset_name}_balanced_final.npy")
      pools_r =np.load(f"assets/stats/pools_r_{dataset_name}_balanced_final.npy")
      
    logging.info(f"Computing fid ...")
    fid = tfgan.eval.frechet_classifier_distance_from_activations(pools_r, pools_f)

    logging.info(f"ckpt-{ckpt} --- fid: {fid:.6e}")

    del pools_f,pools_r
    report_file = os.path.join(this_sample_dir, f"report_pytorch{test}_{num_samples}.npz")
  
    with tf.io.gfile.GFile(report_file,"wb") as f:
      io_buffer = io.BytesIO()
      np.savez_compressed(io_buffer, fid=fid)
      f.write(io_buffer.getvalue())

  


  # Main loop to compute metrics
  start_round = 0
  round_checkpoint_file = os.path.join(this_sample_dir, "round_checkpoint.npz")
  if os.path.exists(round_checkpoint_file) and resume:
    checkpoint_data = np.load(round_checkpoint_file)
    start_round = int(checkpoint_data["round"])
    if num_samples == 5000: start_round = num_sampling_rounds
    if start_round == num_sampling_rounds-1: start_round += 1  
   
  for r in range(start_round,num_sampling_rounds):
    logging.info("sampling -- ckpt: {}, gs: {}, delta: {}, round: {}/{}".format(ckpt, guidance_scale, delta, r+1, num_sampling_rounds))
    np.savez(round_checkpoint_file, round=r)
    # Directory to save samples. Different for each host to avoid writing conflicts
    if guidance_scale > 0.0 and fairness == "balance baseline":
      a = torch.bernoulli(0.5 * torch.ones(batch_size, device=device)).long()
      log_r_fn = partial(
          fairness_log_ratio_from_classifier,
          classifier_fn=classifier_fn,
          a=a
      )

    samples, _ = ddim.sample_jeffrey(eta=eta,
                                  num_diffusion_timesteps=num_sampling_timesteps,
                                  seed=r,
                                  dt_q=None,
                                  skip_type=skip_type,
                                  log_r_fn=log_r_fn,
                                  f_embedder=None,  # set to None to disable embedding and use x0_hat_dt directly for log_r_fn
                                  guidance_scale=guidance_scale,
                                  delta=delta,
                                  show_steps=False,
                                  save_images=False
                                 )
    samples = 0.5 + 0.5 * samples # put in [0,1]
    samples = np.clip(samples.permute(0, 2, 3, 1).cpu().numpy() * 255., 0, 255).astype(np.uint8)
    samples = samples.reshape(
        (-1, model_config["image_size"], model_config["image_size"], 3))
    # Write samples to disk or Google Cloud Storage
    with tf.io.gfile.GFile(os.path.join(this_sample_dir, f"samples_{r}.npz"), "wb") as fout:
      io_buffer = io.BytesIO()
      np.savez_compressed(io_buffer, samples=samples)
      fout.write(io_buffer.getvalue())

    # Force garbage collection before calling TensorFlow code for Inception network
    '''
    gc.collect()
    latents = evaluation.run_inception_distributed(samples, inception_model,
                                                       inceptionv3=inceptionv3)
    # Force garbage collection again before returning to JAX code
    gc.collect()
    # Save latent represents of the Inception network to disk or Google Cloud Storage
    with tf.io.gfile.GFile(os.path.join(this_sample_dir, f"statistics_{r}.npz"), "wb") as fout:
      io_buffer = io.BytesIO()
      np.savez_compressed(io_buffer, pool_3=latents["pool_3"], logits=latents["logits"])
      fout.write(io_buffer.getvalue())
    ''' 
  del log_r_fn
  gc.collect()
  torch.cuda.empty_cache()
  compute_FID_pytorch(this_sample_dir)



if __name__ == "__main__":
  app.run(main)
