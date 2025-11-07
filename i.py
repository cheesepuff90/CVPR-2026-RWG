#!/usr/bin/env python3
"""
imagenet_weighted_embeddings.py

Heavy-lifting pipeline for weighted normative embeddings.
Provides:
  - run_seed_pipeline
  - run_mri_weighting
  - compare_outliers
"""
from __future__ import annotations
from typing import List, Dict, Any, Optional, Iterable, Tuple
import os, tempfile
import contextlib, io
import re
import csv
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from PIL import Image
from torchvision.datasets.folder import default_loader
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
from torchvision.models._api import get_model, get_weight
from torchvision.models import get_model_weights

import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import torchvision.datasets as Datasets
import torchvision.models as Models
from torchvision.datasets import ImageFolder

import numpy as np
from pathlib import Path
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.pairwise import cosine_distances
from sklearn.model_selection import KFold
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd


# Import CLIP and DINO
import clip
from torchvision.models.feature_extraction import create_feature_extractor
try:
    import torch.hub
    # Check if we can access hub models
    torch.hub.list('facebookresearch/dino:main')
    HAS_DINO = True
except:
    HAS_DINO = False
#     print("Warning: Unable to access torch.hub for DINO models")

try:
    import timm
    from timm.data import resolve_model_data_config, create_transform
    HAS_TIMM = True
except Exception:
    HAS_TIMM = False


import shutil, gc
from contextlib import contextmanager

def _hf_hotpatch_from_env():
    """Force huggingface_hub to use env-driven cache dirs even if imported earlier."""
    try:
        import huggingface_hub.constants as hf_c
        hh = os.environ.get("HUGGINGFACE_HUB_CACHE")
        hf = os.environ.get("HF_HOME")
        tr = os.environ.get("TRANSFORMERS_CACHE")
        if hh: hf_c.HF_HUB_CACHE = Path(hh)
        if hf: hf_c.HF_HOME      = Path(hf)
        if tr: hf_c.TRANSFORMERS_CACHE = Path(tr)
        # some versions also read a copy in file_download
        try:
            import huggingface_hub.file_download as fd
            if hh: fd.HF_HUB_CACHE = Path(hh)
        except Exception:
            pass
    except Exception:
        pass

@contextmanager
def per_model_tmpcache(tag: str, base: Path | None = None, min_free_gb: int = 15):
    """
    Route *all* caches (HF/timm/torch/xdg) into a per-model dir and nuke it afterwards.
    Also hot-patch HF constants so paths really switch per model.
    """
    # choose a base with space: env beats arg; default to HOME (avoid /tmp)
    base_root = (
        Path(os.environ["SEEDS_TMPCACHE_ROOT"]).expanduser()
        if "SEEDS_TMPCACHE_ROOT" in os.environ else
        (Path(base).expanduser() if base is not None else Path.home() / "seeds_tmpcache")
    )
    base_root.mkdir(parents=True, exist_ok=True)

    # check free space before downloading multi-GB weights
    try:
        free = shutil.disk_usage(str(base_root)).free
        if free < min_free_gb * (1 << 30):
            raise OSError(28, f"Not enough free space in {base_root} (<{min_free_gb} GB).")
    except Exception:
        # best-effort; don’t hard-fail on weird filesystems
        pass

    slug = re.sub(r'[^a-zA-Z0-9._-]+', '_', str(tag))
    root = base_root / ".tmpcache" / slug
    hf_dir, torch_dir, timm_dir = root / "hf", root / "torch", root / "timm"
    for d in (hf_dir, torch_dir, timm_dir):
        d.mkdir(parents=True, exist_ok=True)

    keys = (
        "HF_HOME", "HUGGINGFACE_HUB_CACHE", "HF_HUB_CACHE", "TRANSFORMERS_CACHE",
        "TIMM_CACHE_DIR", "XDG_CACHE_HOME", "TORCH_HOME", "PYTORCH_HUB_DIR", "TMPDIR"
    )
    backup = {k: os.environ.get(k) for k in keys}

    os.environ["HF_HOME"]               = str(hf_dir)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hf_dir / "hub")
    os.environ["HF_HUB_CACHE"]          = str(hf_dir / "hub")
    os.environ["TRANSFORMERS_CACHE"]    = str(hf_dir / "transformers")
    os.environ["TIMM_CACHE_DIR"]        = str(timm_dir)
    os.environ["XDG_CACHE_HOME"]        = str(root)
    os.environ["TORCH_HOME"]            = str(torch_dir)
    os.environ["PYTORCH_HUB_DIR"]       = str(torch_dir / "hub")
    os.environ["TMPDIR"]                = str(root) 

    _hf_hotpatch_from_env()

    try:
        yield
    finally:
        gc.collect()
        try:
            shutil.rmtree(root, ignore_errors=True)
        finally:
            for k, v in backup.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v



import torch.nn.functional as F
from torch import amp
use_amp = torch.cuda.is_available()
prefer_bf16 = torch.cuda.is_bf16_supported()
amp_dtype = torch.bfloat16 if prefer_bf16 else torch.float16

use_scaler = use_amp and (amp_dtype == torch.float16)
scaler = amp.GradScaler("cuda", enabled=use_scaler)

def cleanup_distributed():
    """Clean up distributed training"""
    dist.destroy_process_group()

@contextlib.contextmanager
def _suppress_hub_output():
    sink_out, sink_err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(sink_out), contextlib.redirect_stderr(sink_err):
        yield

def _load_dino(model_short: str):
    name_map = {
        'vits16': 'dino_vits16',
        'vits8' : 'dino_vits8',
        'vitb16': 'dino_vitb16',
        'vitb8' : 'dino_vitb8',
    }
    hub_id = name_map.get(model_short)
    if hub_id is None:
        raise ValueError(f"Unknown DINO model: {model_short}")
    with _suppress_hub_output():
        return torch.hub.load('facebookresearch/dino:main', hub_id)
    
    
# NEW: Function to implement the Black Swan identification and validation logic.
# This modular function takes the results of the iterative weighting and applies
# the statistical thresholding and validation steps from the user's pseudocode.
def identify_black_swans(
    weights: np.ndarray,
    embeddings: np.ndarray,
    converged_mean_rdm: np.ndarray,
    seed_ids: list[int],
    z_thresh: float = -1.0
) -> dict:
    """
    Identifies Black Swan networks based on converged weights and validates them.

    Args:
        weights: The final, converged weights for each seed.
        embeddings: The normalized RDM vectors for each seed [n_seeds, n_distances].
        converged_mean_rdm: The final, converged group-average RDM vector.
        seed_ids: A list of the seed IDs (e.g., [0, 1, 2, ...]).

    Returns:
        A dictionary containing the list of Black Swan seeds, their data,
        and the threshold used.
    """
    if len(weights) < 2: # Cannot compute stats on a single seed
        return {
            "black_swan_indices": [],
            "black_swan_details": {},
            "threshold": [],
            "mean_weight": np.nan,
            "std_weight": np.nan,
        }
    
    print(weights)

    # NEW: Define threshold based on the �w - 1.5�w rule.
    mu_w = np.mean(weights)
    # sigma_w = np.std(weights)
    # threshold_low = mu_w - 1.5 * sigma_w
    # threshold_high = mu_w + 1.5 * sigma_w

    sigma_w = np.std(weights, ddof=1) + 1e-9
    z_scores = (weights - mu_w) / sigma_w

    black_swan_indices = np.where(z_scores < z_thresh)[0]

    # NEW: Flag the seeds that fall below the threshold.
    # black_swan_indices_1 = np.where(weights < threshold_low)[0]
    # black_swan_indices_2 = np.where(weights > threshold_high)[0]
    # black_swan_indices = np.concatenate((black_swan_indices_1, black_swan_indices_2))

    black_swan_details = {}
    
    # NEW: Validate by re-computing correlation for the flagged Black Swans.
    # This confirms they are indeed dissimilar to the final group average.
    for idx in black_swan_indices:
        seed_rdm_vec = embeddings[idx]
        
        # Calculate the Pearson correlation between the seed's RDM and the group mean RDM.
        # We use a robust, vectorized calculation for Pearson's r.
        seed_centered = seed_rdm_vec - seed_rdm_vec.mean()
        mean_centered = converged_mean_rdm - converged_mean_rdm.mean()
        
        numerator = np.dot(seed_centered, mean_centered)
        denominator = np.linalg.norm(seed_centered) * np.linalg.norm(mean_centered)
        
        # Avoid division by zero if a vector is constant
        if denominator == 0:
            validation_corr = 0.0
        else:
            validation_corr = numerator / denominator

        seed_id = seed_ids[idx]
        black_swan_details[seed_id] = {
            "weight": weights[idx],
            "validation_correlation": validation_corr,
        }
        
    return {
        "black_swan_indices": black_swan_indices,
        "black_swan_details": black_swan_details,
        # "threshold": [float(threshold_low), float(threshold_high)],
        "threshold": float(z_thresh),
        "mean_weight": mu_w,
        "std_weight": sigma_w,
    }

def build_text_prompts(labels: torch.Tensor, classnames: list[str], template: str = "a photo of a {}") -> list[str]:
    return [template.format(classnames[int(y)]) for y in labels.tolist()]

def clip_contrastive_loss(logits_per_image: torch.Tensor, logits_per_text: torch.Tensor) -> torch.Tensor:
    bsz = logits_per_image.size(0)
    target = torch.arange(bsz, device=logits_per_image.device)
    return 0.5 * (F.cross_entropy(logits_per_image, target) +
                  F.cross_entropy(logits_per_text,  target))


def run_seed_pipeline(
    arch_list: list[str],
    n_seeds: int,
    epochs: int,
    data_root: str | Path,
    dataset_type: str,
    device: str,
    ckpt_root: Path,
    batch_size: int = 256,
    learning_rate: float = 0.1,
    weight_decay: float = 1e-4,
    num_workers: int = 8,
    subsample_val_batches: int = 80,
    distributed: bool = False,
    local_rank: int = 0,
    world_size: int = 1,
    pca_dim: int = 50
) -> dict[str, dict[str, np.ndarray]]:

    def _save_ckpt(model, optimizer, scheduler, ckpt_path, **meta):
        to_save = model.module if isinstance(model, DDP) else model
        ckpt = {
            "state_dict": to_save.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
        }
        ckpt.update(meta)  # e.g., epoch, seed, arch, dataset_type, train_loss_epoch
        torch.save(ckpt, ckpt_path)

    ckpt_root.mkdir(parents=True, exist_ok=True)

    if distributed and torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
    device = torch.device(device)

    AMP_AVAILABLE = torch.cuda.is_available()
    amp_dtype = torch.bfloat16 if (hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported()) else torch.float16

    # Transforms
    if dataset_type in ("cifar10", "cifar100"):
        standard_transform = T.Compose([
            T.Resize(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        train_transform = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.Resize(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        num_classes = 10 if dataset_type == "cifar10" else 100
    else:
        standard_transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        train_transform = T.Compose([
            T.RandomResizedCrop(224),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    clip_transform = None
    results = {}

    for arch in arch_list:
        arch_ckpt_root = ckpt_root / arch
        if arch == 'clip_ViT-B/16':
            arch_ckpt_root = ckpt_root / 'clip_vit_b16'
        arch_ckpt_root.mkdir(parents=True, exist_ok=True)

        if dataset_type in ("cifar10", "cifar100"):
            if arch.startswith('clip_'):
                clip_model_name = arch[5:]
                if clip_transform is None:
                    _, clip_preprocess = clip.load(clip_model_name, device='cpu', jit=False)
                    clip_transform = clip_preprocess
                cls = Datasets.CIFAR10 if dataset_type == "cifar10" else Datasets.CIFAR100
                train_ds = cls(root=data_root, train=True, download=True, transform=clip_transform)
                val_ds   = cls(root=data_root, train=False, download=True, transform=clip_transform)
            else:
                cls = Datasets.CIFAR10 if dataset_type == "cifar10" else Datasets.CIFAR100
                train_ds = cls(root=data_root, train=True,  download=True, transform=train_transform)
                val_ds   = cls(root=data_root, train=False, download=True, transform=standard_transform)
        else:
            val_dir = Path(data_root) / 'val'
            train_dir = Path(data_root) / 'train'
            if arch.startswith('clip_'):
                clip_model_name = arch[5:]
                if clip_transform is None:
                    _, clip_preprocess = clip.load(clip_model_name, device='cpu', jit=False)
                    clip_transform = clip_preprocess
                train_ds = Datasets.ImageFolder(train_dir, transform=clip_transform)
                val_ds   = Datasets.ImageFolder(val_dir,   transform=clip_transform)
            else:
                train_ds = Datasets.ImageFolder(train_dir, transform=train_transform)
                val_ds   = Datasets.ImageFolder(val_dir,   transform=standard_transform)

        # Samplers / loaders
        train_sampler = None
        val_sampler = None
        if distributed:
            train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=local_rank)
            val_sampler   = DistributedSampler(val_ds,   num_replicas=world_size, rank=local_rank, shuffle=False)

        eff_bs = batch_size // world_size if distributed else batch_size
        train_loader = DataLoader(
            train_ds, batch_size=eff_bs, shuffle=(train_sampler is None),
            num_workers=num_workers, sampler=train_sampler,
            pin_memory=True, persistent_workers=(num_workers > 0),
        )
        val_loader = DataLoader(
            val_ds, batch_size=eff_bs, shuffle=False,
            num_workers=num_workers, sampler=val_sampler,
            pin_memory=True, persistent_workers=(num_workers > 0),
        )

        layer_names = []
        seed_vecs_per_layer = {}

        # for seed in range(n_seeds):
        for seed in range(10, 30):
            torch.manual_seed(seed); np.random.seed(seed)

            seed_ckpt_dir = arch_ckpt_root / f"seed_{seed}"
            seed_ckpt_dir.mkdir(parents=True, exist_ok=True)

            if arch.startswith('clip_'):
                clip_model_name = arch[5:]
                model, _ = clip.load(clip_model_name, device='cpu', jit=False)

                model = model.float()
                for p in model.parameters():
                    if p.dtype.is_floating_point:
                        p.data = p.data.float()
                with torch.no_grad():
                    model.logit_scale.data = model.logit_scale.data.float()

                model = model.to(device).train()

                if distributed:
                    ddp_kwargs = {} if device.type != 'cuda' else {"device_ids": [local_rank]}
                    model = DDP(model, **ddp_kwargs)
                base = model.module if isinstance(model, DDP) else model

                base_lr = min(learning_rate, 5e-4)
                enc_lr  = min(base_lr, 1e-5)
                opt = torch.optim.AdamW(
                    [
                        {"params": [p for n, p in base.named_parameters() if "logit_scale" not in n and "visual" in n],      "lr": enc_lr},
                        {"params": [p for n, p in base.named_parameters() if "logit_scale" not in n and "visual" not in n], "lr": enc_lr},
                        {"params": [base.logit_scale], "lr": enc_lr},
                    ],
                    lr=enc_lr, weight_decay=weight_decay
                )
                sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

                use_amp_clip = False
                scaler = amp.GradScaler(enabled=(use_amp_clip and amp_dtype == torch.float16))

                classnames = train_ds.classes
                text_template = "a photo of a {}"
                LOGIT_SCALE_MAX = float(np.log(100.0))

                for epoch in range(epochs):
                    if distributed and (train_sampler is not None):
                        train_sampler.set_epoch(epoch)

                    model.train()
                    running = 0.0
                    epoch_loss_sum = 0.0
                    num_batches = 0

                    for i, (images, labels) in enumerate(train_loader):
                        images = images.to(device, non_blocking=True)
                        labels = labels.to(device, non_blocking=True)

                        texts = build_text_prompts(labels, classnames, template=text_template)
                        token_ids = clip.tokenize(texts, context_length=77).to(device, non_blocking=True)

                        opt.zero_grad(set_to_none=True)

                        with amp.autocast(device_type='cuda', dtype=amp_dtype, enabled=(use_amp_clip and AMP_AVAILABLE)):
                            image_features = base.encode_image(images)
                            text_features  = base.encode_text(token_ids)

                            image_features = F.normalize(image_features, dim=-1, eps=1e-6)
                            text_features  = F.normalize(text_features,  dim=-1, eps=1e-6)

                            with torch.no_grad():
                                base.logit_scale.data.clamp_(0, LOGIT_SCALE_MAX)
                            logit_scale = base.logit_scale.exp()

                            logits_per_image = logit_scale * image_features @ text_features.t()
                            logits_per_text  = logits_per_image.t()

                            if not torch.isfinite(logits_per_image).all():
                                print("logit_scale:", float(logit_scale))
                                print("img feat finite:", bool(torch.isfinite(image_features).all()))
                                print("txt feat finite:", bool(torch.isfinite(text_features).all()))
                                raise RuntimeError("Non-finite logits encountered")

                            loss = clip_contrastive_loss(logits_per_image, logits_per_text)

                        if scaler.is_enabled():
                            scaler.scale(loss).backward()
                            scaler.unscale_(opt)
                            torch.nn.utils.clip_grad_norm_(base.parameters(), max_norm=1.0)
                            scaler.step(opt)
                            scaler.update()
                        else:
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(base.parameters(), max_norm=1.0)
                            opt.step()

                        with torch.no_grad():
                            base.logit_scale.data.clamp_(0, LOGIT_SCALE_MAX)

                        # logging
                        loss_val = float(loss)
                        running += loss_val
                        epoch_loss_sum += loss_val
                        num_batches += 1

                        rank_prefix = f"[R{local_rank}] " if distributed else ""
                        if (i + 1) % 100 == 0:
                            msg = f"{rank_prefix}[{arch} seed{seed}] epoch {epoch+1} iter {i+1} loss {running/100:.3f}"
                            if not distributed or local_rank == 0:
                                print(msg)
                            running = 0.0

                    sched.step()

                    if (epoch + 1) % 10 == 0 or (epoch + 1) == epochs:
                    # save every epoch (rank 0 only)
                        if (not distributed) or (local_rank == 0):
                            arch_name = 'clip_vit_b16' if arch == 'clip_ViT-B/16' else arch
                            ckpt_path = seed_ckpt_dir / f"{arch_name}_epoch{epoch + 1}.pth"
                            avg_epoch_loss = epoch_loss_sum / max(1, num_batches)
                            _save_ckpt(
                                model=model,
                                optimizer=opt,
                                scheduler=sched,
                                ckpt_path=ckpt_path,
                                arch=arch_name,
                                seed=seed,
                                epoch=epoch + 1,
                                dataset_type=dataset_type,
                                train_loss_epoch=float(avg_epoch_loss),
                            )
                            print(f"Saved checkpoint: {ckpt_path} (train_loss_epoch={avg_epoch_loss:.6f})")

            else:
                # torchvision models
                if dataset_type in ("cifar10", "cifar100"):
                    if arch == 'alexnet':
                        model = Models.alexnet(weights=None)
                        model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, num_classes)
                    elif arch == 'resnet50':
                        model = Models.resnet50(weights=None)
                        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
                    elif arch == 'vit_b_16':
                        model = Models.vit_b_16(weights=None)
                        model.heads.head = torch.nn.Linear(model.heads.head.in_features, num_classes)
                    else:
                        model = getattr(Models, arch)(weights=None)
                        if hasattr(model, 'fc'):
                            model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
                        elif hasattr(model, 'classifier'):
                            if isinstance(model.classifier, torch.nn.Sequential):
                                model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, num_classes)
                            else:
                                model.classifier = torch.nn.Linear(model.classifier.in_features, num_classes)
                else:
                    model = getattr(Models, arch)(weights=None)

                model = model.to(device)

                if distributed:
                    ddp_kwargs = {} if device.type != 'cuda' else {"device_ids": [local_rank]}
                    model = DDP(model, **ddp_kwargs)

                criterion = torch.nn.CrossEntropyLoss()
                optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
                scheduler = (torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)
                             if dataset_type in ("cifar10", "cifar100")
                             else torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1))

                print(f"[TRAIN] {arch} | seed {seed}")
                model.train()

                for epoch in range(epochs):
                    if distributed and train_sampler is not None:
                        train_sampler.set_epoch(epoch)

                    running_loss = 0.0
                    epoch_loss_sum = 0.0
                    num_batches = 0

                    for i, (inputs, labels) in enumerate(train_loader):
                        inputs = inputs.to(device, non_blocking=True)
                        labels = labels.to(device, non_blocking=True)

                        optimizer.zero_grad(set_to_none=True)
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()

                        # logging
                        loss_val = loss.item()
                        running_loss += loss_val
                        epoch_loss_sum += loss_val
                        num_batches += 1

                        rank_prefix = f"[R{local_rank}] " if distributed else ""
                        if (i + 1) % 100 == 0:
                            msg = f"{rank_prefix}[{arch} seed{seed}] epoch {epoch+1} iter {i+1} loss {running_loss/100:.3f}"
                            if not distributed or local_rank == 0:
                                print(msg)
                            running_loss = 0.0

                    scheduler.step()

                    if (epoch + 1) % 10 == 0 or (epoch + 1) == epochs:
                    # save every epoch (rank 0 only)
                        if not distributed or local_rank == 0:
                            ckpt_path = seed_ckpt_dir / f"{arch}_epoch{epoch + 1}.pth"
                            avg_epoch_loss = epoch_loss_sum / max(1, num_batches)
                            _save_ckpt(
                                model=model,
                                optimizer=optimizer,
                                scheduler=scheduler,
                                ckpt_path=ckpt_path,
                                arch=arch,
                                seed=seed,
                                epoch=epoch + 1,
                                dataset_type=dataset_type,
                                train_loss_epoch=float(avg_epoch_loss),
                            )
                            print(f"Saved checkpoint: {ckpt_path} (train_loss_epoch={avg_epoch_loss:.6f})")

        results[arch] = {}

    return results

def _register_hooks_for_arch(model, arch):
    key = arch.lower()
    if key == 'clip_vit-b/16':
        key = 'clip_vit_b16'

    activations = {}
    layer_names = []

    def get_hook(name):
        def hook(_, __, out):
            activations[name] = out.detach()
        return hook

    def add(m, name):
        layer_names.append(name)
        m.register_forward_hook(get_hook(name))

    if key == 'alexnet':
        # indices match torchvision.models.alexnet.features:
        # 0:conv1 1:relu 2:pool1 3:conv2 4:relu 5:pool2 6:conv3 7:relu
        # 8:conv4 9:relu 10:conv5 11:relu 12:pool5
        add(model.features[2],  'features.pool1')   # stem/early boundary
        add(model.features[5],  'features.pool2')   # early boundary
        add(model.features[6],  'features.conv3')   # mid
        add(model.features[8],  'features.conv4')   # mid-late
        add(model.features[10], 'features.conv5')   # NEW: late conv (replaces fc3)
        add(model.features[12], 'features.pool5')   # final spatial compress
        add(model.classifier[1],'classifier.fc1')   # penultimate stack (1)
        add(model.classifier[4],'classifier.fc2')   # penultimate stack (2)

    elif key == 'resnet50':
        # Use fixed stage exits + two entries, plus stem + readout (8 taps total)
        last = {
            'l1': len(model.layer1) - 1,  # typically 2
            'l2': len(model.layer2) - 1,  # typically 3
            'l3': len(model.layer3) - 1,  # typically 5
            'l4': len(model.layer4) - 1,  # typically 2
        }
        add(model.maxpool,                  'maxpool')                      # stem summary
        add(model.layer1[last['l1']],       f'layer1.{last["l1"]}_postrelu')# stage1 exit (e.g., 1.2)
        add(model.layer2[0],                'layer2.0_postrelu')            # stage2 entry
        add(model.layer2[last['l2']],       f'layer2.{last["l2"]}_postrelu')# stage2 exit
        add(model.layer3[0],                'layer3.0_postrelu')            # stage3 entry
        add(model.layer3[last['l3']],       f'layer3.{last["l3"]}_postrelu')# stage3 exit
        add(model.layer4[last['l4']],       f'layer4.{last["l4"]}_postrelu')# stage4 exit
        add(model.avgpool,                  'avgpool')                      # readout

    elif key == 'vit_b_16':
        # TorchVision ViT-B/16: conv_proj (patch embed) + encoder.layers[*] + encoder.ln
        add(model.conv_proj,                'patch_embed.proj')             # stem
        add(model.encoder.layers[1],        'encoder_block_2_post')
        add(model.encoder.layers[3],        'encoder_block_4_post')
        add(model.encoder.layers[5],        'encoder_block_6_post')
        add(model.encoder.layers[7],        'encoder_block_8_post')
        add(model.encoder.layers[9],        'encoder_block_10_post')
        add(model.encoder.layers[11],       'encoder_block_12_post')
        add(model.encoder.ln,               'ln_post')                      # readout norm (CLS)
        # NOTE: For ViTs, hook outputs are token sequences. Downstream, take CLS: out[:, 0, :].

    elif key in ('clip_rn50', 'clip-rn50'):
        # Mirror ResNet: stem summary + stage entries/exits + attnpool (8 taps)
        v = model.visual
        add(v.avgpool,          'stem.avgpool')     # stem summary (CLIP stem is deeper)
        add(v.layer1[-1],       'layer1.last')      # stage1 exit
        add(v.layer2[0],        'layer2.0_postrelu')# stage2 entry
        add(v.layer2[-1],       'layer2.last')      # stage2 exit
        add(v.layer3[0],        'layer3.0_postrelu')# stage3 entry
        add(v.layer3[-1],       'layer3.last')      # stage3 exit
        add(v.layer4[-1],       'layer4.last')      # stage4 exit
        add(v.attnpool,         'attnpool')         # readout

    elif key in ('clip_vit_b16', 'clip_vit-b/16', 'clip_vit-b16'):
        # OpenAI CLIP ViT-B/16 visual: conv1 (patch embed), ln_pre, transformer.resblocks[*], ln_post
        v = model.visual
        add(v.conv1,                           'patch_embed') # stem
        # Use evenly spaced blocks; FIX token policy to CLS in downstream
        add(v.transformer.resblocks[1],        'block1')
        add(v.transformer.resblocks[3],        'block3')
        add(v.transformer.resblocks[5],        'block5')
        add(v.transformer.resblocks[7],        'block7')
        add(v.transformer.resblocks[9],        'block9')
        add(v.transformer.resblocks[11],       'block11')
        add(v.ln_post,                         'ln_post')     # readout norm (CLS)
        # NOTE: For CLIP ViT, hook outputs are token sequences. Downstream, take CLS: out[:, 0, :].

    else:
        print(f"[WARN] No predefined hooks for {arch}")

    return layer_names, activations


def _register_hooks_for_arch_2(model, arch):
    key = arch.lower().replace('/', '_')

    activations = {}
    layer_names = []

    def add(m, name):
        layer_names.append(name)
        def hook(_, __, out):
            activations[name] = out.detach()
        m.register_forward_hook(hook)

    def _get_clip_visual(m):
        # bare CLIP: m.visual; wrapped: m.clip.visual
        if hasattr(m, "visual"):
            return m.visual
        if hasattr(m, "clip") and hasattr(m.clip, "visual"):
            return m.clip.visual
        return None

    # -------- AlexNet (torchvision) --------
    if key == 'alexnet':
        # features: [0 conv1,1 relu,2 pool,3 conv2,4 relu,5 pool,6 conv3,7 relu,8 conv4,9 relu,10 conv5,11 relu,12 pool]
        add(model.features[2],  'features.pool1')
        add(model.features[5],  'features.pool2')
        add(model.features[6],  'features.conv3_out')
        add(model.features[8],  'features.conv4_out')
        add(model.features[10], 'features.conv5_out')
        add(model.features[12], 'features.pool5')
        # classifier: [0 drop,1 fc1,2 relu,3 drop,4 fc2,5 relu,6 fc3]
        add(model.classifier[1], 'classifier.fc1')
        add(model.classifier[4], 'classifier.fc2')
        add(model.classifier[6], 'classifier.fc3_logits')   # NEW: final logits

    # -------- ResNet-50 (torchvision) --------
    elif key == 'resnet50':
        add(model.maxpool, 'stem.maxpool')  # stem summary
        # hook output of every Bottleneck block in each stage
        for i, b in enumerate(model.layer1): add(b, f'layer1.{i}_out')
        for i, b in enumerate(model.layer2): add(b, f'layer2.{i}_out')
        for i, b in enumerate(model.layer3): add(b, f'layer3.{i}_out')
        for i, b in enumerate(model.layer4): add(b, f'layer4.{i}_out')
        add(model.avgpool, 'avgpool')
        add(model.fc, 'fc_logits')  # final logits

    # -------- ViT-B/16 (torchvision) --------
    elif key == 'vit_b_16':
        add(model.conv_proj, 'patch_embed.proj')
        for i, blk in enumerate(model.encoder.layers):
            add(blk, f'encoder.block_{i+1}_out')   # 1..12
        add(model.encoder.ln, 'encoder.ln_post')
        add(model.heads.head, 'heads.head_logits')          # NEW: final logits

    # -------- CLIP RN50 --------
    elif key in ('clip_rn50', 'clip-rn50'):
        v = _get_clip_visual(model)
        # CLIP RN50 stem has its own stack; avgpool is a good stem summary
        add(v.avgpool, 'visual.stem.avgpool')
        for i, b in enumerate(v.layer1): add(b, f'visual.layer1.{i}_out')
        for i, b in enumerate(v.layer2): add(b, f'visual.layer2.{i}_out')
        for i, b in enumerate(v.layer3): add(b, f'visual.layer3.{i}_out')
        for i, b in enumerate(v.layer4): add(b, f'visual.layer4.{i}_out')
        add(v.attnpool, 'visual.attnpool')

    # -------- CLIP ViT-B/16 --------
    elif key in ('clip_vit_b16', 'clip_vit-b_16', 'clip_vit-b16'):
        v = _get_clip_visual(model)
        add(v.conv1, 'visual.patch_embed')
        for i, blk in enumerate(v.transformer.resblocks):
            add(blk, f'visual.block_{i+1}_out')     # 1..12
        add(v.ln_post, 'visual.ln_post')

    else:
        print(f"[WARN] No predefined non-leaf hooks for {arch}")

    return layer_names, activations



def _compute_rdm_vec(X: np.ndarray, metric='correlation') -> tuple[np.ndarray, np.ndarray]:
    """
    X: [N, d] where N = items (images or classes). Returns (D, vec_upper).
    """
    D = squareform(pdist(X, metric=metric))
    v = D[np.triu_indices(D.shape[0], k=1)]
    return D, v


def _iterative_reweight(
    E: np.ndarray,
    norm: str = "zscore", # "minmax" | "zscore" | "rank"
    corr_mode: str = "pearson", # "pearson" | "spearman"
    weight_mode: str = "huber", # "inverse" | "softmax" | "huber" | "tukey"
    tau: float = 0.05, # temperature for softmax
    eps_abs: float = 1e-6,
    eps_rel: float = 1e-6,
    max_iter: int = 100
):
    E = E.astype(np.float32)
    # print('before norm:', E.mean(axis=1, keepdims=True), E.std(axis=1, keepdims=True))

    # --- normalization ---
    if norm == "minmax":
        E_min = E.min(axis=1, keepdims=True)
        E_max = E.max(axis=1, keepdims=True)
        E_norm = (E - E_min) / (E_max - E_min + 1e-8)
    elif norm == "rank":
        from scipy.stats import rankdata
        E_norm = np.apply_along_axis(rankdata, 1, E).astype(np.float32)
    else:  # zscore
        mu = E.mean(axis=1, keepdims=True)
        sd = E.std(axis=1, keepdims=True) + 1e-8
        E_norm = (E - mu) / sd

    # Spearman means Pearson on ranked data
    if corr_mode == "spearman" and norm != "rank":
        from scipy.stats import rankdata
        E_norm = np.apply_along_axis(rankdata, 1, E_norm).astype(np.float32)

    # Precompute constants
    E_center = E_norm - E_norm.mean(axis=1, keepdims=True)
    row_norms = np.linalg.norm(E_center, axis=1)

    # Init with robust mean
    E_mean = np.median(E_norm, axis=0)
    w = np.ones(E_norm.shape[0], dtype=np.float32)
    w_prev = w.copy()

    for it in range(max_iter):
        E_old = E_mean.copy()
        mean_center = E_mean - E_mean.mean()

        denom = row_norms * (np.linalg.norm(mean_center) + 1e-9)
        numer = E_center @ mean_center
        corr = numer / np.where(denom == 0, 1e-9, denom)
        corr = np.clip(corr, -0.999999, 0.999999)

        # --- weighting ---
        if weight_mode == "inverse":
            dist = 1.0 - corr
            w = 1.0 / (1.0 + dist)
        elif weight_mode == "softmax":
            w = np.exp(corr / max(tau, 1e-6)); w /= (w.sum() + 1e-9)
        elif weight_mode == "huber":
            r = 1.0 - corr
            s = np.median(np.abs(r - np.median(r))) + 1e-9  # robust scale
            t = r / (1.345 * s)
            w = 1.0 / np.maximum(1.0, np.abs(t))  # approximate Huber weights
        else:  # "tukey" (bisquare)
            r = 1.0 - corr
            s = np.median(np.abs(r - np.median(r))) + 1e-9
            t = r / (4.685 * s)
            w = (1 - t**2)**2
            w[t >= 1] = 0.0

        w = np.clip(w, 1e-8, None)
        # Weighted average
        E_mean = np.average(E_norm, axis=0, weights=w)

        # Stopping
        delta = np.linalg.norm(E_mean - E_old)
        rel = delta / (np.linalg.norm(E_old) + 1e-9)
        dw  = np.linalg.norm(w - w_prev) / (np.linalg.norm(w_prev) + 1e-9)
        w_prev = w.copy()
        if (delta < eps_abs or rel < eps_rel) and dw < 1e-6:
            break

    # print('after norm:', E.mean(axis=1, keepdims=True), E.std(axis=1, keepdims=True))
    return w, E_mean, it + 1, E_norm


def plot_weights(
    w,
    layer_name: str = "",
    arch_name: str = "",
    outdir: str | Path = r"C:\Users\BrainInspired\Documents\GitHub\Seeds_Analysis\figures",
    dpi: int = 200,
) -> Path:
    """
    Bar chart of per-seed weights. Saves to {outdir}/{arch}_{layer}_weights.png
    and returns the saved Path.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    def _slug(s: str) -> str:
        s = s.lower()
        s = s.replace("clip_vit-b/16", "clip_vit_b16")
        return s

    arch_slug  = _slug(arch_name)
    layer_slug = _slug(layer_name)
    fname = f"{arch_slug}_{layer_slug}_weights.png"
    fpath = outdir / fname

    n_seeds = len(w)
    x = np.arange(n_seeds)
    w = np.asarray(w, dtype=float)

    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    ax.bar(x, w, edgecolor="black")
    ax.set_xlabel("Seed index")
    ax.set_ylabel("Weight")
    ax.set_title(f"Reweighted seed contributions\n{arch_name} | {layer_name}")
    ax.set_xticks(x)

    ymax = float(np.nanmax(w)) if w.size else 1.0
    ax.set_ylim(0, max(ymax * 1.2, 1e-6))

    fig.savefig(fpath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {fpath}")
    return fpath

def extract_weight_matrix(arch_result):
    """
    arch_result = test_results[arch]  (a dict with keys: 'per_layer', 'layer_order')
    Returns:
      df  : pandas DataFrame [layers x seeds]
      meta: dict with 'layer_names' and 'n_seeds'
    """
    per_layer = arch_result["per_layer"]
    layer_names = arch_result.get("layer_order") or sorted(per_layer.keys())
    # collect weights lists
    rows = []
    for ln in layer_names:
        if ln not in per_layer:
            continue
        w = np.array(per_layer[ln]["weights"])
        rows.append(w)
    if not rows:
        raise ValueError("No layers with weights found.")
    W = np.vstack(rows)  # [L x S]
    # nice seed labels 0..S-1 based on length
    seed_labels = [f"seed_{i}" for i in range(W.shape[1])]
    df = pd.DataFrame(W, index=layer_names, columns=seed_labels)
    return df, {"layer_names": layer_names, "n_seeds": W.shape[1]}

def plot_weight_heatmap(df, arch_name, outdir="weights_heatmaps", vmin=None, vmax=None, annotate=False):
    """
    df: DataFrame [layers x seeds]
    Saves: {outdir}/{arch_name}_weights_heatmap.png
    """
    Path(outdir).mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, max(4, 0.35*len(df.index))))
    im = ax.imshow(df.values, aspect="auto", vmin=vmin, vmax=vmax)  # no explicit colormap to follow your style rules
    ax.set_title(f"{arch_name}  Seed Weights by Layer")
    ax.set_xlabel("Seeds")
    ax.set_ylabel("Layers")
    ax.set_xticks(np.arange(len(df.columns)))
    ax.set_xticklabels(df.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(df.index)))
    ax.set_yticklabels(df.index)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Weight", rotation=90)

    if arch_name == 'clip_ViT-B/16':
        outpath = Path(outdir) / f"clip_vit_b16_weights_heatmap.png"
    else:
        outpath = Path(outdir) / f"{arch_name}_weights_heatmap.png"
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close(fig)
    print(f"[saved] {outpath}")

def plot_all_arch_heatmaps(test_results, outdir="weights_heatmaps", row_normalize=False):
    """
    Makes one heatmap per architecture in test_results.
    If row_normalize=True, each layer is divided by its max (helps compare patterns across layers).
    """
    for arch, arch_result in test_results.items():
        if not arch_result.get("per_layer"):
            print(f"[skip] {arch}: no per_layer data")
            continue
        df, meta = extract_weight_matrix(arch_result)
        if row_normalize:
            df_plot = df.div(df.max(axis=1).replace(0, np.nan), axis=0)
        else:
            df_plot = df

        # Align color scale across layers (optional): use [0, 1] if weights are in that range
        vmin, vmax = (0.0, 1.0) if (df_plot.min().min() >= 0 and df_plot.max().max() <= 1.0) else (None, None)
        plot_weight_heatmap(df_plot, arch, outdir=outdir, vmin=vmin, vmax=vmax)


# def _feat_to_matrix(
#     feat: Any,
#     layer_name: str,
#     batch_size: int | None = None,
# ) -> torch.Tensor | None:
#     """
#     Convert arbitrary layer activation to [B, d]:
#       - 4D (B,C,H,W)  -> GAP over HxW -> (B,C)
#       - 3D (B,T,C)    -> CLS token [:,0,:] if T>=1, else token-mean -> (B,C)
#       - 2D (B,d)      -> pass-through
#       - >4D           -> flatten after batch -> (B, -1)
#       - non-Tensor / 0D / 1D -> return None (caller should skip)
#     """
#     if not isinstance(feat, torch.Tensor):
#         return None
#     x = feat

#     # ensure batch is dim-0 if we can infer it
#     B_expected = batch_size if batch_size is not None else (x.shape[0] if x.ndim >= 1 else None)
#     if (x.ndim >= 1) and (B_expected is not None) and (x.shape[0] != B_expected):
#         for d in range(x.ndim):
#             if x.shape[d] == B_expected:
#                 perm = [d] + [i for i in range(x.ndim) if i != d]
#                 x = x.permute(*perm)
#                 break

#     if x.ndim == 4:
#         # [B, C, H, W] -> [B, C]
#         x = F.adaptive_avg_pool2d(x, (1, 1)).reshape(x.shape[0], -1)
#         return x.contiguous()

#     if x.ndim == 3:
#         # [B, T, C] (ViT/transformers)
#         T = x.shape[1]
#         x = x[:, 0, :] if T >= 1 else x.mean(dim=1)
#         return x.contiguous()

#     if x.ndim == 2:
#         return x.contiguous()

#     if x.ndim > 4:
#         x = x.reshape(x.shape[0], -1)
#         return x.contiguous()

#     # 0D/1D cases don't have batch; skip
#     return None
def _feat_to_matrix(
    feat,
    layer_name: str,
    batch_size: int | None = None,
) -> torch.Tensor | None:
    
    if not isinstance(feat, torch.Tensor):
        return None
    x = feat

    # Ensure batch is dim-0 if we can infer it
    B_expected = batch_size if batch_size is not None else (x.shape[0] if x.ndim >= 1 else None)
    if (x.ndim >= 1) and (B_expected is not None) and (x.shape[0] != B_expected):
        for d in range(x.ndim):
            if x.shape[d] == B_expected:
                perm = [d] + [i for i in range(x.ndim) if i != d]
                x = x.permute(*perm)
                break

    if x.ndim == 4:
        x = F.adaptive_avg_pool2d(x, (1, 1)).reshape(x.shape[0], -1)
        return x.contiguous()

    if x.ndim == 3:
        T = x.shape[1]
        x = x[:, 0, :] if T >= 1 else x.mean(dim=1)
        return x.contiguous()

    if x.ndim == 2:
        return x.contiguous()

    if x.ndim > 4:
        x = x.reshape(x.shape[0], -1)
        return x.contiguous()

    return None



def test_pipeline(
    arch_list: list[str],
    data_root: str | Path,
    dataset_type: str, # "cifar100"
    ckpt_root: Path,
    device: str,
    batch_size: int = 256,
    num_workers: int = 8,
    subsample_test_batches: int | None = None, # None = full test
    save_rdms: bool = True,
    rdm_level: str = "class", # "image" or "class"
    metric: str = "correlation", # distance metric for RDM
    n_seeds: int | None = None, # max seeds to use per arch (None = all)
) -> dict:
    """
    TEST-TIME: extract features per seed on CIFAR test, build RDMs at image- or class-level,
    run iterative reweighting + black-swan, and save per-layer NPZs.
    """
    assert rdm_level in ("image", "class"), "rdm_level must be 'image' or 'class'"
    device = torch.device(device)

    # CIFAR test split (default torchvision transform; CLIP will override)
    if dataset_type not in ("cifar10", "cifar100"):
        raise ValueError("This test refactor expects CIFAR for now.")
    num_classes = 10 if dataset_type == "cifar10" else 100
    standard_transform = T.Compose([
        T.Resize(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    cls = Datasets.CIFAR10 if dataset_type == "cifar10" else Datasets.CIFAR100

    results = {}
    for arch in arch_list:
        arch_dir = ckpt_root / arch
        if arch == 'clip_ViT-B/16':
            arch_dir = ckpt_root / 'clip_vit_b16'
        seed_dirs = sorted(arch_dir.glob("seed_*"))
        if not seed_dirs:
            print(f"[TEST] No seed directories for {arch}")
            continue
        
        if n_seeds is not None:
            seed_dirs = [
                d for d in seed_dirs
                if (m := re.match(r"seed_(\d+)$", d.name)) and int(m.group(1)) < n_seeds
            ]

        if not seed_dirs:
            msg = f"[TEST] No seed directories for {arch}"
            if n_seeds is not None:
                msg += f" after filtering to range(n_seeds) where n_seeds={n_seeds}"
            print(msg)
            continue
        
        layer_names_global = None
        per_layer_vecs: dict[str, list[np.ndarray]] = {}

        # === CLIP model name ===
        clip_name = None
        if arch.startswith('clip_'):
            clip_name = arch[5:]

        # === Build a test loader (CLIP overrides transform) ===
        if clip_name is None:
            test_ds = cls(root=data_root, train=False, download=True, transform=standard_transform)
            base_test_loader = DataLoader(
                test_ds, batch_size=batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=True,
                persistent_workers=(num_workers > 0),
            )
        else:
            # Use CLIP preprocess
            _, clip_preprocess = clip.load(clip_name, device=device, jit=False)
            test_ds = cls(root=data_root, train=False, download=True, transform=clip_preprocess)
            base_test_loader = DataLoader(
                test_ds, batch_size=batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=True,
                persistent_workers=(num_workers > 0),
            )

        for seed_dir in seed_dirs:
            ckpt_files = list(seed_dir.glob(f"{arch}_epoch*.pth"))
            if arch == 'clip_ViT-B/16':
                ckpt_files = list(seed_dir.glob(f"clip_vit_b16_epoch*.pth"))
            if not ckpt_files and clip_name is None:
                print(f"[TEST] No checkpoints in {seed_dir}")
                continue
            ckpt_path = max(ckpt_files, key=lambda p: int(p.stem.split("epoch")[-1])) if ckpt_files else None

            # === Build/Load model ===
            if clip_name is None:
                # torchvision models with CIFAR heads
                if arch == 'alexnet':
                    model = Models.alexnet(weights=None)
                    model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, num_classes)
                elif arch == 'resnet50':
                    model = Models.resnet50(weights=None)
                    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
                elif arch == 'vit_b_16':
                    model = Models.vit_b_16(weights=None)
                    model.heads.head = torch.nn.Linear(model.heads.head.in_features, num_classes)
                else:
                    model = getattr(Models, arch)(weights=None)
                    if hasattr(model, 'fc'):
                        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
                    elif hasattr(model, 'classifier'):
                        if isinstance(model.classifier, torch.nn.Sequential):
                            model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, num_classes)
                        else:
                            model.classifier = torch.nn.Linear(model.classifier.in_features, num_classes)
                model = model.to(device)
                state = torch.load(ckpt_path, map_location=device)
                model.load_state_dict(state, strict=True)
                model.eval()
                test_loader = base_test_loader

            else:
                # === CLIP ===
                model, _ = clip.load(clip_name, device=device, jit=False)
                
                model.float()
                
                for p in model.parameters():
                    if p.dtype.is_floating_point:
                        p.data = p.data.float()
                        
                with torch.no_grad():
                    model.logit_scale.data = model.logit_scale.data.float()
                    
                if ckpt_path is not None:
                    state = torch.load(ckpt_path, map_location=device)
                    
                    if any(k.startswith("module.") for k in state.keys()):
                        state = {k.replace("module.", "", 1): v for k, v in state.items()}
                    
                    missing, unexpected = model.load_state_dict(state, strict=False)
                    
                    if missing or unexpected:
                        print(f"[TEST][{arch}] load_state_dict: missing={len(missing)} unexpected={len(unexpected)}")
                else:
                    print(f"[TEST][{arch}] No checkpoint in {seed_dir}; using pretrained CLIP weights.")
                        
                model = model.to(device).eval()
                test_loader = base_test_loader

            # register hooks
            layer_names, activations = _register_hooks_for_arch(model, arch)
            # layer_names, activations = _register_hooks_for_arch_2(model, arch)

            if layer_names_global is None:
                layer_names_global = list(layer_names)
                per_layer_vecs = {ln: [] for ln in layer_names_global}

            # --- Collect features ---
            if rdm_level == "image":
                layerwise_feats = {ln: [] for ln in layer_names_global}
                with torch.no_grad():
                    for bi, (x, y) in enumerate(test_loader):
                        if subsample_test_batches is not None and bi >= subsample_test_batches:
                            break
                        x = x.to(device)

                        # Forward so hooks fire
                        if clip_name is None:
                            _ = model(x)
                        else:
                            _ = model.encode_image(x)  # CLIP visual forward


                        for ln in layer_names_global:
                            feat = activations[ln]
                            feat = _feat_to_matrix(feat, ln, batch_size=x.size(0))  # [B, d]
                            layerwise_feats[ln].append(feat.cpu().numpy())

                for ln in layer_names_global:
                    X = np.vstack(layerwise_feats[ln])  # [N_images, d]
                    D, v = _compute_rdm_vec(X, metric=metric)
                    per_layer_vecs[ln].append(v)
                    if save_rdms:
                        np.save(seed_dir / f"{arch}_{ln}_rdm_image_seed{seed_dir.name.split('_')[-1]}.npy", D)

            else:  # rdm_level == "class"
                class_sums = {ln: [None]*num_classes for ln in layer_names_global}
                class_counts = {ln: np.zeros(num_classes, dtype=np.int64) for ln in layer_names_global}

                with torch.no_grad():
                    for bi, (x, y) in enumerate(test_loader):
                        if subsample_test_batches is not None and bi >= subsample_test_batches:
                            break
                        x = x.to(device)

                        if clip_name is None:
                            _ = model(x)
                        else:
                            _ = model.encode_image(x)

                        y_np = y.numpy()
                        for ln in layer_names_global:
                            feat = activations[ln]
                            feat = _feat_to_matrix(feat, ln, batch_size=x.size(0))  # [B, d]
                            feat = feat.cpu().numpy()

                            for c in np.unique(y_np):
                                idx = (y_np == c)
                                block_sum = feat[idx].sum(axis=0)
                                if class_sums[ln][c] is None:
                                    class_sums[ln][c] = block_sum
                                else:
                                    class_sums[ln][c] += block_sum
                                class_counts[ln][c] += idx.sum()

                for ln in layer_names_global:
                    means = []
                    for c in range(num_classes):
                        assert class_counts[ln][c] > 0, f"No samples for class {c}"
                        means.append(class_sums[ln][c] / class_counts[ln][c])
                    Xc = np.stack(means, axis=0)  # [C, d]
                    D, v = _compute_rdm_vec(Xc, metric=metric)
                    per_layer_vecs[ln].append(v)
                    if save_rdms:
                        if arch == 'clip_ViT-B/16':
                            np.save(seed_dir / f"clip_vit_b16_{ln}_rdm_class_seed{seed_dir.name.split('_')[-1]}.npy", D)
                        else:
                            np.save(seed_dir / f"{arch}_{ln}_rdm_class_seed{seed_dir.name.split('_')[-1]}.npy", D)

        # --- After all seeds: stack and reweight per layer ---
        arch_results = {}
        for ln in (layer_names_global or []):
            if len(per_layer_vecs[ln]) == 0:
                continue
            E = np.stack(per_layer_vecs[ln], axis=0)  # [n_seeds, n_distances]
            print(f"[TEST][{arch}][{ln}] Running iterative reweighting...")
            w, E_mean, iters, E_norm = _iterative_reweight(E)
            plot_weights(w, layer_name=ln, arch_name=arch)

            seed_ids = [int(s.name.split('_')[-1]) for s in seed_dirs]
            bs = identify_black_swans(w, E_norm, E_mean, seed_ids)

            suffix = "class" if rdm_level == "class" else "image"
            if arch == 'clip_ViT-B/16':
                np.savez(
                    arch_dir / f"clip_vit_b16_{ln}_seed_embeddings_weighted_{suffix}.npz",
                    embeddings=E_norm, weights=w
                )
            else:
                np.savez(
                    arch_dir / f"{arch}_{ln}_seed_embeddings_weighted_{suffix}.npz",
                    embeddings=E_norm, weights=w
                )

            arch_results[ln] = {
                "embeddings": E_norm,
                "weights": w,
                "converged_mean": E_mean,
                "convergence_iterations": iters,
                "black_swan_analysis": bs,
                "rdm_level": rdm_level,
                "metric": metric,
            }

            print("Threshold:", bs["threshold"])
            print("Mean weight:", bs["mean_weight"])
            print("Std weight:", bs["std_weight"])
            print("Black Swan indices:", bs["black_swan_indices"])
            print("Details per seed:")
            for sid, details in bs["black_swan_details"].items():
                print(f"  Seed {sid}: weight={details['weight']:.4f}, corr={details['validation_correlation']:.4f}")

        results[arch] = {"per_layer": arch_results, "layer_order": layer_names_global or []}
        plot_all_arch_heatmaps(results, outdir=r"C:\Users\BrainInspired\Documents\GitHub\Seeds_Analysis\figures", row_normalize=False)
        print(f"[TEST] Completed {arch} ({rdm_level}-level)")

    return results


# def _timm_pretrained_models():
#     if not HAS_TIMM:
#         return []
#     names = timm.list_models(pretrained=True)

#     # tokens commonly used in timm model *names* for ImageNet(-1k/12k/21k/22k)
#     imagenet_name_tokens = (
#         "in1k", "in12k", "in21k", "in22k", "imagenet",
#         "imgnet", "img1k", "i1k"  # rare/alias-y, included just in case
#     )

#     def _is_imagenet_by_name(model_id: str) -> bool:
#         s = model_id.lower()
#         if any(tok in s for tok in imagenet_name_tokens):
#             return True
#         # also catch separators like "-in1k", "_in1k" (covered by substring test),
#         # but keep a regex hook here if you want stricter matching later:
#         # return bool(re.search(r"(?:^|[_\-])in(?:1|12|21|22)k(?:$|[_\-])", s))
#         return False

def _timm_pretrained_models():
    if not HAS_TIMM:
        return []

    names = timm.list_models(pretrained=True,
                             exclude_filters=['*in1k*','*in12k*','*in21k*','*in22k*',
                                              '*imagenet*','*imgnet*','*i1k*'])

    keep = []
    for mid in names:
        s = mid.lower()
        if any(tok in s for tok in ("in1k","in12k","in21k","in22k","imagenet","imgnet","img1k","i1k")):
            continue
        keep.append(mid)
    return keep


_SMALL_MID_ALLOW = re.compile(
    r"(?:^|[_\-])(tiny|small|base|nano|micro|mini|lite|pico|femto|atto)(?:$|[_\-])"
    r"|efficientnet_b[0-4]\b"
    r"|regnet[xy].*(?:0\.[0-9]|1\.[0-9]|2\.[0-9]|3\.[0-9]|4\.[0-9])"   # ~≤4GF lines
    r"|convnext(?:_v2)?_(?:atto|femto|pico|nano|tiny|small|base)\b"
    r"|vit_(?:tiny|small|base)\b"
    r"|deit_(?:tiny|small|base)\b"
    r"|swin_(?:tiny|small|base)\b"
    r"|mobilenet"
    r"|mnasnet"
    r"|densenet(?:121|169)\b"
    r"|resnet(?:18|34|50)\b"
    r"|resnext50_32x4d\b",
    re.IGNORECASE
)

_BIG_DENY = re.compile(
    r"(aimv2_(?:1b|3b|large|xlarge|xxlarge|huge))"
    r"|(?:^|[_\-])(?:3b|1b)(?:$|[_\-])"
    r"|(?:^|[_\-])(giant|gigantic|huge|xlarge|xxlarge)(?:$|[_\-])"
    r"|convnext_.*xxlarge"
    r"|vit_[gh]|\bvit_(?:giant|huge)\b"
    r"|beit3_.*large|beit_.*(?:huge|giant)"
    r"|swinv2_.*giant|swin_.*huge"
    r"|eva(?:02)?_.*(?:large|giant|huge)",
    re.IGNORECASE
)

def _timm_pretrained_models_small(verbose=True):
    all_ok = _timm_pretrained_models()
    small_mid = [m for m in all_ok if _SMALL_MID_ALLOW.search(m) and not _BIG_DENY.search(m)]
    if verbose:
        print(f"[timm] after ImageNet filter: {len(all_ok)} models")
        print(f"[timm] small/mid retained:    {len(small_mid)} models")
        # peek a few families so you can sanity-check
        for m in sorted(small_mid):
            print("  ", m)
    return small_mid[80:]


def _timm_model_and_transform(model_id: str, device: torch.device):
    model = timm.create_model(model_id, pretrained=True, num_classes=0).to(device).eval()
    data_cfg = resolve_model_data_config(model)
    transform = create_transform(**data_cfg)   # resize/crop/normalize match pretraining
    size = tuple(data_cfg.get("input_size", (3, 224, 224))[-2:])
    return model, transform, size


def _register_all_layers_safely(model: nn.Module, leaf_only: bool = True):
    """
    Auto-hook 'meaningful' layers (skip micro-layers), summarize to [B,d] in-hook,
    and offload to CPU to reduce GPU memory.

    Returns:
      layer_names: List[str]        # stable, unique names in traversal order
      activations: Dict[str,Tensor] # filled during forward; clear() per batch
    """
    # ---------- config: what to SKIP / KEEP ----------
    _MICRO_TYPES = (
        nn.ReLU, nn.GELU, nn.SiLU, nn.LeakyReLU, nn.PReLU, nn.Tanh, nn.Sigmoid, nn.Hardswish,
        nn.Dropout, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout,
        nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm,
        nn.LayerNorm, nn.GroupNorm, nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d,
        nn.Identity, nn.Flatten, nn.Upsample,
    )
    try:
        from timm.layers import DropPath, LayerScale, BlurPool
        _MICRO_TYPES = _MICRO_TYPES + (DropPath, LayerScale, BlurPool)
    except Exception:
        pass

    _SKIP_NAME_PATTERNS = tuple(re.compile(p) for p in [
        r"\b(drop|droppath|stochastic[_-]?depth)\b",
        r"\b(add|getitem|reshape|permute|transpose)\b",
        r"\b(layerscale|layer_scale)\b",
    ])

    # things we DO want to keep even if they’re simple (readouts / stage boundaries)
    _KEEP_NAME_PATTERNS = tuple(re.compile(p) for p in [
        r"\b(stem|patch_embed|conv1)\b",
        r"\b(layer[1-4](?:\.\d+)?(?:_out)?)\b",          # resnet stages / blocks
        r"\b(blocks?\.\d+(?:_out)?)\b",                  # vit/transformer blocks
        r"\b(stages?\.\d+|stage\d+)\b",                  # convnext/etc. stage taps
        r"\b(attnpool|avgpool|global_pool|pre_logits)\b",
        r"\b(fc|head|classifier)\b",
        r"\b(ln_post|encoder\.ln|norm)$",                # final LN / norm-ish readouts
    ])

    def _looks_meaningful(name: str) -> bool:
        lname = name.lower()
        return any(p.search(lname) for p in _KEEP_NAME_PATTERNS)

    def _should_skip_micro(module: nn.Module, name: str) -> bool:
        # skip known micro layers unless they’re meaningful readouts by name
        if isinstance(module, _MICRO_TYPES) and not _looks_meaningful(name):
            return True
        lname = name.lower()
        if any(p.search(lname) for p in _SKIP_NAME_PATTERNS):
            return True
        return False

    # ---------- helpers ----------
    def _pick_tensor(out):
        # Accept a single Tensor, or search tuples/lists/dicts for the first Tensor.
        if isinstance(out, torch.Tensor):
            return out
        if isinstance(out, (tuple, list)):
            for t in out:
                if isinstance(t, torch.Tensor):
                    return t
            return None
        if isinstance(out, dict):
            for v in out.values():
                if isinstance(v, torch.Tensor):
                    return v
            return None
        return None

    def _gap4d(x: torch.Tensor) -> torch.Tensor:
        # [B,C,H,W] -> [B,C]
        return F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)

    def _tokens3d(x: torch.Tensor) -> torch.Tensor:
        # [B,T,C] -> CLS (or mean over tokens if CLS not meaningful)
        if x.dim() == 3:
            return x[:, 0, :] if x.size(1) >= 1 else x.mean(1)
        return x

    activations: dict[str, torch.Tensor] = {}
    layer_names: list[str] = []
    used_names: set[str] = set()
    handles = []  # we keep them to avoid GC; you can choose to remove later if desired

    def _unique(name: str) -> str:
        if name not in used_names:
            used_names.add(name)
            return name
        i = 1
        while f"{name}#{i}" in used_names:
            i += 1
        u = f"{name}#{i}"
        used_names.add(u)
        return u

    # ---------- register hooks ----------
    for name, module in model.named_modules():
        if name == "":
            continue  # skip root
        if leaf_only and any(True for _ in module.children()):
            continue  # only leaves, to avoid duplicate container captures
        if _should_skip_micro(module, name):
            continue  # skip micro layers and wiring ops

        safe_name = _unique(name)
        layer_names.append(safe_name)

        def _make_hook(nm: str):
            def _hook(_mod, _inp, out):
                try:
                    t = _pick_tensor(out)
                    if t is None:
                        return
                    # summarize to [B,d] right here (saves a lot of memory)
                    if t.dim() == 4:
                        t = _gap4d(t)
                    elif t.dim() == 3:
                        t = _tokens3d(t)
                    elif t.dim() == 2:
                        pass
                    else:
                        # rare shapes: just flatten per-sample
                        t = t.flatten(1)
                    # offload to CPU immediately to release GPU memory
                    activations[nm] = t.detach().to("cpu", copy=True).contiguous()
                except Exception:
                    # swallow edge cases; don’t kill the run for one layer
                    pass
            return _hook

        try:
            h = module.register_forward_hook(_make_hook(safe_name), with_kwargs=False)
            handles.append(h)
        except Exception:
            # some modules may refuse hooks; ignore them
            pass

    # NOTE:
    # - You should `activations.clear()` after each batch.
    # - If you want to remove hooks later, keep `handles` and call h.remove() per handle.
    return layer_names, activations

def run_nod_pipeline_pretrained(
    nod_root: Path,
    subsample_val_batches: int,
    batch_size: int,
    num_workers: int,
    device: str,
    rdm_level: str = "class",                     # "image" or "class"
) -> Dict[str, Dict[str, Any]]:
    """
    For each architecture and seed:
      - forward NOD once while capturing the selected (non-leaf) layers via hooks,
      - convert each layer's activations -> [items, d] (GAP for CNN maps, CLS for ViTs),
      - compute an RDM per layer (matrix + vector),
      - stack per-seed results: per_seed_rdms_mat: (n_seeds, L, N, N), per_seed_rdms_vec: (n_seeds, L, D).

    Returns:
      per_arch = {
        arch: {
          "seed_ids": List[int],
          "layer_names": List[str],                  # length L (order is consistent across seeds)
          "rdm_level": str,                          # "image" | "class"
          "metric": "correlation",
          "per_seed_rdms_mat": np.ndarray,           # (n_seeds, L, N, N)
          "per_seed_rdms_vec": np.ndarray,           # (n_seeds, L, D)
        },
        ...
      }
    """
    assert rdm_level in ("image", "class"), "rdm_level must be 'image' or 'class'"
    metric = "correlation"
    device = torch.device(device)

    nod_rdms_per_layer = {}

    # --- Default (ImageNet-like) preprocessing for torchvision models ---
    preprocess_default = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Base loader used by non-CLIP models
    nod_ds_default = FilteredImageFolder(nod_root, transform=preprocess_default)
    nod_loader_default = DataLoader(
        nod_ds_default, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    per_arch: Dict[str, Dict[str, Any]] = {}

    all_models = _timm_pretrained_models_small(verbose=True)
    print(f"[timm] found {len(all_models)} pretrained models")

    env = os.environ.get("SEEDS_TMPCACHE_ROOT")
    if env:
        base_tmp = Path(env).expanduser().resolve()
    else:
        base_tmp = (Path(tempfile.gettempdir()) / "seeds_tmpcache").resolve()

    base_tmp.mkdir(parents=True, exist_ok=True)

    for mid in all_models:
        arch_key = f"timm:{mid}"

        with per_model_tmpcache(mid, base_tmp):
            try:
                print(f"[PRETRAINED][{arch_key}] loading …")
                model, transform, _ = _timm_model_and_transform(mid, device)

                # dataset/loader with model-native preprocessing
                nod_ds = FilteredImageFolder(nod_root, transform=transform)
                nod_loader = DataLoader(
                    nod_ds, batch_size=batch_size, shuffle=False,
                    num_workers=num_workers, pin_memory=True, persistent_workers=False
                )

                # hook almost all layers safely
                candidate_layers, activations = _register_all_layers_safely(model)
                if not candidate_layers:
                    print(f"[SKIP][{arch_key}] no hookable layers.")
                    continue

                # per-class aggregation
                class_sums: Dict[str, np.ndarray] = {}
                class_counts: Dict[str, np.ndarray] = {}
                seen_layers: set[str] = set()
                n_classes = len(getattr(nod_loader.dataset, "classes", [])) or None

                with torch.no_grad():
                    for bi, (x, y) in enumerate(nod_loader):
                        if subsample_val_batches and bi >= subsample_val_batches:
                            break
                        B = x.size(0)
                        x = x.to(device, non_blocking=True)
                        _ = model(x)

                        y_np = y.numpy()
                        classes_in_batch = np.unique(y_np)

                        for ln, feat in list(activations.items()):
                            if feat is None:
                                print(f"[WARN][{arch_key}][{ln}] no activation captured.")
                                continue
                            if isinstance(feat, (tuple, list)):
                                feat = next((t for t in feat if isinstance(t, torch.Tensor)), None)
                                if feat is None:
                                    print(f"[WARN][{arch_key}][{ln}] no Tensor found in activation tuple/list.")
                                    continue

                            f = _feat_to_matrix(feat, ln, batch_size=B)  # -> [B, d]
                            if f.dim() != 2 or f.size(0) != B:
                                print(f"[WARN][{arch_key}][{ln}] unexpected feature shape {feat.shape}.")
                                continue

                            f = f.detach().cpu().numpy()
                            if ln not in class_sums:
                                assert n_classes is not None, "Cannot infer number of classes from dataset."
                                class_sums[ln] = np.zeros((n_classes, f.shape[1]), dtype=np.float64)
                                class_counts[ln] = np.zeros((n_classes,), dtype=np.int64)

                            for c in classes_in_batch:
                                idx = (y_np == c)
                                if idx.any():
                                    class_sums[ln][c] += f[idx].sum(axis=0)
                                    class_counts[ln][c] += int(idx.sum())

                            seen_layers.add(ln)

                        activations.clear()  # control peak memory

                # build per-layer RDMs
                kept_layers = [ln for ln in class_sums.keys()]
                print(f"[PRETRAINED]building RDM for {len(kept_layers)} layers …")

                layer_rdms_mat, layer_rdms_vec = [], []
                for ln in kept_layers:
                    cc = class_counts[ln]          # shape [C]
                    mask = cc > 0
                    if mask.sum() < 2:
                        continue                   # need ≥2 classes for an RDM
                    Xc = (class_sums[ln][mask] / cc[mask, None]).astype(np.float32)
                    D, v = _compute_rdm_vec(Xc, metric="correlation")
                    layer_rdms_mat.append(D.astype(np.float32))
                    layer_rdms_vec.append(v.astype(np.float32))

                if not layer_rdms_mat:
                    print(f"[SKIP][{arch_key}] no valid layers produced class means.")
                    continue

                seed_rdms_mat = np.stack(layer_rdms_mat, axis=0)[None, ...]  # (1, L, N, N)
                seed_rdms_vec = np.stack(layer_rdms_vec, axis=0)[None, ...]  # (1, L, D)

                # per_arch[arch_key] = {
                #     "seed_ids": [0],
                #     "layer_names": kept_layers,
                #     "rdm_level": rdm_level,
                #     "metric": metric,
                #     "per_seed_rdms_mat": seed_rdms_mat,
                #     "per_seed_rdms_vec": seed_rdms_vec,
                # }

                per_arch[arch_key] = {
                    "layer_names": kept_layers,
                    "rdm_level": rdm_level,
                }
                nod_rdms_per_layer[arch_key] = seed_rdms_vec

                L = seed_rdms_mat.shape[1]; N = seed_rdms_mat.shape[2]; Ddim = seed_rdms_vec.shape[-1]
                print(f"[PRETRAINED][{arch_key}] layers={L} N={N} D={Ddim}")

            except Exception as e:
                print(f"[SKIP][{arch_key}] load failed: {e}")
                continue
            finally:
                try:
                    del model, transform, nod_loader, nod_ds, activations
                except Exception:
                    pass
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

    return nod_rdms_per_layer, per_arch


def _build_model_and_transform_generic(name: str, device: torch.device):
    """
    name formats:
      - 'tv:resnet50'           -> torchvision with Weights.DEFAULT
      - 'timm:swin_tiny_patch4_window7_224' -> timm with pretrained=True
      - bare 'resnet50' (for BC) -> treated as 'tv:resnet50'
    Returns: (model.eval().to(device), preprocess_transform)
    """
    if ":" not in name:
        prefix, model_id = "tv", name
    else:
        prefix, model_id = name.split(":", 1)

    if prefix == "tv":
        weights_enum = get_model_weights(model_id)
        weights = getattr(weights_enum, "DEFAULT", None) if weights_enum else None
        model = get_model(model_id, weights=weights).to(device).eval()
        if weights is not None:
            preprocess = weights.transforms()
        else:
            preprocess = T.Compose([
                T.Resize(256), T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
            ])
        return model, preprocess

    if prefix == "timm":
        assert HAS_TIMM, "timm not installed. pip install timm"
        model = timm.create_model(model_id, pretrained=True).to(device).eval()
        cfg = resolve_model_data_config(model)
        preprocess = create_transform(**cfg)
        return model, preprocess

    raise ValueError(f"Unknown model prefix '{prefix}' in '{name}'")


@torch.no_grad()
def run_pretrained_nod_per_layer_allnodes(
    arch_list: List[str],
    nod_root: Path,
    batch_size: int,
    num_workers: int,
    device: str,
    rdm_level: str = "class",   # "class" expected
):
    assert rdm_level in ("class", "image")
    dev = torch.device(device)
    nod_rdms_per_layer, per_arch_meta = {}, {}

    # Dataset with model-native transform (updated per arch)
    class Filtered(ImageFolder):
        def __init__(self, root, transform=None):
            super().__init__(root, transform=transform)
        def find_classes(self, directory: str):
            classes = [d.name for d in os.scandir(directory) if d.is_dir() and not d.name.startswith(".")]
            classes.sort()
            class_to_idx = {c: i for i, c in enumerate(classes)}
            return classes, class_to_idx

    for arch in arch_list:
        print(f"[PRETRAINED] {arch}")
        model, preprocess = _build_model_and_transform_generic(arch, dev)

        # fresh dataset per arch to honor model's transform
        ds = Filtered(nod_root, transform=preprocess)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)

        # grab eval nodes; prefilter obvious non-activations
        train_nodes, eval_nodes = get_graph_node_names(model)
        BAD = ("layer_scale", "num_batches_tracked", ".weight", ".bias")
        cand = [n for n in eval_nodes if not any(b in n for b in BAD)]
        if not cand:
            print(f"[WARN][{arch}] no candidate nodes")
            continue

        fx = create_feature_extractor(model, return_nodes={n: n for n in cand}).to(dev).eval()

        usable_nodes: List[str] = []
        class_names = ds.classes
        L = None

        # accumulators
        class_sums: Dict[str, List[np.ndarray | None]] = {}
        class_counts: Dict[str, np.ndarray] = {}

        first_batch_done = False
        for bi, (x, y) in enumerate(loader):
            x = x.to(dev, non_blocking=True)
            y_np = y.numpy()
            B = x.shape[0]

            feats = fx(x)

            # first-batch probe: keep only tensor outputs with batch-first [B, ...]
            if not first_batch_done:
                usable_nodes = [n for n, f in feats.items()
                                if isinstance(f, torch.Tensor) and (f.ndim >= 2) and (f.shape[0] == B)]
                if not usable_nodes:
                    print(f"[WARN][{arch}] no usable tensor nodes; skipping arch.")
                    usable_nodes = []
                    break
                # shrink extractor to usable nodes
                fx = create_feature_extractor(model, return_nodes={n: n for n in usable_nodes}).to(dev).eval()
                feats = fx(x)  # re-run
                L = len(usable_nodes)
                for name in usable_nodes:
                    class_sums[name] = [None] * len(class_names)
                    class_counts[name] = np.zeros(len(class_names), dtype=np.int64)
                first_batch_done = True

            # accumulate per class
            for name in usable_nodes:
                M = _feat_to_matrix(feats[name], name, batch_size=B)
                if M is None:
                    continue
                if (M.ndim != 2) or (M.shape[0] != B):
                    continue
                M = M.detach().cpu().numpy()

                for c in np.unique(y_np):
                    idx = (y_np == c)
                    if idx.shape[0] != M.shape[0]:
                        continue
                    block_sum = M[idx].sum(axis=0)
                    if class_sums[name][c] is None:
                        class_sums[name][c] = block_sum
                    else:
                        class_sums[name][c] += block_sum
                    class_counts[name][c] += int(idx.sum())

        if not usable_nodes:
            continue

        # build class means per node, then RDM vec
        layer_vecs = []
        layer_names = []
        for name in usable_nodes:
            counts = class_counts[name]
            sums = class_sums[name]
            if (counts is None) or (sums is None) or np.any(counts == 0):
                continue
            Xc = np.stack([s / counts[i] for i, s in enumerate(sums)], axis=0).astype(np.float32)  # [C, d]
            _, v = _compute_rdm_vec(Xc, metric="correlation")
            layer_vecs.append(v.astype(np.float32))
            layer_names.append(name)

        if not layer_vecs:
            print(f"[WARN][{arch}] no layer vectors made.")
            continue

        E = np.stack(layer_vecs, axis=0)  # (L, D)
        nod_rdms_per_layer[arch] = E[None, :, :]  # (1, L, D) one "seed"
        per_arch_meta[arch] = {"layer_names": layer_names, "rdm_level": rdm_level}
        print(f"[PRETRAINED][{arch}] layers={len(layer_names)} vector_dim={E.shape[-1]}")

    return nod_rdms_per_layer, per_arch_meta


def _max_epoch_ckpt(files: List[Path]) -> Path:
    """Pick the checkpoint with the largest epochNNN number."""
    def _epoch_num(p: Path) -> int:
        m = re.search(r"epoch(\d+)", p.stem)
        return int(m.group(1)) if m else -1
    return max(files, key=_epoch_num)

def _load_imagenet_label_map(path_candidates):
    """
    Each line: '<wnid> <new name ...>'
    We keep only the FIRST word of the new name (punctuation stripped).
    """
    for p in path_candidates:
        if p and os.path.exists(p):
            mapping = {}
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) < 2:
                        continue
                    wnid = parts[0]
                    first_word = parts[1]
                    first_word = re.sub(r"[^A-Za-z0-9_\-]", "", first_word)
                    if first_word:
                        mapping[wnid] = first_word
            return mapping
    return {}

SKIP_DIRS = {"info"} 

def safe_loader(path: str):
    try:
        return default_loader(path)
    except Exception as e:
        print(f"[WARN] Could not load image: {path} ({e}); using a black placeholder.")
        return Image.new("RGB", (224, 224))

class FilteredImageFolder(ImageFolder):
    def __init__(self, root, transform=None, target_transform=None,
                 loader=None, is_valid_file=None, map_file=None):
        # self._label_map = _load_imagenet_label_map([
        #     os.path.join("../", "imagenet_label.txt")
        # ])

        label_map_candidates = []

        if map_file:
            label_map_candidates.append(map_file)
        label_map_candidates.extend([
            os.path.join(root, "..", "imagenet_label.txt"),
            os.path.join("../", "imagenet_label.txt"),
        ])
        self._label_map = _load_imagenet_label_map(label_map_candidates)


        if loader is None:
            loader = safe_loader

        super().__init__(root, transform=transform, target_transform=target_transform,
                         loader=loader, is_valid_file=is_valid_file)

        self.wnids = list(self.classes)
        self.classes = [self._label_map.get(wnid, wnid) for wnid in self.wnids]

        try:
            counts = np.bincount(self.targets, minlength=len(self.classes))
        except AttributeError:
            idxs = [cls_idx for _, cls_idx in self.samples]
            counts = np.bincount(np.asarray(idxs), minlength=len(self.classes))

        # print("[INFO] Class image counts:")
        # for i, cname in enumerate(self.classes):
        #     print(f"{cname}, {int(counts[i])}")

    def find_classes(self, directory: str):
        classes = [d.name for d in os.scandir(directory) if d.is_dir()]
        classes = [
            c for c in classes
            if (c.lower() not in SKIP_DIRS) and (not c.startswith("."))
        ]
        classes.sort()  # stable ordering for reproducibility
        if not classes:
            raise FileNotFoundError(f"No class folders found under: {directory}")
        class_to_idx = {c: i for i, c in enumerate(classes)}
        return classes, class_to_idx
    

class ClipWithHead(torch.nn.Module):
    def __init__(self, clip_model: nn.Module, out_dim: int):
        super().__init__()
        self.clip = clip_model
        # infer feature dim from encode_image output
        with torch.no_grad():
            dev = next(self.clip.parameters()).device
            dummy = torch.zeros(1, 3, 224, 224, device=dev)
            feat_dim = self.clip.encode_image(dummy).shape[-1]
        self.classifier = torch.nn.Linear(feat_dim, out_dim)

    def forward(self, x):
        feats = self.clip.encode_image(x)   # [B, D] (normalized)
        return self.classifier(feats)       # [B, out_dim]

    def encode_image(self, x):
        return self.clip.encode_image(x)

def _infer_head_out_dim(state_dict: dict, default: int = 100) -> int:
    # look for known head keys
    for k, v in state_dict.items():
        if k.endswith(("classifier.weight","probe_head.weight","head.weight")) and v.ndim == 2:
            return v.shape[0]  # out_dim
    return default

def run_nod_pipeline(ckpt_root: Path,
                     nod_root: Path,
                     arch_list: List[str],
                     subsample_val_batches: int,
                     batch_size: int,
                     num_workers: int,
                     device: str,
                     rdm_level: str = "class",
                     seed_ids: Optional[Iterable[int]] = None
    ):
    """
    For each architecture and seed, load the final checkpoint, run inference on the NOD dataset,
    and compute a correlation-distance RDM vector.
    Returns nod_rdms[arch] = array of shape (n_seeds, n_items*(n_items-1)/2)
    where n_items is the number of items in the NOD dataset.
    """
    device = torch.device(device)

    # Default (ImageNet) preprocessing for torchvision models
    preprocess_default = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Build a single default DataLoader (reused by non-CLIP models)
    nod_ds_default = FilteredImageFolder(nod_root, transform=preprocess_default)
    nod_loader_default = DataLoader(
        nod_ds_default,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    nod_rdms: Dict[str, np.ndarray] = {}

    for arch in arch_list:
        if arch == "clip_ViT-B/16":
            arch_dir = ckpt_root / "clip_vit_b16"
        else:
            arch_dir = ckpt_root / arch

        # seed_dirs = sorted(arch_dir.glob("seed_*"))
        seed_dirs = sorted(
            arch_dir.glob("seed_*"),
            key=lambda d: int(re.match(r"seed_(\d+)$", d.name).group(1)) if re.match(r"seed_(\d+)$", d.name) else 10**9
        )
        
        if seed_ids is not None:
            wanted = set(int(m.group(1)) for m in map(lambda d: re.match(r"seed_(\d+)$", d.name), seed_dirs) if m)
            wanted &= set(int(s) for s in seed_ids)
            seed_dirs = [d for d in seed_dirs if (m := re.match(r"seed_(\d+)$", d.name)) and int(m.group(1)) in wanted]
    
        all_vecs = []

        for seed_dir in seed_dirs:
            if arch == "clip_ViT-B/16":
                ckpt_files = list(seed_dir.glob("clip_vit_b16_epoch*.pth"))
            else:
                ckpt_files = list(seed_dir.glob(f"{arch}_epoch*.pth"))
            if not ckpt_files:
                continue

            ckpt_path = _max_epoch_ckpt(ckpt_files)

            # ----- Model selection -----
            is_clip = False
            if arch == "alexnet":
                model = Models.alexnet(weights=None)
                # keep classifier[6] shaped for your training setup
                model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, 100)
                nod_loader = nod_loader_default
            elif arch == "resnet50":
                model = Models.resnet50(weights=None)
                model.fc = torch.nn.Linear(model.fc.in_features, 100)
                nod_loader = nod_loader_default
            elif arch == "vit_b_16":
                model = Models.vit_b_16(weights=None)
                model.heads.head = torch.nn.Linear(model.heads.head.in_features, 100)
                nod_loader = nod_loader_default
            elif arch == "clip_RN50":
                # OpenAI CLIP expects its own preprocess and encode_image
                model, clip_preprocess = clip.load("RN50", device="cpu", jit=False)
                out_dim = _infer_head_out_dim(model.state_dict(), default=100)
                model = ClipWithHead(model, out_dim=out_dim)  # add a linear head for your training setup
                is_clip = True
                nod_ds_clip = FilteredImageFolder(nod_root, transform=clip_preprocess)
                nod_loader = DataLoader(
                    nod_ds_clip,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=True,
                )
            elif arch == "clip_ViT-B/16":
                # Note: correct model ID is "ViT-B/16" (case matters)
                model, clip_preprocess = clip.load("ViT-B/16", device="cpu", jit=False)
                out_dim = _infer_head_out_dim(model.state_dict(), default=100)
                model = ClipWithHead(model, out_dim=out_dim)  # add a linear head for your training setup
                is_clip = True
                nod_ds_clip = FilteredImageFolder(nod_root, transform=clip_preprocess)
                nod_loader = DataLoader(
                    nod_ds_clip,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=True,
                )
            else:
                # Generic torchvision fallback
                if not hasattr(Models, arch):
                    raise ValueError(f"Unknown architecture '{arch}'.")
                model = getattr(Models, arch)(weights=None)
                nod_loader = nod_loader_default

            model = model.to(device)

            # ----- Load weights (handle both raw and 'state_dict' wrapped) -----
            state = torch.load(ckpt_path, map_location=device)
            state_dict = state.get("state_dict", state)
            # If keys are prefixed (e.g., 'module.'), strip them
            if any(k.startswith("module.") for k in state_dict.keys()):
                state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict, strict=False)
            model.eval()

            # ----- Inference -----
            if rdm_level == "image":
                feats = []
                with torch.no_grad():
                    for i, (x, _) in enumerate(nod_loader):
                        if subsample_val_batches and i >= subsample_val_batches:
                            break
                        x = x.to(device, non_blocking=True)
                        if is_clip:
                            out = model.encode_image(x)
                        else:
                            out = model(x)
                        feats.append(out.view(out.size(0), -1).cpu().numpy())

                if not feats:
                    continue
            
            else:  # rdm_level == "class"
                class_sums = None
                class_counts = None
                n_classes = len(nod_ds_default.classes) if not is_clip else len(nod_ds_clip.classes)

                with torch.no_grad():
                    for i, (x, y) in enumerate(nod_loader):
                        if subsample_val_batches and i >= subsample_val_batches:
                            break
                        x = x.to(device, non_blocking=True)
                        if is_clip:
                            out = model.encode_image(x)
                        else:
                            out = model(x)
                        out_np = out.view(out.size(0), -1).cpu().numpy()
                        y_np = y.numpy()

                        if class_sums is None:
                            d = out_np.shape[1]
                            class_sums = np.zeros((n_classes, d), dtype=np.float64)
                            class_counts = np.zeros(n_classes, dtype=np.int64)

                        for c in np.unique(y_np):
                            idx = (y_np == c)
                            class_sums[c] += out_np[idx].sum(axis=0)
                            class_counts[c] += idx.sum()

                if class_sums is None or class_counts is None or np.any(class_counts == 0):
                    print(f"[NOD][{arch}][{seed_dir.name}] Incomplete class counts; skipping.")
                    continue

                feats = [class_sums[c] / class_counts[c] for c in range(n_classes)]

            feats = np.vstack(feats)
            _, v = _compute_rdm_vec(feats)
            all_vecs.append(v)

        if all_vecs:
            nod_rdms[arch] = np.stack(all_vecs, axis=0)
        else:
            nod_rdms[arch] = np.empty((0, 0), dtype=float)

        print(f"[NOD] Completed {arch}: {nod_rdms[arch].shape[0]} seeds, RDM vector dim {nod_rdms[arch].shape[1]}")
    return all_vecs, nod_rdms
    # return nod_rdms


def run_nod_pipeline_per_layer(
    ckpt_root: Path,
    nod_root: Path,
    arch_list: List[str],
    subsample_val_batches: int,
    batch_size: int,
    num_workers: int,
    device: str,
    rdm_level: str = "class",                     # "image" or "class"
    seed_ids: Optional[Iterable[int]] = None,     # restrict to subset of seeds
) -> Dict[str, Dict[str, Any]]:
    """
    For each architecture and seed:
      - forward NOD once while capturing the selected (non-leaf) layers via hooks,
      - convert each layer's activations -> [items, d] (GAP for CNN maps, CLS for ViTs),
      - compute an RDM per layer (matrix + vector),
      - stack per-seed results: per_seed_rdms_mat: (n_seeds, L, N, N), per_seed_rdms_vec: (n_seeds, L, D).

    Returns:
      per_arch = {
        arch: {
          "seed_ids": List[int],
          "layer_names": List[str],                  # length L (order is consistent across seeds)
          "rdm_level": str,                          # "image" | "class"
          "metric": "correlation",
          "per_seed_rdms_mat": np.ndarray,           # (n_seeds, L, N, N)
          "per_seed_rdms_vec": np.ndarray,           # (n_seeds, L, D)
        },
        ...
      }
    """
    assert rdm_level in ("image", "class"), "rdm_level must be 'image' or 'class'"
    metric = "correlation"
    device = torch.device(device)

    nod_rdms_per_layer = {}

    # --- Default (ImageNet-like) preprocessing for torchvision models ---
    preprocess_default = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Base loader used by non-CLIP models
    nod_ds_default = FilteredImageFolder(nod_root, transform=preprocess_default)
    nod_loader_default = DataLoader(
        nod_ds_default, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    per_arch: Dict[str, Dict[str, Any]] = {}

    for arch in arch_list:
        # --- Resolve arch dir & seeds (sorted by numeric id) ---
        arch_dir = ckpt_root / ("clip_vit_b16" if arch == "clip_ViT-B/16" else arch)
        seed_dirs = sorted(
            arch_dir.glob("seed_*"),
            key=lambda d: int(re.match(r"seed_(\d+)$", d.name).group(1)) if re.match(r"seed_(\d+)$", d.name) else 10**9
        )
        if seed_ids is not None:
            wanted = set(int(m.group(1)) for m in map(lambda d: re.match(r"seed_(\d+)$", d.name), seed_dirs) if m)
            wanted &= set(int(s) for s in seed_ids)
            seed_dirs = [d for d in seed_dirs if (m := re.match(r"seed_(\d+)$", d.name)) and int(m.group(1)) in wanted]
    
        if not seed_dirs:
            print(f"[NOD][{arch}] no seeds found after filtering; skipping.")
            continue

        layer_names_global: Optional[List[str]] = None
        per_seed_rdms_mat: List[np.ndarray] = []      # append (L, N, N)
        per_seed_rdms_vec: List[np.ndarray] = []      # append (L, D)
        kept_seed_ids: List[int] = []

        # --- Build a CLIP-aware loader if needed (once per arch) ---
        clip_preprocess = None  # lazily set when creating model for first seed
        nod_loader_clip = None

        for seed_dir in seed_dirs:
            # --- pick checkpoint ---
            if arch == "clip_ViT-B/16":
                ckpt_files = list(seed_dir.glob("clip_vit_b16_epoch*.pth"))
            else:
                ckpt_files = list(seed_dir.glob(f"{arch}_epoch*.pth"))
            if not ckpt_files:
                print(f"[NOD][{arch}][{seed_dir.name}] no checkpoints; skipping seed.")
                continue
            ckpt_path = _max_epoch_ckpt(ckpt_files)

            # --- build model & loader ---
            is_clip = False
            if arch == "alexnet":
                model = Models.alexnet(weights=None)
                model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, 100)
                nod_loader = nod_loader_default
            elif arch == "resnet50":
                model = Models.resnet50(weights=None)
                model.fc = torch.nn.Linear(model.fc.in_features, 100)
                nod_loader = nod_loader_default
            elif arch == "vit_b_16":
                model = Models.vit_b_16(weights=None)
                model.heads.head = torch.nn.Linear(model.heads.head.in_features, 100)
                nod_loader = nod_loader_default
            elif arch == "clip_RN50":
                if clip_preprocess is None:
                    base_model, clip_preprocess = clip.load("RN50", device="cpu", jit=False)
                out_dim = _infer_head_out_dim(base_model.state_dict(), default=100)
                model = ClipWithHead(base_model, out_dim=out_dim)
                is_clip = True
                if nod_loader_clip is None:
                    nod_ds_clip = FilteredImageFolder(nod_root, transform=clip_preprocess)
                    nod_loader_clip = DataLoader(
                        nod_ds_clip, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True,
                    )
                nod_loader = nod_loader_clip
            elif arch == "clip_ViT-B/16":
                if clip_preprocess is None:
                    base_model, clip_preprocess = clip.load("ViT-B/16", device="cpu", jit=False)
                out_dim = _infer_head_out_dim(base_model.state_dict(), default=100)
                model = ClipWithHead(base_model, out_dim=out_dim)
                is_clip = True
                if nod_loader_clip is None:
                    nod_ds_clip = FilteredImageFolder(nod_root, transform=clip_preprocess)
                    nod_loader_clip = DataLoader(
                        nod_ds_clip, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True,
                    )
                nod_loader = nod_loader_clip
            else:
                if not hasattr(Models, arch):
                    print(f"[NOD][{arch}] unknown arch; skipping.")
                    continue
                model = getattr(Models, arch)(weights=None)
                nod_loader = nod_loader_default

            model = model.to(device)

            # --- load checkpoint (handle wrapped dicts / 'module.' prefix) ---
            state = torch.load(ckpt_path, map_location=device)
            state_dict = state.get("state_dict", state)
            if any(k.startswith("module.") for k in state_dict.keys()):
                state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict, strict=False)
            model.eval()

            # --- register non-leaf layer hooks you selected for ridge ---
            layer_names, activations = _register_hooks_for_arch_2(model, arch)
            if layer_names_global is None:
                layer_names_global = list(layer_names)
            else:
                # Ensure identical layer order per seed
                if layer_names != layer_names_global:
                    print(f"[NOD][{arch}] WARN: layer list changed for {seed_dir.name}. Using first-seed order.")
                    # Optionally, re-map later; here we assume same order.

            # --- Collect per-layer features across items ---
            # We aggregate either per-image or per-class BEFORE computing RDM.
            if rdm_level == "image":
                # per-layer list of [B,d] chunks -> later vstack to [N_items,d]
                layer_feats: Dict[str, List[np.ndarray]] = {ln: [] for ln in layer_names_global}
                with torch.no_grad():
                    for bi, (x, y) in enumerate(nod_loader):
                        if subsample_val_batches and bi >= subsample_val_batches:
                            break
                        x = x.to(device, non_blocking=True)
                        if arch.startswith("clip_") or is_clip:
                            _ = model.encode_image(x)
                        else:
                            _ = model(x)

                        for ln in layer_names_global:
                            f = _feat_to_matrix(activations[ln], ln, batch_size=x.size(0))
                            layer_feats[ln].append(f.detach().cpu().numpy())

                # Build RDM per layer
                layer_rdms_mat: List[np.ndarray] = []
                layer_rdms_vec: List[np.ndarray] = []
                for ln in layer_names_global:
                    X = np.vstack(layer_feats[ln])                      # [N_items, d]
                    D, v = _compute_rdm_vec(X, metric=metric)          # D:[N,N], v:[D]
                    layer_rdms_mat.append(D.astype(np.float32))
                    layer_rdms_vec.append(v.astype(np.float32))

            else:  # rdm_level == "class"
                # per-layer class sums / counts -> class means -> RDM
                # Initialize lazily on first batch when we know d & n_classes
                n_classes = len(nod_loader.dataset.classes) if hasattr(nod_loader.dataset, "classes") else None
                class_sums: Dict[str, np.ndarray] = {}
                class_counts: Dict[str, np.ndarray] = {}

                with torch.no_grad():
                    for bi, (x, y) in enumerate(nod_loader):
                        if subsample_val_batches and bi >= subsample_val_batches:
                            break
                        x = x.to(device, non_blocking=True)
                        if arch.startswith("clip_") or is_clip:
                            _ = model.encode_image(x)
                        else:
                            _ = model(x)

                        y_np = y.numpy()
                        classes_in_batch = np.unique(y_np)
                        for ln in layer_names_global:
                            f = _feat_to_matrix(activations[ln], ln, batch_size=x.size(0)).detach().cpu().numpy()  # [B,d]
                            if ln not in class_sums:
                                assert n_classes is not None, "Cannot infer number of classes."
                                class_sums[ln] = np.zeros((n_classes, f.shape[1]), dtype=np.float64)
                                class_counts[ln] = np.zeros((n_classes,), dtype=np.int64)
                            for c in classes_in_batch:
                                idx = (y_np == c)
                                class_sums[ln][c] += f[idx].sum(axis=0)
                                class_counts[ln][c] += int(idx.sum())

                layer_rdms_mat, layer_rdms_vec = [], []
                for ln in layer_names_global:
                    if np.any(class_counts[ln] == 0):
                        raise RuntimeError(f"[NOD][{arch}][{seed_dir.name}] class {np.where(class_counts[ln]==0)[0]} has zero samples.")
                    Xc = (class_sums[ln] / class_counts[ln][:, None]).astype(np.float32)  # [C,d]
                    D, v = _compute_rdm_vec(Xc, metric=metric)
                    layer_rdms_mat.append(D.astype(np.float32))
                    layer_rdms_vec.append(v.astype(np.float32))

            # stack per seed: (L,N,N) and (L,D)
            L = len(layer_names_global)
            seed_rdms_mat = np.stack(layer_rdms_mat, axis=0)  # (L,N,N)
            seed_rdms_vec = np.stack(layer_rdms_vec, axis=0)  # (L,D)
            per_seed_rdms_mat.append(seed_rdms_mat)
            per_seed_rdms_vec.append(seed_rdms_vec)
            kept_seed_ids.append(int(seed_dir.name.split("_")[-1]))

            # clear hooks buffer for safety (not strictly required)
            activations.clear()

        if not per_seed_rdms_mat:
            print(f"[NOD][{arch}] no seeds produced RDMs; skipping arch.")
            continue

        per_arch[arch] = {
            "seed_ids": kept_seed_ids,
            "layer_names": layer_names_global or [],
            "rdm_level": rdm_level,
            "metric": metric,
            "per_seed_rdms_mat": np.stack(per_seed_rdms_mat, axis=0),  # (n_seeds, L, N, N)
            "per_seed_rdms_vec": np.stack(per_seed_rdms_vec, axis=0),  # (n_seeds, L, D)
        }

        ns, L, N, _ = per_arch[arch]["per_seed_rdms_mat"].shape
        D = per_arch[arch]["per_seed_rdms_vec"].shape[-1]
        print(f"[NOD][{arch}] per-layer RDMs ready: seeds={ns}, layers={L}, N={N}, D={D}")

        nod_rdms_per_layer[arch] = per_arch[arch].get("per_seed_rdms_vec", np.empty((0,0,0), dtype=float))


    return nod_rdms_per_layer, per_arch

    

def run_mri_weighting(
    fmri_rdm_file: str | Path,
    pca_dim: int = 50,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load fMRI RDMs, vectorize, PCA reduce, perform iterative weighting.

    For each subject:
      1) Stack that subject's ROI vectors -> (R, D)
      2) Apply iterative reweighting across ROIs to get a *subject embedding*
         (the ROI-reweighted mean for that subject).

    Then:
      3) Stack subject embeddings -> (S, D)
      4) Apply iterative reweighting across subjects to get the final mean.

    Returns (E, w, rdms_vec) where E is [subs, pca_dim], w is [subs], rdms_vec original vectors.
    """
    # # load array or npz
    # arr = np.load(fmri_rdm_file)
    # rdms = arr.get('rdms', arr)  # shape [subs, items, items]
    # n_subj = rdms.shape[0]
    # # vectorize upper triangle
    # vecs = []
    # for i in range(n_subj):
    #     dm = rdms[i]
    #     vecs.append(dm[np.triu_indices(dm.shape[0], k=1)])
    # rdms_vec = np.stack(vecs, axis=0)
    
    
    base = Path(fmri_rdm_file)
    sub_dirs = sorted(base.glob("sub-*"))
    if not sub_dirs:
        raise FileNotFoundError(f"No subject directories matching 'sub-*' under: {base}")
    
    subj_embeddings = []
    vecs = []
    for sub in sub_dirs:
        roi_files = sorted(sub.glob("*.npy"))
        roi_vecs = [np.load(f).astype(np.float32) for f in roi_files]
        first_shape = roi_vecs[0].shape
        if any(p.shape != first_shape for p in roi_vecs):
            raise ValueError(f"ROI vector shapes differ for {sub}")

        # --- Step 1: stack ROIs for this subject ---
        E_roi = np.stack(roi_vecs, axis=0).astype(np.float32)  # (R, D)

        # --- Step 2: iterative reweight across ROIs to get a subject embedding ---
        _, subj_mean, _, _ = _iterative_reweight(E_roi)    # subj_mean: (D,)
        subj_embeddings.append(subj_mean.astype(np.float32))

    # --- Step 3: stack subject embeddings ---
    E_subj = np.stack(subj_embeddings, axis=0).astype(np.float32)
    
    # Simple (unweighted) mean across subjects
    simple_mean_vec = E_subj.mean(axis=0).astype(np.float32)

    w, E_mean, _, E_norm = _iterative_reweight(E_subj)

    return simple_mean_vec, E_mean, w, E_norm

def spearman_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman between two 1D vectors (robust to ties)."""
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
    # if either vector is constant, rho is undefined -> treat as max dissimilar
    if np.all(a == a[0]) or np.all(b == b[0]):
        return np.nan
    rho, _ = spearmanr(a, b)
    if np.isnan(rho):  # e.g., one vector is constant (no rank variance)
        return 0.0
    return float(np.clip(rho, -1.0, 1.0))


def run_seeds_reweighting(
    all_vecs: Dict[str, np.ndarray] | List[np.ndarray] | np.ndarray,  # final-output RDM vectors, one per seed
    arch: str,
    fmri_simple_mean: np.ndarray,              # (d,)
    fmri_reweighted_mean: np.ndarray,          # (d,)
    do_plot: bool = True,
) -> Dict[str, Any]:
    """
    Operates on FINAL output RDM vectors (no per-layer). Steps:
      1) Stack seeds -> E_raw [n_seeds, d]
      2) Compute simple mean (raw) and cosine distance to fmri_simple_mean
      3) Iterative reweight -> weights and normalized embeddings
      4) Compute weighted mean (raw) with learned weights and cosine distance to fmri_reweighted_mean
      5) Save NPZ (embeddings=E_norm, weights=w), run Black Swan analysis, collect results
    Returns a dict with all artifacts and distances.
    """
    # seed_vecs = all_vecs[arch]
    # # --- stack to [n_seeds, d] ---
    # if isinstance(seed_vecs, list):
    #     E_raw = np.stack([np.asarray(v, dtype=np.float32) for v in seed_vecs], axis=0)
    # else:
    #     E_raw = np.asarray(seed_vecs, dtype=np.float32)

    if isinstance(all_vecs, dict):
        seed_vecs = all_vecs[arch]
    elif isinstance(all_vecs, list):
        seed_vecs = np.stack([np.asarray(v, dtype=np.float32) for v in all_vecs], axis=0)
    else:
        seed_vecs = np.asarray(all_vecs, dtype=np.float32)

    E_raw = np.asarray(seed_vecs, dtype=np.float32)
    if E_raw.ndim != 2:
        raise ValueError(f"Expected [n_seeds, d], got {E_raw.shape}")

    # --- sanity: dims must match fMRI vectors ---
    d0 = E_raw.shape[-1]
    if fmri_simple_mean.shape[-1] != d0 or fmri_reweighted_mean.shape[-1] != d0:
        raise ValueError(
            f"[{arch}] Dim mismatch: seed dim={d0}, "
            f"fmri_simple_dim={fmri_simple_mean.shape[-1]}, "
            f"fmri_reweighted_dim={fmri_reweighted_mean.shape[-1]}"
        )

    # --- simple (unweighted) mean in RAW space ---
    seed_simple_mean = E_raw.mean(axis=0).astype(np.float32)
    # cos_to_fmri_simple = spearman_distance(seed_simple_mean[None, :], fmri_simple_mean[None, :])
    cos_to_fmri_simple     = spearman_distance(seed_simple_mean, fmri_simple_mean)


    # --- reweighted mean in RAW space using learned weights ---
    print(f"[TEST][{arch}] Running iterative reweighting (final output)...")
    w, E_mean_norm, iters, E_norm = _iterative_reweight(E_raw)
    seed_weighted_mean = ((w[:, None] * E_raw).sum(axis=0) / w.sum()).astype(np.float32)
    if do_plot:
        plot_weights(w, layer_name="final_output", arch_name=arch, outdir="final_output_weights")

    # cos_to_fmri_reweighted = spearman_distance(seed_weighted_mean[None, :], fmri_reweighted_mean[None, :])
    cos_to_fmri_reweighted = spearman_distance(seed_weighted_mean, fmri_reweighted_mean)


    # --- assemble result ---
    result: Dict[str, Any] = {
        # raw-space summaries
        "seed_simple_mean": seed_simple_mean,                    # (d,)
        "seed_weighted_mean_raw": seed_weighted_mean,            # (d,)
        "spearman_to_fmri_simple": cos_to_fmri_simple,
        "spearman_to_fmri_reweighted": cos_to_fmri_reweighted,

        # reweighting outputs (normalized space)
        "embeddings": E_norm,                                    # (n_seeds, d) normalized
        "weights": w,                                            # (n_seeds,)
        "converged_mean_norm": E_mean_norm,                      # (d,)
        "convergence_iterations": iters,
    }

    print(f"Cos(seed_simple, fmri_simple) = {cos_to_fmri_simple:.6f}")
    print(f"Cos(seed_weighted, fmri_reweighted) = {cos_to_fmri_reweighted:.6f}")

    return result


def compare_outliers(
    seed_embeds: dict[str, np.ndarray],     # {arch: (n_seeds, d)}
    human_embeds: np.ndarray | None = None, # (n_humans, d) OR None
    human_w: np.ndarray | None = None,      # (n_humans,)   OR None
    z_thresh: float = -1.0,
    human_mean: np.ndarray | None = None,   # <-- new: (d,)
) -> dict:
    
    results: dict[str, Any] = {
        "seed2human_dist": {},
        "seed2human_z": {},
        "seed_outliers": {},
        "seed_outlier_details": {},
        "seed_vs_human_matrix": None,
    }

    # establish HUMAN reference vector
    if human_mean is not None:
        H_mean = np.asarray(human_mean, dtype=float)
        H = np.asarray(human_embeds, dtype=float) if human_embeds is not None else None
    else:
        if human_embeds is None or human_w is None:
            raise ValueError("Provide either human_mean or (human_embeds, human_w).")
        H = np.nan_to_num(np.asarray(human_embeds, dtype=float))
        hw = np.asarray(human_w, dtype=float)
        H_mean = np.average(H, axis=0, weights=hw)
    
    if human_embeds is None or human_w is None or len(human_embeds) == 0:
        print("[WARN] No human embeddings/weights provided; skipping distance-based outliers.")
        # Still return empty containers for seeds
        for arch, E in seed_embeds.items():
            results["seed2human_dist"][arch] = np.array([])
            results["seed2human_z"][arch] = np.array([])
            results["seed_outliers"][arch] = np.array([], dtype=int)
            results["seed_outlier_details"][arch] = {}
        return results

    # Keep full seed�human matrices too (useful for diagnostics/plots)
    full_mats: dict[str, np.ndarray] = {}

    # ---- Per-arch: seed distances to HUMAN mean, then z-score, then outliers ----
    for arch, E in seed_embeds.items():
        E = np.nan_to_num(np.asarray(E, dtype=float))  # (n_seeds, d)
        if E.ndim != 2 or E.shape[0] == 0:
            # Nothing to do
            results["seed2human_dist"][arch] = np.array([])
            results["seed2human_z"][arch] = np.array([])
            results["seed_outliers"][arch] = np.array([], dtype=int)
            results["seed_outlier_details"][arch] = {}
            continue

        # Distances of each seed to HUMAN mean
        d_mean = cosine_distances(E, H_mean[None, :]) # (n_seeds,)
        mu = d_mean.mean()
        sigma = d_mean.std(ddof=1) + 1e-9
        z = (d_mean - mu) / sigma

        out_idx = np.where(z < z_thresh)[0]

        # Save + print
        results["seed2human_dist"][arch] = d_mean
        results["seed2human_z"][arch] = z
        results["seed_outliers"][arch] = out_idx
        results["seed_outlier_details"][arch] = {
            int(i): {"distance": float(d_mean[i]), "z": float(z[i])} for i in out_idx
        }
        print(f"[{arch}] seed�HUMAN-mean: �={mu:.6f}, �={sigma:.6f}, z<{z_thresh:.2f} outliers: {out_idx.tolist()}")

        # Also keep the full seed�human matrix (raw distances)
        mat = cosine_distances(E, H)  # (n_seeds, n_humans)
        full_mats[arch] = mat

    results["seed_vs_human_matrix"] = full_mats

    return results



def run_mri_weighting_by_roi(
    fmri_rdm_dir: str | Path,
) -> Dict[str, Dict[str, Any]]:
    """
    For each ROI, stack subjects' ROI vectors (S, D) and reweight ACROSS SUBJECTS.
    Keeps only ROIs present in all subjects with consistent vector length.

    Returns:
      roi_results[roi] = {
        "subject_ids": [ ... ],
        "simple_mean": (D,),             # unweighted mean across subjects (raw space)
        "reweighted_mean": (D,),         # normalized-space mean from _iterative_reweight
        "weights": (S,),
        "embeddings": (S, D),            # normalized embeddings used by the loop
        "vector_dim": int,
        "convergence_iterations": int,
      }
    """
    base = Path(fmri_rdm_dir)
    sub_dirs = sorted(base.glob("sub-*"))
    if not sub_dirs:
        raise FileNotFoundError(f"No subject directories matching 'sub-*' under: {base}")

    subj_ids = [d.name for d in sub_dirs]
    # discover ROIs common to all subjects
    roi_sets = []
    for sd in sub_dirs:
        roi_sets.append({p.stem for p in sd.glob("*.npy")})
    common_rois = set.intersection(*roi_sets)
    if not common_rois:
        raise RuntimeError("No ROI present in all subjects.")

    roi_results: Dict[str, Dict[str, Any]] = {}
    for roi in sorted(common_rois):
        vecs: List[np.ndarray] = []
        dims: List[int] = []
        for sd in sub_dirs:
            v = np.load(sd / f"{roi}.npy").astype(np.float32)
            vecs.append(v)
            dims.append(v.size)
        if len(set(dims)) != 1:
            print(f"[WARN] ROI '{roi}' skipped due to inconsistent dims across subjects: {set(dims)}")
            continue

        E_subj = np.stack(vecs, axis=0).astype(np.float32)  # (S, D)
        simple_mean = E_subj.mean(axis=0).astype(np.float32)
        w, E_mean, iters, E_norm = _iterative_reweight(E_subj)

        roi_results[roi] = {
            "subject_ids": subj_ids,
            "simple_mean": simple_mean,
            "reweighted_mean": E_mean.astype(np.float32),
            "weights": w.astype(np.float32),
            "embeddings": E_norm,
            "vector_dim": int(E_subj.shape[1]),
            "convergence_iterations": int(iters),
        }
    if not roi_results:
        raise RuntimeError("No ROI survived consistency checks.")
    return roi_results


def compute_seed_roi_alignment(
    nod_rdms: Dict[str, np.ndarray],         # {arch: (n_seeds, D)}
    roi_results: Dict[str, Dict[str, Any]],  # output of run_mri_weighting_by_roi
    use_seed_weights: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """
    Records *all seeds'* Spearman distances to:
      - ROI SIMPLE mean
      - ROI REWEIGHTED mean

    Returns:
      align = {
        <arch>: {
          <roi>: {
            "seed_ids": [...],                         # kept seed indices
            "d_seed_vs_roi_simple": [...],             # len = n_seeds_used
            "d_seed_vs_roi_reweighted": [...],         # len = n_seeds_used
            "mean_d_seed_vs_roi_simple": float,
            "mean_d_seed_vs_roi_reweighted": float,
            "std_d_seed_vs_roi_simple": float,
            "std_d_seed_vs_roi_reweighted": float,
            "best_seed_simple": {"seed_index": int, "distance": float},
            "best_seed_reweighted": {"seed_index": int, "distance": float},
            "d_seedmean_vs_roi_simple": float,         # legacy mean-vs-mean
            "d_weighted_seedmean_vs_roi_reweighted": float,
            "n_seeds_used": int,
            "vector_dim": int,
          },
          ...
        },
        "_flat_records": [
          {
            "arch": str,
            "seed_index": int,
            "roi": str,
            "d_vs_roi_simple": float,
            "d_vs_roi_reweighted": float,
            "vector_dim": int,
          },
          ...
        ]
      }
    """
    out: Dict[str, Dict[str, Any]] = {}
    flat_records: List[Dict[str, Any]] = []

    for arch, E in nod_rdms.items():
        E = np.asarray(E, dtype=np.float32)
        if E.ndim != 2 or E.size == 0:
            print("[SKIP] E size: ", E.size, "dim: ", E.ndim)
            continue

        # Filter out seeds with non-finite entries
        finite_mask = np.isfinite(E).all(axis=1)
        if not np.any(finite_mask):
            print("[SKIP] infinite: ", arch)
            continue
        E_used = E[finite_mask]
        seed_ids = [i for i, keep in enumerate(finite_mask) if keep]

        # Seed means (legacy mean-vs-mean)
        seed_simple_mean = E_used.mean(axis=0).astype(np.float32)
        if use_seed_weights and E_used.shape[0] >= 2:
            try:
                w_s, _, _, _ = _iterative_reweight(E_used)
                w_s = np.asarray(w_s, dtype=np.float32)
                seed_weighted_mean = ((w_s[:, None] * E_used).sum(axis=0) / w_s.sum()).astype(np.float32)
            except Exception as e:
                print(f"[WARN] _iterative_reweight (seeds) failed for arch '{arch}': {e}  fallback to simple mean.")
                seed_weighted_mean = seed_simple_mean
        else:
            seed_weighted_mean = seed_simple_mean

        arch_res: Dict[str, Any] = {}

        for roi, info in roi_results.items():
            roi_simple = np.asarray(info["simple_mean"], dtype=np.float32).ravel()
            roi_rew   = np.asarray(info["reweighted_mean"], dtype=np.float32).ravel()

            D = E_used.shape[1]
            if roi_simple.shape[0] != D or roi_rew.shape[0] != D:
                # skip mismatched dimensionality
                print("[SKIP] roi simple shape: ", roi_simple.shape[0], "roi rew shape ", roi_rew.shape[0])
                continue

            # Per-seed distances
            d_seed_vs_roi_simple = np.empty(E_used.shape[0], dtype=np.float32)
            d_seed_vs_roi_rew    = np.empty(E_used.shape[0], dtype=np.float32)

            for idx_in_used, v in enumerate(E_used):
                d_seed_vs_roi_simple[idx_in_used] = float(spearman_distance(v, roi_simple))
                d_seed_vs_roi_rew[idx_in_used]    = float(spearman_distance(v, roi_rew))

                # ---- record every seeds distances (flat record) ----
                flat_records.append({
                    "arch": arch,
                    "seed_index": seed_ids[idx_in_used],
                    "roi": roi,
                    "d_vs_roi_simple": float(d_seed_vs_roi_simple[idx_in_used]),
                    "d_vs_roi_reweighted": float(d_seed_vs_roi_rew[idx_in_used]),
                    "vector_dim": int(D),
                })

            # Aggregates
            mean_simple = float(np.mean(d_seed_vs_roi_simple))
            mean_rew    = float(np.mean(d_seed_vs_roi_rew))
            std_simple  = float(np.std(d_seed_vs_roi_simple, ddof=0))
            std_rew     = float(np.std(d_seed_vs_roi_rew, ddof=0))

            best_idx_simple = int(np.argmin(d_seed_vs_roi_simple))
            best_idx_rew    = int(np.argmin(d_seed_vs_roi_rew))

            # Legacy mean-vs-mean
            d_seedmean_vs_roi_simple = float(spearman_distance(seed_simple_mean,   roi_simple))
            d_wseedmean_vs_roi_rew   = float(spearman_distance(seed_weighted_mean, roi_rew))

            d_seedmean_vs_roi_rew = float(spearman_distance(seed_simple_mean,   roi_rew))
            d_wseedmean_vs_roi_simple   = float(spearman_distance(seed_weighted_mean, roi_simple))

            arch_res[roi] = {
                "seed_ids": seed_ids,
                "d_seed_vs_roi_simple": d_seed_vs_roi_simple.tolist(),
                "d_seed_vs_roi_reweighted": d_seed_vs_roi_rew.tolist(),

                "mean_d_seed_vs_roi_simple": mean_simple,
                "mean_d_seed_vs_roi_reweighted": mean_rew,
                "std_d_seed_vs_roi_simple": std_simple,
                "std_d_seed_vs_roi_reweighted": std_rew,
                "best_seed_simple": {
                    "seed_index": seed_ids[best_idx_simple],
                    "distance": float(d_seed_vs_roi_simple[best_idx_simple]),
                },
                "best_seed_reweighted": {
                    "seed_index": seed_ids[best_idx_rew],
                    "distance": float(d_seed_vs_roi_rew[best_idx_rew]),
                },

                "d_seedmean_vs_roi_simple": d_seedmean_vs_roi_simple,
                "d_weighted_seedmean_vs_roi_reweighted": d_wseedmean_vs_roi_rew,
                "d_seedmean_vs_roi_reweighted": d_seedmean_vs_roi_rew,
                "d_weighted_seedmean_vs_roi_simple": d_wseedmean_vs_roi_simple,

                "n_seeds_used": int(E_used.shape[0]),
                "vector_dim": int(D),
            }

        out[arch] = arch_res

    # Attach flat records at the top level for easy CSV export
    out["_flat_records"] = flat_records
    return out

def _roi_name_from_filename(p: Path) -> str:
    """
    Extract ROI name as the first token in the filename before the first underscore.
    Examples:
      LO1_rdm_vectors.npy      -> LO1
      V1.npy                   -> V1
      FFC-Right_roi_vec.npy    -> FFC-Right  (keeps hyphens)
    """
    stem = p.stem  # filename without extension
    # Split on the first underscore; if none, whole stem is the ROI name
    roi = stem.split("_", 1)[0]
    return roi

def _load_roi_subject_vectors(fmri_root: Path) -> dict[str, np.ndarray]:
    """
    fmri_root/
      sub-01/ LO1_rdm_vectors.npy, V1_rdm_vectors.npy, ...
      sub-02/ LO1_rdm_vectors.npy, V1_rdm_vectors.npy, ...
    Returns: {roi_name: array [n_subjects, d]}
    """
    fmri_root = Path(fmri_root)
    sub_dirs = sorted(fmri_root.glob("sub-*"))
    if not sub_dirs:
        raise FileNotFoundError(f"No subject directories under: {fmri_root}")

    # Discover the ROI set from the first subject
    first = sub_dirs[0]
    first_files = sorted(first.glob("*.npy"))
    if not first_files:
        raise FileNotFoundError(f"No ROI .npy files under: {first}")
    roi_names_first = { _roi_name_from_filename(p): p for p in first_files }

    # Verify each subject has the same ROI set (by ROI name) and collect vectors
    roi_to_rows: dict[str, list[np.ndarray]] = {roi: [] for roi in roi_names_first.keys()}

    for sub in sub_dirs:
        files = sorted(sub.glob("*.npy"))
        if not files:
            raise FileNotFoundError(f"No ROI .npy files under: {sub}")

        # Map this subjects files by parsed ROI name
        this_map: dict[str, Path] = {}
        for p in files:
            roi = _roi_name_from_filename(p)
            this_map[roi] = p

        missing = sorted(set(roi_names_first.keys()) - set(this_map.keys()))
        extra   = sorted(set(this_map.keys()) - set(roi_names_first.keys()))
        if missing:
            raise FileNotFoundError(f"{sub.name} missing ROI(s): {missing}")
        if extra:
            # Not fatalignore unknown ROIs to keep the set consistent
            pass

        # Load in a consistent ROI order
        for roi in roi_names_first.keys():
            v = np.load(this_map[roi]).astype(np.float32).ravel()
            roi_to_rows[roi].append(v)

    # Stack per-ROI into [n_subjects, d]
    roi_to_mat: dict[str, np.ndarray] = {}
    for roi, rows in roi_to_rows.items():
        mat = np.stack(rows, axis=0).astype(np.float32)
        # sanity: all subjects must share dimensionality
        if np.unique([row.shape[0] for row in rows]).size != 1:
            raise ValueError(f"Dim mismatch across subjects for ROI {roi}")
        roi_to_mat[roi] = mat

    return roi_to_mat

def compute_fmri_subject_vs_seed_means(
    roi_name: str,
    roi_subject_vecs: np.ndarray,   # [n_subjects, d]
    seed_vecs: np.ndarray,          # [n_seeds, d]
) -> dict:
    """
    For a single ROI:
      - Seed simple mean & weighted mean (RAW).
      - Per-subject Spearman rho to both seed means.
      - fMRI simple mean & weighted mean (RAW).
      - All 4 mean-vs-mean Spearman rhos.
    """
    H = np.asarray(roi_subject_vecs, dtype=np.float32)  # [N_subj, d]
    E = np.asarray(seed_vecs,        dtype=np.float32)  # [N_seed, d]
    if H.ndim != 2 or E.ndim != 2:
        raise ValueError(f"Expected 2D arrays: got H {H.shape}, E {E.shape}")
    if H.shape[1] != E.shape[1]:
        raise ValueError(f"Dim mismatch: subjects d={H.shape[1]} vs seeds d={E.shape[1]}")

    # Seed means
    seed_simple = E.mean(axis=0).astype(np.float32)
    w_seed, seed_weighted , _, _ = _iterative_reweight(E)

    # Per-subject rhos
    subj_rho_to_seed_simple   = np.array([spearman_distance(h, seed_simple)   for h in H], dtype=np.float32)
    subj_rho_to_seed_weighted = np.array([spearman_distance(h, seed_weighted) for h in H], dtype=np.float32)

    # fMRI (subject) means
    fmri_simple = H.mean(axis=0).astype(np.float32)
    w_subj, fmri_weighted , _, _ = _iterative_reweight(H)

    # All four combos
    rho_fsimple_ssimple       = float(spearman_distance(fmri_simple,   seed_simple))
    rho_fsimple_sweighted     = float(spearman_distance(fmri_simple,   seed_weighted))
    rho_fweighted_ssimple     = float(spearman_distance(fmri_weighted, seed_simple))
    rho_fweighted_sweighted   = float(spearman_distance(fmri_weighted, seed_weighted))

    return {
        "roi": roi_name,
        "n_subjects": int(H.shape[0]),
        "n_seeds": int(E.shape[0]),
        "subject_rho_to_seed_simple":   subj_rho_to_seed_simple,     # [n_subjects]
        "subject_rho_to_seed_weighted": subj_rho_to_seed_weighted,   # [n_subjects]
        "seed_weights": w_seed,           # [n_seeds]
        "subject_weights": w_subj,        # [n_subjects]
        "fmri_seed_combo_rhos": {
            "fmri_simple_vs_seed_simple":     rho_fsimple_ssimple,
            "fmri_simple_vs_seed_weighted":   rho_fsimple_sweighted,
            "fmri_weighted_vs_seed_simple":   rho_fweighted_ssimple,
            "fmri_weighted_vs_seed_weighted": rho_fweighted_sweighted,
        },
    }


def run_fmri_subject_vs_seed_means(
    cfg: dict,
    args,
    ckpt_root: Path,
    nod_rdms: Dict[str, np.ndarray],     # from run_nod_pipeline / NPZ load: {arch: [n_seeds, d]}
) -> Dict[str, Any]:
    """
    Runner that, for each ROI and each architecture, computes:
      - per-subject Spearman rho to seed simple/weighted means,
      - the four mean-vs-mean Spearman rhos (fmri simple/weighted � seed simple/weighted),
    and writes:
      - summary CSV (one row per (arch, ROI) with the four rhos),
      - per-subject CSV (one row per (arch, ROI, subject) with the two rhos).
    """
    fmri_root = Path(args.fmri_rdm_file or cfg["data"]["fmri_rdm_file"])
    if not fmri_root.exists():
        raise FileNotFoundError(f"fMRI RDM path not found: {fmri_root}")

    out_root = Path(cfg["paths"].get("output_root", ckpt_root / "outputs"))
    out_root.mkdir(parents=True, exist_ok=True)

    # Load ROI�[subjects�d]
    roi_to_subject_mat = _load_roi_subject_vectors(fmri_root)

    # CSVs
    summary_csv   = out_root / f"roi_fmriSubject_vs_seedMeans_SUMMARY_{args.rdm_level}.csv"
    per_subj_csv  = out_root / f"roi_fmriSubject_vs_seedMeans_PER_SUBJECT_{args.rdm_level}.csv"

    with open(summary_csv, "w", newline="") as fsum, open(per_subj_csv, "w", newline="") as fsubj:
        sw = csv.writer(fsum)
        pw = csv.writer(fsubj)

        sw.writerow([
            "arch","roi","n_subjects","n_seeds",
            "rho_fmriSimple_vs_seedSimple",
            "rho_fmriSimple_vs_seedWeighted",
            "rho_fmriWeighted_vs_seedSimple",
            "rho_fmriWeighted_vs_seedWeighted",
        ])
        pw.writerow([
            "arch","roi","subject_index",
            "rho_subject_vs_seedSimple",
            "rho_subject_vs_seedWeighted",
        ])

        for arch, E in nod_rdms.items():
            E = np.asarray(E, dtype=np.float32)
            if E.ndim != 2 or E.size == 0:
                print(f"[SKIP] {arch}: empty seed matrix.")
                continue

            for roi_name, H in roi_to_subject_mat.items():
                try:
                    res = compute_fmri_subject_vs_seed_means(roi_name, H, E)
                except Exception as e:
                    print(f"[WARN] Failed ({arch}, {roi_name}): {e}")
                    continue

                rhos = res["fmri_seed_combo_rhos"]
                sw.writerow([
                    arch, roi_name, res["n_subjects"], res["n_seeds"],
                    f"{rhos['fmri_simple_vs_seed_simple']:.6f}",
                    f"{rhos['fmri_simple_vs_seed_weighted']:.6f}",
                    f"{rhos['fmri_weighted_vs_seed_simple']:.6f}",
                    f"{rhos['fmri_weighted_vs_seed_weighted']:.6f}",
                ])

                subj_simple   = res["subject_rho_to_seed_simple"]
                subj_weighted = res["subject_rho_to_seed_weighted"]
                for si, (rs, rw) in enumerate(zip(subj_simple, subj_weighted)):
                    pw.writerow([arch, roi_name, si, f"{rs:.6f}", f"{rw:.6f}"])

    print(f"[INF] Saved subject-vs-seedMeans SUMMARY:     {summary_csv}")
    print(f"[INF] Saved subject-vs-seedMeans PER-SUBJECT: {per_subj_csv}")
    return {
        "summary_csv": str(summary_csv),
        "per_subject_csv": str(per_subj_csv),
    }

# def _ridge_fit_single_target(X: np.ndarray,
#                              y: np.ndarray,
#                              alphas: np.ndarray,
#                              cv_folds: int = 5,
#                              eps: float = 1e-8):

#     X = np.asarray(X, np.float32)
#     y = np.asarray(y, np.float32).ravel()
#     N, L = X.shape

#     # Drop non-finite and zero-variance columns (per feature/layer)
#     finite_cols = np.isfinite(X).all(axis=0)
#     std_cols    = X.std(axis=0) > 0
#     keep_mask   = finite_cols & std_cols
#     kept_idx    = np.where(keep_mask)[0]

#     if kept_idx.size == 0:
#         return np.zeros(L, np.float32), float(alphas[0] if alphas.size else 1.0), float('-inf')

#     Xk = X[:, kept_idx]
#     Xk = Xk - Xk.mean(axis=0, keepdims=True)

#     # K-fold guard
#     k = min(cv_folds, max(2, Xk.shape[0]))  # at least 2 if possible
#     if k < 2:
#         a_fit = float(max(np.max(alphas) if alphas.size else 1.0, eps))
#         XtX, XtY = Xk.T @ Xk, Xk.T @ y
#         A = XtX + (a_fit + eps) * np.eye(Xk.shape[1], dtype=np.float32)
#         try:
#             wk = np.linalg.solve(A, XtY).astype(np.float32)
#         except np.linalg.LinAlgError:
#             wk = (np.linalg.pinv(A) @ XtY).astype(np.float32)
#         w_full = np.zeros(L, np.float32); w_full[kept_idx] = wk
#         return w_full, a_fit, float('nan')

#     kf = KFold(n_splits=k, shuffle=True, random_state=0)

#     def _fit(Xtr, ytr, a):
#         XtX, XtY = Xtr.T @ Xtr, Xtr.T @ ytr
#         A = XtX + (a + eps) * np.eye(Xtr.shape[1], dtype=np.float32)
#         try:
#             w = np.linalg.solve(A, XtY)
#         except np.linalg.LinAlgError:
#             w = np.linalg.pinv(A) @ XtY
#         return w.astype(np.float32)

#     best_a, best_score = None, -np.inf
#     for a in alphas.astype(np.float32):
#         a = float(max(a, eps))
#         fold_scores = []
#         for tr, vl in kf.split(Xk):
#             Xtr, Xvl = Xk[tr], Xk[vl]
#             ytr, yvl = y[tr],  y[vl]
#             wk = _fit(Xtr, ytr, a)
#             yvl_pred = Xvl @ wk
#             # R^2 on centered targets
#             yvl_c = yvl - yvl.mean()
#             res   = yvl_c - (yvl_pred - yvl_pred.mean())
#             res64   = res.astype(np.float64, copy=False)
#             yvlc64  = yvl_c.astype(np.float64, copy=False)
#             ss_res  = float(np.dot(res64,  res64)) 
#             ss_tot  = float(np.dot(yvlc64, yvlc64) + 1e-12)
#             fold_scores.append(1.0 - ss_res/ss_tot)
#         mean_r2 = float(np.mean(fold_scores)) if fold_scores else float('-inf')
#         if mean_r2 > best_score:
#             best_score, best_a = mean_r2, a

#     wk = _fit(Xk, y, best_a if best_a is not None else 1.0)
#     w_full = np.zeros(L, np.float32); w_full[kept_idx] = wk
#     return w_full, float(best_a if best_a is not None else 1.0), float(best_score)


def _ridge_fit_single_target(X: np.ndarray,
                             y: np.ndarray,
                             alphas: np.ndarray,
                             cv_folds: int = 5,
                             eps: float = 1e-8):
    """
    Single-target ridge with:
      - drop non-finite / zero-variance cols
      - column-centering
      - primal/dual auto
      - CV over given alphas
    """
    X = np.asarray(X, np.float32)
    y = np.asarray(y, np.float32).ravel()
    N, L = X.shape

    # Keep finite, non-constant columns
    finite_cols = np.isfinite(X).all(axis=0)
    std_cols    = X.std(axis=0) > 0
    keep_mask   = finite_cols & std_cols
    kept_idx    = np.where(keep_mask)[0]

    if kept_idx.size == 0:
        return np.zeros(L, np.float32), float(alphas[0] if alphas.size else 1.0), float('-inf')

    Xk = X[:, kept_idx].astype(np.float32, copy=False)
    # Global column-centering (fast; consistent with R^2 centering below)
    Xk = Xk - Xk.mean(axis=0, keepdims=True)

    # Minimal guard for small N
    k = min(cv_folds, max(2, Xk.shape[0]))
    if k < 2:
        a_fit = float(max(np.max(alphas) if alphas.size else 1.0, eps))
        # final fit with Cholesky in float64
        XtX = (Xk.T @ Xk).astype(np.float64, copy=False)
        XtY = (Xk.T @ y).astype(np.float64, copy=False)
        A = XtX + (a_fit + eps) * np.eye(Xk.shape[1], dtype=np.float64)
        try:
            Lc = np.linalg.cholesky(A)
            wk = np.linalg.solve(Lc.T, np.linalg.solve(Lc, XtY)).astype(np.float32, copy=False)
        except np.linalg.LinAlgError:
            wk = (np.linalg.pinv(A) @ XtY).astype(np.float32, copy=False)
        w_full = np.zeros(L, np.float32); w_full[kept_idx] = wk
        return w_full, a_fit, float('nan')

    # Cache splits once
    kf = KFold(n_splits=k, shuffle=True, random_state=0)
    splits = list(kf.split(Xk))

    # Decide primal vs dual per fold based on shape
    use_dual = (L > 1.5 * N)

    # Helper: compute mean-centered R^2 quickly
    def r2_centered(y_true, y_pred):
        y_true = y_true.astype(np.float64, copy=False)
        y_pred = y_pred.astype(np.float64, copy=False)
        yt = y_true - y_true.mean()
        yp = y_pred - y_pred.mean()
        ss_res = float(np.dot(yt - yp, yt - yp))
        ss_tot = float(np.dot(yt, yt) + 1e-12)
        return 1.0 - ss_res / ss_tot

    best_a, best_score = None, -np.inf

    # Precompute per-fold spectral facts once, then loop alphas cheaply
    per_fold_cache = []
    for tr, vl in splits:
        Xtr = Xk[tr]; ytr = y[tr]
        Xvl = Xk[vl]; yvl = y[vl]

        # Center target on train for stability (we don't add intercept)
        ytr_c = (ytr - ytr.mean()).astype(np.float64, copy=False)

        if not use_dual:
            # PRIMAL: eig of G = Xtr^T Xtr
            G = (Xtr.T @ Xtr).astype(np.float64, copy=False)
            b = (Xtr.T @ ytr_c).astype(np.float64, copy=False)
            # Symmetric → eigh
            evals, V = np.linalg.eigh(G)  # G = V diag(evals) V^T
            Vt_b = V.T @ b
            # For validation predictions, we’ll need Xvl @ w(a)
            per_fold_cache.append(("primal", (evals, V, Vt_b, Xvl)))
        else:
            # DUAL: eig of K = Xtr Xtr^T (size m x m)
            K = (Xtr @ Xtr.T).astype(np.float64, copy=False)
            evals, U = np.linalg.eigh(K)  # K = U diag(evals) U^T
            Ut_y = U.T @ ytr_c
            # For validation predictions, precompute K_vl_tr = Xvl Xtr^T
            K_vl_tr = (Xvl @ Xtr.T).astype(np.float64, copy=False)
            per_fold_cache.append(("dual", (evals, U, Ut_y, K_vl_tr)))

    # Loop over alphas; per fold do cheap diagonal divisions
    for a in alphas.astype(np.float32):
        a = float(max(a, eps))
        fold_scores = []

        for (mode, pack), (tr, vl) in zip(per_fold_cache, splits):
            yvl = y[vl]

            if mode == "primal":
                evals, V, Vt_b, Xvl = pack
                # w(a) = V * ((V^T b)/(evals + a))
                w_a = (V @ (Vt_b / (evals + a))).astype(np.float64, copy=False)
                yvl_pred = (Xvl @ w_a).astype(np.float64, copy=False)

            else:  # dual
                evals, U, Ut_y, K_vl_tr = pack
                # beta(a) = U * ((U^T y)/(evals + a))
                beta_a = (U @ (Ut_y / (evals + a))).astype(np.float64, copy=False)
                yvl_pred = (K_vl_tr @ beta_a).astype(np.float64, copy=False)

            fold_scores.append(r2_centered(yvl, yvl_pred))

        mean_r2 = float(np.mean(fold_scores)) if fold_scores else float('-inf')
        if mean_r2 > best_score:
            best_score, best_a = mean_r2, a

    # Final refit on ALL data at best_a
    a_fit = float(best_a if best_a is not None else 1.0)
    if not use_dual:
        XtX = (Xk.T @ Xk).astype(np.float64, copy=False)
        Xty = (Xk.T @ (y - y.mean())).astype(np.float64, copy=False)
        A = XtX + (a_fit + eps) * np.eye(Xk.shape[1], dtype=np.float64)
        try:
            Lc = np.linalg.cholesky(A)
            wk = np.linalg.solve(Lc.T, np.linalg.solve(Lc, Xty)).astype(np.float32, copy=False)
        except np.linalg.LinAlgError:
            wk = (np.linalg.pinv(A) @ Xty).astype(np.float32, copy=False)
    else:
        K_full = (Xk @ Xk.T).astype(np.float64, copy=False)
        y_c = (y - y.mean()).astype(np.float64, copy=False)
        A = K_full + (a_fit + eps) * np.eye(Xk.shape[0], dtype=np.float64)
        try:
            Lc = np.linalg.cholesky(A)
            beta = np.linalg.solve(Lc.T, np.linalg.solve(Lc, y_c))
        except np.linalg.LinAlgError:
            beta = np.linalg.pinv(A) @ y_c
        wk = (Xk.T @ beta).astype(np.float32, copy=False)

    w_full = np.zeros(L, np.float32); w_full[kept_idx] = wk
    return w_full, a_fit, float(best_score)


def ridge_layers_per_seed_vs_fmri_means(
    nod_rdms_per_layer: Dict[str, np.ndarray],   # arch -> (1, L, D)   <-- 1 seed now
    per_arch: Dict[str, Dict[str, Any]],         # arch -> {"layer_names": [...]}
    roi_results: Dict[str, Dict[str, Any]],      # from run_mri_weighting_by_roi
    alpha_min: float = 1e-3,
    alpha_max: float = 1e3,
    n_alphas: int = 13,
    cv_folds: int = 5,
    use_ridge: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """
    Single-SEED, batched-ROI ridge.

    For each arch:
      1) grab X = (D, L) from the only seed
      2) collect all ROIs whose vector dim == D
      3) run ONE alpha-search over ALL those ROIs (fast)
      4) for EACH ROI, call _ridge_fit_single_target with that alpha (keeps your logic)
      5) write back in the same format as before
    """
    alphas = np.logspace(np.log10(alpha_min), np.log10(alpha_max), n_alphas).astype(np.float32)
    out: Dict[str, Dict[str, Any]] = {}

    for arch, SLD in nod_rdms_per_layer.items():
        print(arch)
        if SLD.ndim != 3 or SLD.size == 0:
            continue

        S, L, D = SLD.shape
        # you said: "we don't have seeds anymore" -> we expect S == 1
        X_orig = SLD[0].T.astype(np.float32, copy=False)   # (D, L)
        N = X_orig.shape[0]

        layer_names = per_arch.get(arch, {}).get("layer_names", [f"layer_{i}" for i in range(L)])

        # collect ROIs that match this D
        roi_keys = []
        Y_simple_list = []
        Y_rew_list = []
        for roi, info in roi_results.items():
            y_s = np.asarray(info["simple_mean"], dtype=np.float32).ravel()
            y_r = np.asarray(info["reweighted_mean"], dtype=np.float32).ravel()
            if y_s.shape[0] != D:
                continue
            roi_keys.append(roi)
            Y_simple_list.append(y_s)
            Y_rew_list.append(y_r)

        if not roi_keys:
            out[arch] = {}
            continue

        # if user wants no ridge, fallback to uniform
        if not use_ridge:
            arch_res = {}
            Yhat = X_orig.T[None, :, :].mean(axis=1).astype(np.float32)  # but we don't really use this path
            for roi in roi_keys:
                arch_res[roi] = {
                    "layer_names": layer_names,
                    "seed_ids": [0],
                    "simple": {
                        "layer_weights": np.full((1, L), 1.0 / L, dtype=np.float32),
                        "predicted_vecs": Yhat,
                        "alpha": None,
                        "cv_r2_mean": float("nan"),
                    },
                    "reweighted": {
                        "layer_weights": np.full((1, L), 1.0 / L, dtype=np.float32),
                        "predicted_vecs": Yhat,
                        "alpha": None,
                        "cv_r2_mean": float("nan"),
                    },
                }
            out[arch] = arch_res
            continue

        # ----- BATCHED ALPHA SEARCH -----
        # stack targets: (R, D)
        Y_simple = np.stack(Y_simple_list, axis=0).astype(np.float32, copy=False)
        Y_rew    = np.stack(Y_rew_list,    axis=0).astype(np.float32, copy=False)
        R = Y_simple.shape[0]

        # apply the same column-centering idea here
        Xc = X_orig - X_orig.mean(axis=0, keepdims=True)

        k = min(cv_folds, max(2, N))
        kf = KFold(n_splits=k, shuffle=True, random_state=0)
        splits = list(kf.split(Xc))

        use_dual = (L > 1.5 * N)

        # precompute per fold
        per_fold_cache = []
        for tr, vl in splits:
            Xtr = Xc[tr]
            Xvl = Xc[vl]

            Ysim_tr = Y_simple[:, tr]
            Yrew_tr = Y_rew[:, tr]
            Ysim_tr_c = (Ysim_tr - Ysim_tr.mean(axis=1, keepdims=True)).astype(np.float64, copy=False)
            Yrew_tr_c = (Yrew_tr - Yrew_tr.mean(axis=1, keepdims=True)).astype(np.float64, copy=False)

            Ysim_vl = Y_simple[:, vl].astype(np.float64, copy=False)
            Yrew_vl = Y_rew[:, vl].astype(np.float64, copy=False)

            if not use_dual:
                G = (Xtr.T @ Xtr).astype(np.float64, copy=False)
                evals, V = np.linalg.eigh(G)
                B_sim = (Xtr.T @ Ysim_tr_c.T).astype(np.float64, copy=False)  # (L, R)
                B_rew = (Xtr.T @ Yrew_tr_c.T).astype(np.float64, copy=False)
                per_fold_cache.append((
                    "primal",
                    (evals, V, Xvl.astype(np.float64, copy=False),
                     Ysim_vl, Yrew_vl,
                     B_sim, B_rew)
                ))
            else:
                Ktr = (Xtr @ Xtr.T).astype(np.float64, copy=False)
                evals, U = np.linalg.eigh(Ktr)
                Ut_Ysim = (U.T @ Ysim_tr_c.T).astype(np.float64, copy=False)
                Ut_Yrew = (U.T @ Yrew_tr_c.T).astype(np.float64, copy=False)
                K_vl_tr = (Xvl @ Xtr.T).astype(np.float64, copy=False)
                per_fold_cache.append((
                    "dual",
                    (evals, U,
                     K_vl_tr,
                     Ysim_vl, Yrew_vl,
                     Ut_Ysim, Ut_Yrew)
                ))

        best_alpha_simple, best_scores_simple = None, None
        best_alpha_rew,    best_scores_rew    = None, None

        for a in alphas:
            a = float(max(a, 1e-8))
            scores_sim = np.zeros(R, dtype=np.float64)
            scores_rew = np.zeros(R, dtype=np.float64)
            counts     = np.zeros(R, dtype=np.int32)

            for (mode, pack), (tr, vl) in zip(per_fold_cache, splits):
                if mode == "primal":
                    evals, V, Xvl, Ysim_vl, Yrew_vl, B_sim, B_rew = pack
                    W_sim = V @ (B_sim / (evals[:, None] + a))
                    W_rew = V @ (B_rew / (evals[:, None] + a))
                    Ysim_pred = (Xvl @ W_sim).T   # (R, n_vl)
                    Yrew_pred = (Xvl @ W_rew).T
                else:
                    evals, U, K_vl_tr, Ysim_vl, Yrew_vl, Ut_Ysim, Ut_Yrew = pack
                    beta_sim = U @ (Ut_Ysim / (evals[:, None] + a))
                    beta_rew = U @ (Ut_Yrew / (evals[:, None] + a))
                    Ysim_pred = (K_vl_tr @ beta_sim).T
                    Yrew_pred = (K_vl_tr @ beta_rew).T

                # R^2 per ROI
                for r in range(R):
                    yv  = Ysim_vl[r]
                    yp  = Ysim_pred[r]
                    yvc = yv - yv.mean()
                    ypc = yp - yp.mean()
                    ss_res = float(np.dot(yvc - ypc, yvc - ypc))
                    ss_tot = float(np.dot(yvc, yvc) + 1e-12)
                    scores_sim[r] += (1.0 - ss_res / ss_tot)
                    counts[r]     += 1

                    yv2  = Yrew_vl[r]
                    yp2  = Yrew_pred[r]
                    yv2c = yv2 - yv2.mean()
                    yp2c = yp2 - yp2.mean()
                    ss_res2 = float(np.dot(yv2c - yp2c, yv2c - yp2c))
                    ss_tot2 = float(np.dot(yv2c, yv2c) + 1e-12)
                    scores_rew[r] += (1.0 - ss_res2 / ss_tot2)

            scores_sim = scores_sim / np.maximum(counts, 1)
            scores_rew = scores_rew / np.maximum(counts, 1)

            if (best_scores_simple is None) or (scores_sim.mean() > best_scores_simple.mean()):
                best_scores_simple = scores_sim
                best_alpha_simple  = a

            if (best_scores_rew is None) or (scores_rew.mean() > best_scores_rew.mean()):
                best_scores_rew = scores_rew
                best_alpha_rew  = a

        # ----- FINAL PER-ROI FIT USING YOUR HELPER -----
        arch_res: Dict[str, Any] = {}
        for idx, roi in enumerate(roi_keys):
            print("idx: ", idx, "  roi: ", roi)
            # SIMPLE
            w_s, _, _ = _ridge_fit_single_target(
                X_orig,
                Y_simple[idx],
                alphas=np.array([best_alpha_simple], dtype=np.float32),
                cv_folds=1,    # <--- final fit only, no extra CV
            )
            yhat_s = (X_orig @ w_s).astype(np.float32, copy=False)

            # REWEIGHTED
            w_r, _, _ = _ridge_fit_single_target(
                X_orig,
                Y_rew[idx],
                alphas=np.array([best_alpha_rew], dtype=np.float32),
                cv_folds=1,
            )
            yhat_r = (X_orig @ w_r).astype(np.float32, copy=False)

            arch_res[roi] = {
                "layer_names": layer_names,
                "seed_ids": [0],
                "simple": {
                    "layer_weights": w_s[None, :],
                    "predicted_vecs": yhat_s[None, :],
                    "alpha": float(best_alpha_simple),
                    "cv_r2_mean": float(best_scores_simple[idx]),
                },
                "reweighted": {
                    "layer_weights": w_r[None, :],
                    "predicted_vecs": yhat_r[None, :],
                    "alpha": float(best_alpha_rew),
                    "cv_r2_mean": float(best_scores_rew[idx]),
                },
            }

        out[arch] = arch_res

    return out



def subjects_vs_seedridge_means(
    ridge_part1: Dict[str, Dict[str, Any]],     # output of ridge_layers_per_seed_vs_fmri_means
    roi_results: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """
    For each (arch, ROI):
      - SeedRidgeSimpleMean:    simple mean across seeds of per-seed predictions vs fMRI simple mean
      - SeedRidgeWeightedMean:  weighted mean (via _iterative_reweight) across seeds of per-seed predictions vs fMRI reweighted mean
      - Compute per-subject Spearman distances to both means.
    Returns:
      out[arch][roi] = {
        "seedridge_simple_mean_vec": (D,),
        "seedridge_weighted_mean_vec": (D,),
        "d_subject_to_seedridge_simple_mean": (S_subj,),
        "d_subject_to_seedridge_weighted_mean": (S_subj,),
        "n_subjects": int,
        "vector_dim": int,
      }
    """
    out: Dict[str, Dict[str, Any]] = {}

    for arch, roi_map in ridge_part1.items():
        arch_res: Dict[str, Any] = {}
        for roi, rec in roi_map.items():
            if "simple" not in rec or "reweighted" not in rec:
                continue
            Yhat_simple = np.asarray(rec["simple"]["predicted_vecs"], dtype=np.float32)     # (S_seed, D)
            Yhat_rew    = np.asarray(rec["reweighted"]["predicted_vecs"], dtype=np.float32) # (S_seed, D)
            if Yhat_simple.ndim != 2 or Yhat_rew.ndim != 2:
                continue

            info = roi_results.get(roi, None)
            if info is None:
                continue
            H = np.asarray(info.get("embeddings"), dtype=np.float32)  # (S_subj, D)
            if H.ndim != 2 or H.shape[1] != Yhat_simple.shape[1]:
                continue

            # simple mean across seeds (simple-target predictions)
            seedridge_simple_mean_vec = Yhat_simple.mean(axis=0).astype(np.float32)

            # weighted mean across seeds (reweighted-target predictions)
            try:
                w_seed, seedridge_weighted_mean_vec, _, _ = _iterative_reweight(Yhat_rew.astype(np.float32))
                seedridge_weighted_mean_vec = seedridge_weighted_mean_vec.astype(np.float32)
            except Exception as e:
                print(f"[WARN] _iterative_reweight on seed predictions failed for ({arch}, {roi}): {e}")
                seedridge_weighted_mean_vec = Yhat_rew.mean(axis=0).astype(np.float32)

            # per-subject Spearman distances
            S_subj = H.shape[0]
            d_subj_to_simple = np.empty(S_subj, dtype=np.float32)
            d_subj_to_weight = np.empty(S_subj, dtype=np.float32)
            for si in range(S_subj):
                d_subj_to_simple[si] = float(spearman_distance(H[si], seedridge_simple_mean_vec))
                d_subj_to_weight[si] = float(spearman_distance(H[si], seedridge_weighted_mean_vec))

            arch_res[roi] = {
                "seedridge_simple_mean_vec": seedridge_simple_mean_vec,
                "seedridge_weighted_mean_vec": seedridge_weighted_mean_vec,
                "d_subject_to_seedridge_simple_mean": d_subj_to_simple,
                "d_subject_to_seedridge_weighted_mean": d_subj_to_weight,
                "n_subjects": int(S_subj),
                "vector_dim": int(H.shape[1]),
            }
        out[arch] = arch_res
    return out

def export_roi_alignment_from_ridge(
    ridge_part1: dict,                # from ridge_layers_per_seed_vs_fmri_means(...)
    roi_results: dict,                # from run_mri_weighting_by_roi(...)
    out_root: Path,
    args,
    use_simple_target: bool = True,   # True => use predictions trained on ROI simple mean; False => trained on reweighted mean
) -> dict:
    """
    For each ROI:
      - Build nod_rdms_like[arch] = (n_seeds, D) from ridge predictions for that ROI
      - Call compute_seed_roi_alignment(...) with that single-ROI target
      - Write summary + per-seed CSV (one pair per ROI)
    Returns: dict of CSV paths per ROI.
    """
    out_root.mkdir(parents=True, exist_ok=True)
    out = {}

    target_key = "simple" if use_simple_target else "reweighted"
    suffix = "ridgeSimple" if use_simple_target else "ridgeRew"

    for roi in roi_results.keys():
        # Assemble nod_rdms-like dict (per this ROI)
        nod_rdms_like: dict[str, np.ndarray] = {}
        for arch, roi_map in ridge_part1.items():
            rec = roi_map.get(roi)
            if not rec:
                continue
            tgt = rec.get(target_key)
            if not tgt or "predicted_vecs" not in tgt:
                continue
            Yhat = np.asarray(tgt["predicted_vecs"], dtype=np.float32)  # (n_seeds, D)
            if Yhat.ndim != 2 or Yhat.size == 0:
                continue
            nod_rdms_like[arch] = Yhat

        if not nod_rdms_like:
            continue

        # Single-ROI roi_results slice
        roi_results_one = {roi: roi_results[roi]}

        # Compute alignment
        roi_align = compute_seed_roi_alignment(nod_rdms_like, roi_results_one, use_seed_weights=True)

        # Output paths
        summary_csv = out_root / f"roi_alignments_summary_{roi}_{suffix}.csv"
        flat_csv    = out_root / f"roi_alignments_per_seed_{roi}_{suffix}.csv"

        # # Summary
        # with open(summary_csv, "w", newline="") as f:
        #     import csv
        #     w = csv.writer(f)
        #     w.writerow([
        #         "arch","roi","n_seeds_used","vector_dim",
        #         "mean_d_seed_vs_roi_simple","std_d_seed_vs_roi_simple",
        #         "mean_d_seed_vs_roi_reweighted","std_d_seed_vs_roi_reweighted",
        #         "best_seed_simple_index","best_seed_simple_distance",
        #         "best_seed_reweighted_index","best_seed_reweighted_distance",
        #         "d_seedmean_vs_roi_simple","d_weighted_seedmean_vs_roi_reweighted",
        #         "d_seedmean_vs_roi_reweighted","d_weighted_seedmean_vs_roi_simple",
        #     ])
        #     for arch, per_roi in roi_align.items():
        #         if arch == "_flat_records":
        #             continue
        #         if roi not in per_roi:
        #             continue
        #         m = per_roi[roi]
        #         w.writerow([
        #             arch, roi, int(m["n_seeds_used"]), int(m["vector_dim"]),
        #             f"{m['mean_d_seed_vs_roi_simple']:.6f}",
        #             f"{m['std_d_seed_vs_roi_simple']:.6f}",
        #             f"{m['mean_d_seed_vs_roi_reweighted']:.6f}",
        #             f"{m['std_d_seed_vs_roi_reweighted']:.6f}",
        #             int(m["best_seed_simple"]["seed_index"]),
        #             f"{m['best_seed_simple']['distance']:.6f}",
        #             int(m["best_seed_reweighted"]["seed_index"]),
        #             f"{m['best_seed_reweighted']['distance']:.6f}",
        #             f"{m['d_seedmean_vs_roi_simple']:.6f}",
        #             f"{m['d_weighted_seedmean_vs_roi_reweighted']:.6f}",
        #             f"{m['d_seedmean_vs_roi_reweighted']:.6f}",
        #             f"{m['d_weighted_seedmean_vs_roi_simple']:.6f}",
        #         ])

        # Flat
        flat = roi_align.get("_flat_records", [])
        with open(flat_csv, "w", newline="") as f:
            import csv
            w = csv.writer(f)
            w.writerow(["arch","roi","seed_index","d_vs_roi_simple","d_vs_roi_reweighted","vector_dim"])
            for r in flat:
                # Use the ROI stored in each record for robustness
                w.writerow([
                    r["arch"], r["roi"], int(r["seed_index"]),
                    f"{float(r['d_vs_roi_simple']):.6f}",
                    f"{float(r['d_vs_roi_reweighted']):.6f}",
                    int(r["vector_dim"]),
                ])

        print(f"[INF] Saved ridge-based ROI alignment CSVs for {roi} ({suffix})")
        out[roi] = {"flat_csv": str(flat_csv)}

    return out


def export_subjects_part2_csv(
    subjects_part2: dict,        # from subjects_vs_seedridge_means(...)
    out_root: Path,
    args,
) -> dict:
    """
    Writes two CSVs:
      - summary: one row per (arch, ROI) with mean/std of subject distances
      - per_subject: one row per (arch, ROI, subject) with the two distances
    """
    out_root.mkdir(parents=True, exist_ok=True)

    summary_csv  = out_root / f"subjects_vs_seedridgeMeans_SUMMARY.csv"
    per_subj_csv = out_root / f"subjects_vs_seedridgeMeans_PER_SUBJECT.csv"

    with open(summary_csv, "w", newline="") as fsum, open(per_subj_csv, "w", newline="") as fsub:
        import csv
        sw, pw = csv.writer(fsum), csv.writer(fsub)
        sw.writerow([
            "arch","roi","n_subjects","vector_dim",
            "mean_d_subject_to_seedridge_simple_mean","std_d_subject_to_seedridge_simple_mean",
            "mean_d_subject_to_seedridge_weighted_mean","std_d_subject_to_seedridge_weighted_mean",
        ])
        pw.writerow([
            "arch","roi","subject_index",
            "d_subject_to_seedridge_simple_mean",
            "d_subject_to_seedridge_weighted_mean",
        ])

        for arch, roi_map in subjects_part2.items():
            for roi, rec in roi_map.items():
                ds = np.asarray(rec["d_subject_to_seedridge_simple_mean"], dtype=np.float32)
                dw = np.asarray(rec["d_subject_to_seedridge_weighted_mean"], dtype=np.float32)
                sw.writerow([
                    arch, roi, int(rec["n_subjects"]), int(rec["vector_dim"]),
                    f"{float(ds.mean()):.6f}", f"{float(ds.std(ddof=0)):.6f}",
                    f"{float(dw.mean()):.6f}", f"{float(dw.std(ddof=0)):.6f}",
                ])
                for si, (dsi, dwi) in enumerate(zip(ds, dw)):
                    pw.writerow([arch, roi, int(si), f"{float(dsi):.6f}", f"{float(dwi):.6f}"])

    print(f"[INF] Saved subjects-vs-seedridgeMeans SUMMARY:     {summary_csv}")
    print(f"[INF] Saved subjects-vs-seedridgeMeans PER_SUBJECT: {per_subj_csv}")
    return {"summary_csv": str(summary_csv), "per_subject_csv": str(per_subj_csv)}


def ridge_four_combos_per_roi(
    ridge_part1: dict,              # from ridge_layers_per_seed_vs_fmri_means(...)
    roi_results: dict,              # from run_mri_weighting_by_roi(...)
) -> dict:
    """
    For each (arch, ROI), compute the ridge versions of the classic four-combo distances.

    Uses per-seed ridge predictions:
      - simple-target predictions:    Yhat_simple (S, D)
      - reweighted-target predictions: Yhat_rew    (S, D)

    Defines ridge seed means:
      - seedridge_simple_mean_vec   = mean_s Yhat_simple[s]
      - seedridge_weighted_mean_vec = iterative_reweight over rows of Yhat_rew

    Returns:
      out[arch][roi] = {
        # Seed-mean � ROI mean (ridge versions)
        "d_seedridgeSimple_vs_roiSimple": float,
        "d_seedridgeWeighted_vs_roiReweighted": float,
        "d_seedridgeSimple_vs_roiReweighted": float,
        "d_seedridgeWeighted_vs_roiSimple": float,

        # fMRI mean � seed-mean (ridge versions)
        "rho_fmriSimple_vs_seedridgeSimple": float,
        "rho_fmriSimple_vs_seedridgeWeighted": float,
        "rho_fmriReweighted_vs_seedridgeSimple": float,
        "rho_fmriReweighted_vs_seedridgeWeighted": float,

        # (optional) vectors used
        "seedridge_simple_mean_vec": (D,),
        "seedridge_weighted_mean_vec": (D,),
        "fmri_simple_mean_vec": (D,),
        "fmri_reweighted_mean_vec": (D,),
      }
    """
    out = {}
    for arch, roi_map in ridge_part1.items():
        arch_res = {}
        for roi, rec in roi_map.items():
            if "simple" not in rec or "reweighted" not in rec:
                continue
            Yhat_simple = np.asarray(rec["simple"]["predicted_vecs"], dtype=np.float32)     # (S, D)
            Yhat_rew    = np.asarray(rec["reweighted"]["predicted_vecs"], dtype=np.float32) # (S, D)
            if Yhat_simple.ndim != 2 or Yhat_rew.ndim != 2:
                continue

            # filter out non-finite rows (seed predictions with NaNs/Infs)
            finite_rows_simple = np.isfinite(Yhat_simple).all(axis=1)
            finite_rows_rew    = np.isfinite(Yhat_rew).all(axis=1)
            finite_rows        = finite_rows_simple & finite_rows_rew
            if not np.any(finite_rows):
                continue
            Yhat_simple = Yhat_simple[finite_rows]
            Yhat_rew    = Yhat_rew[finite_rows]

            roi_info = roi_results.get(roi, None)
            if roi_info is None:
                continue
            roi_simple = np.asarray(roi_info["simple_mean"], dtype=np.float32).ravel()
            roi_rew    = np.asarray(roi_info["reweighted_mean"], dtype=np.float32).ravel()
            D = Yhat_simple.shape[1]
            if roi_simple.shape[0] != D or roi_rew.shape[0] != D:
                continue

            # ridge seed means
            seedridge_simple_mean_vec = Yhat_simple.mean(axis=0).astype(np.float32)
            try:
                _, seedridge_weighted_mean_vec, _, _ = _iterative_reweight(Yhat_rew.astype(np.float32))
                seedridge_weighted_mean_vec = seedridge_weighted_mean_vec.astype(np.float32)
            except Exception:
                seedridge_weighted_mean_vec = Yhat_rew.mean(axis=0).astype(np.float32)

            # Seed-mean � ROI mean (ridge versions)
            d_ss_rs = float(spearman_distance(seedridge_simple_mean_vec,   roi_simple))
            d_sw_rr = float(spearman_distance(seedridge_weighted_mean_vec, roi_rew))
            d_ss_rr = float(spearman_distance(seedridge_simple_mean_vec,   roi_rew))
            d_sw_rs = float(spearman_distance(seedridge_weighted_mean_vec, roi_simple))

            # fMRI mean � seed-mean (ridge versions)
            rho_fs_ss = float(spearman_distance(roi_simple,   seedridge_simple_mean_vec))
            rho_fs_sw = float(spearman_distance(roi_simple,   seedridge_weighted_mean_vec))
            rho_fw_ss = float(spearman_distance(roi_rew,      seedridge_simple_mean_vec))
            rho_fw_sw = float(spearman_distance(roi_rew,      seedridge_weighted_mean_vec))

            arch_res[roi] = {
                "d_seedridgeSimple_vs_roiSimple": d_ss_rs,
                "d_seedridgeWeighted_vs_roiReweighted": d_sw_rr,
                "d_seedridgeSimple_vs_roiReweighted": d_ss_rr,
                "d_seedridgeWeighted_vs_roiSimple": d_sw_rs,

                "rho_fmriSimple_vs_seedridgeSimple": rho_fs_ss,
                "rho_fmriSimple_vs_seedridgeWeighted": rho_fs_sw,
                "rho_fmriReweighted_vs_seedridgeSimple": rho_fw_ss,
                "rho_fmriReweighted_vs_seedridgeWeighted": rho_fw_sw,

                "seedridge_simple_mean_vec": seedridge_simple_mean_vec,
                "seedridge_weighted_mean_vec": seedridge_weighted_mean_vec,
                "fmri_simple_mean_vec": roi_simple,
                "fmri_reweighted_mean_vec": roi_rew,
            }
        out[arch] = arch_res
    return out

def export_ridge_four(ridge_four, out_root, args):
    import csv, numpy as np
    out_csv = out_root / f"ridge_four_combos_{args.rdm_level}.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "arch","roi",
            "d_seedridgeSimple_vs_roiSimple",
            "d_seedridgeWeighted_vs_roiReweighted",
            "d_seedridgeSimple_vs_roiReweighted",
            "d_seedridgeWeighted_vs_roiSimple",
            "rho_fmriSimple_vs_seedridgeSimple",
            "rho_fmriSimple_vs_seedridgeWeighted",
            "rho_fmriReweighted_vs_seedridgeSimple",
            "rho_fmriReweighted_vs_seedridgeWeighted",
        ])
        for arch, roi_map in ridge_four.items():
            for roi, m in roi_map.items():
                w.writerow([
                    arch, roi,
                    f"{m['d_seedridgeSimple_vs_roiSimple']:.6f}",
                    f"{m['d_seedridgeWeighted_vs_roiReweighted']:.6f}",
                    f"{m['d_seedridgeSimple_vs_roiReweighted']:.6f}",
                    f"{m['d_seedridgeWeighted_vs_roiSimple']:.6f}",
                    f"{m['rho_fmriSimple_vs_seedridgeSimple']:.6f}",
                    f"{m['rho_fmriSimple_vs_seedridgeWeighted']:.6f}",
                    f"{m['rho_fmriReweighted_vs_seedridgeSimple']:.6f}",
                    f"{m['rho_fmriReweighted_vs_seedridgeWeighted']:.6f}",
                ])
    print(f"[INF] Saved ridge four-combos CSV: {out_csv}")
    return str(out_csv)