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
from torchvision.models.feature_extraction import create_feature_extractor

import numpy as np
from pathlib import Path
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.pairwise import cosine_distances
from sklearn.model_selection import KFold
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd


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
        # best-effort; dont hard-fail on weird filesystems
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


def _compute_rdm_vec(X: np.ndarray, metric='correlation') -> tuple[np.ndarray, np.ndarray]:
    """
    X: [N, d] where N = items (images or classes). Returns (D, vec_upper).
    """
    D = squareform(pdist(X, metric=metric))
    v = D[np.triu_indices(D.shape[0], k=1)]
    return D, v


def _iterative_reweight(
    E: np.ndarray,
    eps_abs: float = 1e-6,
    eps_rel: float = 1e-6,
    max_iter: int = 100
):
    """
    Iterative reweighting using Tukey's biweight for robust averaging.
    
    Args:
        E: Array of shape [n_subjects, n_features]
        eps_abs: Absolute convergence tolerance
        eps_rel: Relative convergence tolerance
        max_iter: Maximum number of iterations
    
    Returns:
        w: Subject weights [n_subjects]
        E_mean: Robust mean [n_features]
        n_iter: Number of iterations until convergence
        E_norm: Z-score normalized data [n_subjects, n_features]
    """
    E = E.astype(np.float32)
    E_norm = E

    # # --- Z-score normalization per subject
    # mu = E.mean(axis=1, keepdims=True)
    # sd = E.std(axis=1, keepdims=True) + 1e-8
    # E_norm = (E - mu) / sd

    # # === Z-SCORE NORMALIZATION DIAGNOSTICS ===
    # if True:  # Set to False to disable
    #     print(f"\n    === Z-SCORE NORMALIZATION IMPACT ===")
    #     print(f"    BEFORE normalization:")
    #     print(f"      E[0]: mean={E[0].mean():.6f}, std={E[0].std():.6f}")
    #     print(f"      E[1]: mean={E[1].mean():.6f}, std={E[1].std():.6f}")
    #     print(f"      Raw correlation E[0] vs E[1]: {np.corrcoef(E[0], E[1])[0,1]:.6f}")
        
    #     print(f"    AFTER normalization:")
    #     print(f"      E_norm[0]: mean={E_norm[0].mean():.6f}, std={E_norm[0].std():.6f}")
    #     print(f"      E_norm[1]: mean={E_norm[1].mean():.6f}, std={E_norm[1].std():.6f}")
    #     print(f"      Normalized correlation E_norm[0] vs E_norm[1]: {np.corrcoef(E_norm[0], E_norm[1])[0,1]:.6f}")

    # Precompute constants
    E_center = E_norm - E_norm.mean(axis=1, keepdims=True)
    row_norms = np.linalg.norm(E_center, axis=1)

    # Initialize with median
    E_mean = np.median(E_norm, axis=0)
    w = np.ones(E_norm.shape[0], dtype=np.float32)
    w_prev = w.copy()

    for it in range(max_iter):
        E_old = E_mean.copy()
        mean_center = E_mean - E_mean.mean()

        # Compute correlation between each subject and current mean
        denom = row_norms * (np.linalg.norm(mean_center) + 1e-9)
        numer = E_center @ mean_center
        corr = numer / np.where(denom == 0, 1e-9, denom)
        corr = np.clip(corr, -0.999999, 0.999999)

         # DEBUG: First iteration diagnostics
        if it == 0:
            print(f"\n    === ITERATION 0 DIAGNOSTICS ===")
            print(f"    E_norm shape: {E_norm.shape}")
            print(f"    E_center shape: {E_center.shape}")
            print(f"    mean_center shape: {mean_center.shape}")
            print(f"    row_norms: min={row_norms.min():.6f}, max={row_norms.max():.6f}, mean={row_norms.mean():.6f}")
            print(f"    ||mean_center||: {np.linalg.norm(mean_center):.6f}")
            print(f"    numer (dot products): min={numer.min():.6f}, max={numer.max():.6f}, mean={numer.mean():.6f}")
            print(f"    denom: min={denom.min():.6f}, max={denom.max():.6f}, mean={denom.mean():.6f}")
            print(f"    corr (before clip): min={corr.min():.6f}, max={corr.max():.6f}, mean={corr.mean():.6f}")

        # Tukey's biweight weighting
        # r = 1.0 - corr  # Residual
        # s = np.median(np.abs(r - np.median(r))) + 1e-9  # Scale estimate (MAD)
        # t = r / (4.685 * s)  # Normalized residual (4.685 for 95% efficiency)
        # # Use much larger tuning constant for homogeneous data

        r = 1.0 - corr
        r_centered = r - np.median(r)               # now median(r_centered) ≈ 0
        s = np.median(np.abs(r_centered)) + 1e-9    # MAD of centered residuals
        # t = r_centered / (4.685 * s)               # or a slightly larger constant if you like
        t = r / (1.345 * s)
        w = 1.0 / np.maximum(1.0, np.abs(t))


        if it == 0:
            print(f"    r (1-corr): min={r.min():.6f}, max={r.max():.6f}, mean={r.mean():.6f}")
            print(f"    median(r): {np.median(r):.6f}")
        
        # w = (1 - t**2)**2
        # w[t >= 1] = 0.0  # Zero weight for extreme outliers

        # w = 1.0 / (1.0 + (t / 2.0)**2)  # Soft decay function

        if it == 0:
            print(f"    w (before clip): min={w.min():.6f}, max={w.max():.6f}, sum={w.sum():.6f}")

        w = np.clip(w, 1e-8, None)  # Ensure positive weights

        # Compute weighted average
        E_mean = np.average(E_norm, axis=0, weights=w)

        # Check convergence
        delta = np.linalg.norm(E_mean - E_old)
        rel = delta / (np.linalg.norm(E_old) + 1e-9)
        dw = np.linalg.norm(w - w_prev) / (np.linalg.norm(w_prev) + 1e-9)
        w_prev = w.copy()
        
        if (delta < eps_abs or rel < eps_rel) and dw < 1e-6:
            break

    print(f"    Converged in {it+1} iterations")
    print(f"    Subject weights: min={w.min():.3f}, max={w.max():.3f}, "
                  f"mean={w.mean():.3f}, std={w.std():.3f}")

    return w, E_mean, it + 1, E_norm


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
    r"|regnet[xy].*(?:0\.[0-9]|1\.[0-9]|2\.[0-9]|3\.[0-9]|4\.[0-9])"   # ~d4GF lines
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

    # things we DO want to keep even if theyre simple (readouts / stage boundaries)
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
        # skip known micro layers unless theyre meaningful readouts by name
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
                    # swallow edge cases; dont kill the run for one layer
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
                print(f"[PRETRAINED][{arch_key}] loading &")
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
                print(f"[PRETRAINED]building RDM for {len(kept_layers)} layers &")

                layer_rdms_mat, layer_rdms_vec = [], []
                for ln in kept_layers:
                    cc = class_counts[ln]          # shape [C]
                    mask = cc > 0
                    if mask.sum() < 2:
                        continue                   # need e2 classes for an RDM
                    Xc = (class_sums[ln][mask] / cc[mask, None]).astype(np.float32)
                    D, v = _compute_rdm_vec(Xc, metric="correlation")
                    layer_rdms_mat.append(D.astype(np.float32))
                    layer_rdms_vec.append(v.astype(np.float32))

                if not layer_rdms_mat:
                    print(f"[SKIP][{arch_key}] no valid layers produced class means.")
                    continue

                seed_rdms_mat = np.stack(layer_rdms_mat, axis=0)[None, ...]  # (1, L, N, N)
                seed_rdms_vec = np.stack(layer_rdms_vec, axis=0)[None, ...]  # (1, L, D)

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
            # Symmetric � eigh
            evals, V = np.linalg.eigh(G)  # G = V diag(evals) V^T
            Vt_b = V.T @ b
            # For validation predictions, well need Xvl @ w(a)
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



def export_ridge_layer_weights_long(
    ridge_part1: Dict[str, Dict[str, Any]],
    per_arch: Dict[str, Dict[str, Any]],
    out_root: Path,
    filename: str = "ridge_layer_weights_long.csv",
) -> str:
    """
    Writes one row per (arch, roi, mode, layer).
    Columns: arch, roi, mode, alpha, cv_r2_mean, layer, weight, rank_abs
    (weights are exactly what ridge_layers_per_seed_vs_fmri_means produced)
    """
    out_path = out_root / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arch","roi","mode","alpha","cv_r2_mean","layer","weight","rank_abs"])

        for arch, roi_dict in ridge_part1.items():
            # layer names from per_arch (fallback to indices)
            ln = per_arch.get(arch, {}).get("layer_names")
            for roi, bundle in roi_dict.items():
                for mode in ("simple","reweighted"):
                    rec = bundle[mode]
                    alpha = rec["alpha"]
                    cv    = rec["cv_r2_mean"]

                    # shape is (1, L) in your current pipeline (single seed)
                    wvec = np.asarray(rec["layer_weights"], dtype=np.float32).ravel()
                    # absolute-weight ranks (1 = largest |w|)
                    order = np.argsort(-np.abs(wvec))
                    ranks = np.empty_like(order)
                    ranks[order] = np.arange(1, wvec.size+1)

                    for i, ww in enumerate(wvec):
                        layer_name = (ln[i] if ln and i < len(ln) else f"layer_{i}")
                        w.writerow([arch, roi, mode, alpha, cv, layer_name, float(ww), int(ranks[i])])

    return str(out_path)


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

        flat_csv    = out_root / f"roi_alignments_per_seed_{roi}_{suffix}.csv"

        flat = roi_align.get("_flat_records", [])
        with open(flat_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "arch","roi",
                "d_vs_roi_simple","d_vs_roi_reweighted",
                "change","perc_change"
            ])
            for r in flat:
                dv_simple = float(r["d_vs_roi_simple"])
                dv_rew    = float(r["d_vs_roi_reweighted"])
                change    = dv_rew - dv_simple
                denom     = abs(dv_simple) if abs(dv_simple) > 1e-12 else 1e-12
                perc      = (change / denom) * 100.0

                w.writerow([
                    r["arch"], r["roi"],
                    f"{dv_simple:.4f}",
                    f"{dv_rew:.4f}",
                    f"{change:.4f}",
                    f"{perc:.4f}",
                ])

        print(f"[INF] Saved ridge-based ROI alignment CSVs for {roi} ({suffix})")
        out[roi] = {"flat_csv": str(flat_csv)}

    return out
    

def export_layerwise_alignment_pre_ridge(
    nod_rdms_per_layer: dict,   # arch -> (L, D) or (1, L, D), BEFORE ridge, single seed only
    per_arch: dict,             # arch -> {"layer_names": [...]}
    roi_results: dict,          # from run_mri_weighting_by_roi(...)
    out_root: Path,
    args,
    use_seed_weights: bool = True,  # no effect with 1 seed, kept for API symmetry
) -> dict:
    """
    Single-seed version.
    For each ROI and each (arch, layer), compute alignment (no ridge) and write:
      - layerwise_alignments_{roi}.csv  (per-layer rows with change/perc_change)
      - top_20_improved_layers_{roi}.csv
      - bottom_10_improved_layers_{roi}.csv

    CSV columns (flat):
      arch, roi, layer_index, layer_name, d_vs_roi_simple, d_vs_roi_reweighted, change, perc_change, vector_dim
    """
    out_root.mkdir(parents=True, exist_ok=True)
    out = {}

    for roi in roi_results.keys():
        flat_rows = []
        layer_changes = []  # list of tuples: (arch, roi, lidx, lname, change, perc_change)

        for arch, arr in nod_rdms_per_layer.items():
            E = np.asarray(arr, dtype=np.float32)
            if E.ndim == 3:
                # Expect (1, L, D)
                if E.shape[0] != 1:
                    # If you truly have >1 seeds, stop early to avoid silent mistakes
                    raise ValueError(f"Expected single seed for arch '{arch}', got shape {E.shape}")
                E = E[0]  # -> (L, D)
            if E.ndim != 2 or E.size == 0:
                continue

            L, D = E.shape
            layer_names = per_arch.get(arch, {}).get("layer_names", None)
            if not layer_names or len(layer_names) != L:
                layer_names = [f"layer_{i}" for i in range(L)]

            for l in range(L):
                # (1, D) to keep compatibility with compute_seed_roi_alignment
                seeds_layer = E[l, :][None, :]
                roi_align = compute_seed_roi_alignment(
                    {arch: seeds_layer}, {roi: roi_results[roi]}, use_seed_weights=use_seed_weights
                )
                flat = roi_align.get("_flat_records", [])
                if not flat:
                    continue

                r = flat[0]  # single seed -> single record
                dv_simple = float(r["d_vs_roi_simple"])
                dv_rew    = float(r["d_vs_roi_reweighted"])
                change    = dv_rew - dv_simple
                denom     = abs(dv_simple) if abs(dv_simple) > 1e-12 else 1e-12
                perc      = (change / denom) * 100.0
                vdim      = int(r.get("vector_dim", D))

                flat_rows.append([
                    arch, roi, l, layer_names[l],
                    f"{dv_simple:.4f}",
                    f"{dv_rew:.4f}",
                    f"{change:.4f}",
                    f"{perc:.4f}",
                    vdim
                ])
                layer_changes.append((arch, roi, l, layer_names[l], change, perc))

        # Write flat CSV
        flat_csv = out_root / f"layerwise_alignments_{roi}.csv"
        with open(flat_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "arch","roi","layer_index","layer_name",
                "d_vs_roi_simple","d_vs_roi_reweighted","change","perc_change","vector_dim"
            ])
            w.writerows(flat_rows)

        # Rank layers by improvement (change)
        layer_changes.sort(key=lambda x: x[4], reverse=True)  # x[4] = change
        top20    = layer_changes[:20]
        bottom10 = sorted(layer_changes, key=lambda x: x[4])[:10]

        # Write top/bottom CSVs
        top_csv = out_root / f"top_20_improved_layers_{roi}.csv"
        bot_csv = out_root / f"bottom_10_improved_layers_{roi}.csv"

        with open(top_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["arch","roi","layer_index","layer_name","change","perc_change"])
            for arch, roi_name, lidx, lname, chg, pchg in top20:
                w.writerow([arch, roi_name, lidx, lname, f"{chg:.4f}", f"{pchg:.4f}"])

        with open(bot_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["arch","roi","layer_index","layer_name","change","perc_change"])
            for arch, roi_name, lidx, lname, chg, pchg in bottom10:
                w.writerow([arch, roi_name, lidx, lname, f"{chg:.4f}", f"{pchg:.4f}"])

        print(f"[INF] Saved single-seed layerwise (pre-ridge) ROI alignment CSVs for {roi}")
        out[roi] = {
            "flat_csv": str(flat_csv),
            "top20_csv": str(top_csv),
            "bottom10_csv": str(bot_csv),
        }

    return out
