#!/usr/bin/env python3
"""
train.py

Unified entry for the Seeds project:
    --mode all        : train -> test -> inference
    --mode train      : train only (saves checkpoints)
    --mode test       : test only (uses saved checkpoints, saves NPZ embeddings/weights/RDMs)
    --mode inference  : inference only (loads NPZs from test step, aligns to human RDMs)

- Training uses DDP if --distributed and >1 GPU; test/inference run single-process on rank 0.
- Test saves per-arch / per-layer NPZs:
    {ckpt_root}/{arch}/{arch}_{layer}_seed_embeddings_weighted_{image|class}.npz
  Each NPZ has: embeddings [n_seeds, n_distances], weights [n_seeds].
- Inference loads those NPZs, aggregates across layers (concat embeddings, mean weights),
  runs fMRI weighting, and calls compare_outliers.
"""
from __future__ import annotations
import argparse, sys, os, time, json, glob
from pathlib import Path
from typing import Dict, Tuple, List
import re

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# local modules
from config_util import load_config, prepare_folders
import imagenet_weighted_embeddings as iwe


# ---------------- CLI ----------------

def parse_cli() -> Tuple[argparse.Namespace, Dict[str, str]]:
    p = argparse.ArgumentParser("Seeds unified runner")
    p.add_argument("--config", "-c", type=str, default="configs/default.yaml")

    # run mode
    p.add_argument("--mode", type=str, default="all",
                   choices=["all", "train", "test", "inference", "inference2", "pretrained"],
                   help="pipeline stage to run")

    # DDP
    p.add_argument("--distributed", action="store_true", help="enable DDP for training")
    p.add_argument("--world_size", type=int, default=1, help="total processes for DDP")
    p.add_argument("--local_rank", type=int, default=0, help="rank (set by launcher)")

    # test / inference opts
    p.add_argument("--rdm-level", type=str, default="class", choices=["image", "class"])
    p.add_argument("--rdm-metric", type=str, default="correlation")
    p.add_argument("--subsample-test-batches", type=int, default=None)
    p.add_argument("--nick-subject", type=str, default="sub-01")

    # inference opts (override fmri path)
    p.add_argument("--fmri-rdm-file", type=str, default=None,
                   help="override cfg.data.fmri_rdm_file")

    # dotted-key overrides: key=val
    p.add_argument("overrides", nargs="*",
                   help="Optional dotted-key overrides like 'training.epochs=30'")

    args = p.parse_args()

    ov = {}
    for item in args.overrides or []:
        if "=" not in item:
            sys.exit(f"Invalid override '{item}'. Use key=val.")
        k, v = item.split("=", 1)
        ov[k] = v
    return args, ov


# ------------- DDP helpers -------------

def init_dist_if_needed(local_rank: int, world_size: int) -> Tuple[bool, str]:
    """Initialize process group when world_size>1. Returns (is_dist, device_str)."""
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    is_dist = world_size > 1
    if is_dist:
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "12355")
        backend = "nccl" if (os.name != "nt" and torch.cuda.is_available()) else "gloo"
        if torch.cuda.is_available():
            torch.cuda.setDevice(local_rank) if hasattr(torch.cuda, "setDevice") else torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=backend, rank=local_rank, world_size=world_size)
        device = f"cuda:{local_rank}" if gpu_count > 0 else "cpu"
        print(
            f"[DDP:init] backend={backend} world_size={world_size} "
            f"rank={dist.get_rank()} local_rank={local_rank} device={device} "
            f"visible_gpus={torch.cuda.device_count()}"
        )
    else:
        device = "cuda:0" if gpu_count > 0 else "cpu"
        print(f"[DDP:init] single-process device={device} visible_gpus={torch.cuda.device_count()}")
    return is_dist, device


# ------------- loaders for saved NPZs -------------

def _layer_from_npz_name(npz_path: Path, arch: str, suffix: str) -> str:
    pat = re.compile(rf"^{re.escape(arch)}_(.+)_seed_embeddings_weighted_{re.escape(suffix)}\.npz$")
    m = pat.match(npz_path.name)
    return m.group(1) if m else npz_path.stem

def load_test_embeddings_from_npz(
    ckpt_root: Path,
    arch_list: List[str],
    rdm_level: str
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, Tuple[np.ndarray, List[str]]], Dict[str, np.ndarray]]:
    """
    Load the per-layer NPZs saved by iwe.test_pipeline and aggregate per-arch.

    Returns:
        seed_embeds[arch] -> E_concat [n_seeds, sum(n_distances per layer)]
        seed_w[arch]      -> W_mean   [n_seeds]
        per_layer_embeds[arch] -> (E_stack [n_layers, n_seeds, n_distances], layer_names)
        per_layer_w[arch] -> W_stack [n_layers, n_seeds]
    """
    suffix = "class" if rdm_level == "class" else "image"
    seed_embeds, seed_w = {}, {}
    per_layer_embeds, per_layer_w = {}, {}

    for arch in arch_list:
        arch_dir = ckpt_root / arch
        npzs = sorted(arch_dir.glob(f"{arch}_*_seed_embeddings_weighted_{suffix}.npz"))
        if not npzs:
            print(f"[INF] No test NPZs found for {arch} ({suffix}); skipping.")
            continue

        # consistent layer order by filename
        layer_names = [_layer_from_npz_name(p, arch, suffix) for p in npzs]
        Es, Ws = [], []
        for p in npzs:
            z = np.load(p)
            E = z["embeddings"] # [n_seeds, n_distances]
            W = z["weights"] # [n_seeds]
            Es.append(E)
            Ws.append(W)

        # concat across layers along distance dimension
        E_concat = np.concatenate(Es, axis=1)
        # stack for diagnostic/optional use
        try:
            E_stack = np.stack(Es, axis=0)  # [n_layers, n_seeds, n_distances]
        except ValueError:
            E_stack = None
        W_stack = np.stack(Ws, axis=0) # [n_layers, n_seeds]
        W_mean = W_stack.mean(axis=0) # [n_seeds]

        seed_embeds[arch] = E_concat
        seed_w[arch] = W_mean
        per_layer_embeds[arch] = (E_stack, layer_names)
        per_layer_w[arch] = W_stack

    return seed_embeds, seed_w, per_layer_embeds, per_layer_w


# ------------- stage runners -------------

def run_train(cfg, args, data_root: str | Path, ckpt_root: Path, local_rank: int) -> None:
    is_dist, device = init_dist_if_needed(local_rank, args.world_size)

    # single-process override device if provided
    if not is_dist and cfg["training"].get("device"):
        device = cfg["training"]["device"]

    dataset_type = cfg["data"].get("dataset", "imagenet_tiny")

    train_kwargs = {
        "arch_list": cfg["training"]["architectures"],
        "n_seeds": cfg["training"]["n_seeds"],
        "epochs": cfg["training"]["epochs"],
        "data_root": data_root,
        "dataset_type": dataset_type,
        "device": device,
        "ckpt_root": ckpt_root,
        "batch_size": cfg["training"]["batch_size"],
        "learning_rate": cfg["training"]["learning_rate"],
        "weight_decay": cfg["training"]["weight_decay"],
        "num_workers": cfg["misc"]["num_workers"],
        "subsample_val_batches": cfg["misc"]["subsample_val_batches"],
        "distributed": is_dist,
        "local_rank": local_rank,
        "world_size": args.world_size,
        "pca_dim": cfg["training"]["pca_dim"],
    }

    iwe.run_seed_pipeline(**train_kwargs)

    if is_dist:
        dist.barrier()
        dist.destroy_process_group()


def run_test(cfg, args, data_root: str | Path, ckpt_root: Path) -> dict:
    dataset_type = cfg["data"].get("dataset", "imagenet_tiny")
    device = cfg["training"].get("device") or ("cuda:0" if torch.cuda.is_available() else "cpu")

    # IMPORTANT for CLIP: ensure iwe.test_pipeline loads fine-tuned ckpts (patch in module).
    test_kwargs = {
        "arch_list": cfg["training"]["architectures"],
        "data_root": data_root,
        "dataset_type": dataset_type,
        "ckpt_root": ckpt_root,
        "device": device,
        "batch_size": cfg["training"]["batch_size"],
        "num_workers": cfg["misc"]["num_workers"],
        "subsample_test_batches": args.subsample_test_batches,
        "save_rdms": True,
        "rdm_level": args.rdm_level,
        "metric": args.rdm_metric,
        "n_seeds": cfg["training"]["n_seeds"]
    }
    return iwe.test_pipeline(**test_kwargs)

def _json_default(o):
    import numpy as np
    if isinstance(o, np.ndarray):     return o.tolist()
    if isinstance(o, (np.integer,)):  return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, (np.bool_,)):    return bool(o)
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")

def _to_jsonable(x):
    import numpy as np
    if isinstance(x, dict):  return {k:_to_jsonable(v) for k,v in x.items()}
    if isinstance(x, list):  return [_to_jsonable(v) for v in x]
    if isinstance(x, tuple): return [_to_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):     return x.tolist()
    if isinstance(x, (np.integer,)):  return int(x)
    if isinstance(x, (np.floating,)): return float(x)
    if isinstance(x, (np.bool_,)):    return bool(x)
    return x

def run_pretrained_inference_all_layers(cfg, args, ckpt_root: Path):
    device = cfg["training"].get("device") or ("cuda:0" if torch.cuda.is_available() else "cpu")
    arch_list = cfg["training"]["architectures"]  # list of torchvision model names you want
    nod_root = Path(cfg["data"]["nod_root"])
    out_root = Path(cfg["paths"].get("output_root", ckpt_root / "outputs_100"))
    out_root.mkdir(parents=True, exist_ok=True)

    nod_rdms_per_layer, per_arch_meta = iwe.run_nod_pipeline_pretrained(
        nod_root=nod_root,
        subsample_val_batches=None,
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["misc"]["num_workers"],
        device=device,
        rdm_level="class",
    )

    pl_npz = out_root / f"timm_pretrained_nod_rdms.npz"
    pl_meta = out_root / f"timm_pretrained_nod_rdms.json"

    # save arrays and meta
    np.savez(pl_npz, **nod_rdms_per_layer)
    with open(pl_meta, "w") as f:
        json.dump(per_arch_meta, f, indent=2, default=_json_default)

    print(f"[INF] Saved pretrained per-layer RDMs: {pl_npz}")
    print(f"[INF] Saved meta: {pl_meta}")
    return {"npz": str(pl_npz), "meta": str(pl_meta)}



# def run_inference(cfg, args, ckpt_root: Path) -> dict[str, Any]:
#     """Load test NPZs, compute human weighting, and compare."""
#     fmri_path = Path(args.fmri_rdm_file or cfg["data"]["fmri_rdm_file"])
#     if not fmri_path.exists():
#         raise FileNotFoundError(f"fMRI RDM path not found: {fmri_path}")

#     # Load seed embeddings produced by your test phase
#     arch_list = cfg["training"]["architectures"]
#     seed_embeds, seed_w, per_layer_embeds, per_layer_w = load_test_embeddings_from_npz(
#         ckpt_root, arch_list, rdm_level=args.rdm_level
#     )
#     if not seed_embeds:
#         raise RuntimeError("No NPZ embeddings loaded. Run `--mode test` first to generate them.")

#     # Run human weighting on fMRI (directory of sub-*/ .npy OR a single .npz, per your implementation)
#     simple_H, reweighted_H, human_w, E_norm = iwe.run_mri_weighting(fmri_path, pca_dim=cfg["training"]["pca_dim"])

#     # Compare & print inside function
#     results: dict[str, Any] = iwe.compare_outliers(
#         seed_embeds, seed_w, E_norm, human_w, simple_H
#     )

#     # Quick console summary (shapes)
#     for arch, arr in results.get("seed2human_dists", {}).items() if results.get("seed2human_dists") else []:
#         print(f"[INF] seed→human distances shape [{arch}]: {np.asarray(arr).shape}")
#     for arch, mat in results.get("seed_vs_human_matrix", {}).items() if results.get("seed_vs_human_matrix") else []:
#         print(f"[INF] seed↔human matrix shape [{arch}]: {np.asarray(mat).shape}")

#     # JSON summary (outliers only)
#     summary_path = ckpt_root / f"brain_alignment_summary_{args.rdm_level}.json"
#     try:
#         payload = {
#             "seed_outliers": {k: [int(x) for x in v] for k, v in results.get("seed_outliers", {}).items()},
#             "human_outliers": (
#                 results.get("human_outliers", None).tolist()
#                 if isinstance(results.get("human_outliers", None), np.ndarray) else results.get("human_outliers", None)
#             ),
#         }
#         with open(summary_path, "w") as f:
#             json.dump(payload, f, indent=2)
#         print(f"[INF] Saved summary: {summary_path}")
#     except Exception as e:
#         print(f"[WARN] Failed to save summary: {e}")
        
    
#     nod_root = cfg["data"].get("nod_root")
#     if nod_root:
#         nod_path = Path(nod_root)
        
#         if nod_path.exists():
#             device = cfg["training"].get("device") or ("cuda:0" if torch.cuda.is_available() else "cpu")
#             all_vecs, nod_rdms = iwe.run_nod_pipeline(
#                 ckpt_root=ckpt_root,
#                 nod_root=nod_path,
#                 arch_list=arch_list,
#                 subsample_val_batches=cfg["misc"]["subsample_val_batches"],
#                 batch_size=cfg["training"]["batch_size"],
#                 num_workers=cfg["misc"]["num_workers"],
#                 device=device
#             )
            
#             out_root = Path(cfg["paths"].get("output_root", ckpt_root / "outputs"))
#             out_root.mkdir(parents=True, exist_ok=True)
            
#             np.savez(out_root / f"nod_rdms_{args.rdm_level}.npz", **nod_rdms)
#             results["nod_rdms_shapes"] = {k: np.asarray(v).shape for k, v in nod_rdms.items()}
#             print("[INF] NOD RDM shapes:", results["nod_rdms_shapes"])
#         else:
#             print(f"[WARN] NOD root does not exist: {nod_path}")

#         rdms = {}
#         for arch in arch_list:
#             if arch in seed_embeds:
#                 res = iwe.run_seeds_reweighting(
#                     all_vecs,
#                     arch,
#                     fmri_simple_mean=simple_H,
#                     fmri_reweighted_mean=reweighted_H,
#                     do_plot=True
#                 )

#                 rdms[arch] = res
    

#     return results, rdms

def run_inference(cfg, args, ckpt_root: Path, mode) -> dict[str, Any]:
    """
    1) Load per-layer NPZs from test step and concatenate per-arch seed vectors.
    2) Compute fMRI weighting (subjects) -> simple + reweighted means.
    3) Compare seed outliers to human reference (correct arg passing).
    4) Run NOD inference: seeds 0..9 for all architectures.
    5) Seed reweighting: seeds 0..29 for resnet50, vit_b_16, alexnet (using NPZ concatenated vectors).
    """
    fmri_path = Path(args.fmri_rdm_file or cfg["data"]["fmri_rdm_file"])
    if not fmri_path.exists():
        raise FileNotFoundError(f"fMRI RDM path not found: {fmri_path}")
    
    arch_list = cfg["training"]["architectures"]
    arch_count = len(arch_list)

    # ---- 2) Human (fMRI) weighting ----
    simple_H, reweighted_H, human_w, H_norm = iwe.run_mri_weighting(
        fmri_path, pca_dim=cfg["training"]["pca_dim"]
    )

    print(f"[INF] fMRI simple mean shape: {simple_H.shape} reweighted mean shape: {reweighted_H.shape}")

    outputs: Dict[str, Any] = {}

    nod_rdms = None
    out_root = Path(cfg["paths"].get("output_root", ckpt_root / "outputs"))
    out_root.mkdir(parents=True, exist_ok=True)

    # Preferred filename pattern includes rdm_level, #arch, and arch_tag.
    nod_npz_name = f"nod_rdms_{args.rdm_level}_{arch_count}.npz"
    nod_npz_path = out_root / nod_npz_name

    def _try_load_nod_npz() -> Optional[dict]:
        # 1) exact match (new naming)
        if nod_npz_path.exists():
            print(f"[INF] Loading NOD NPZ: {nod_npz_path}")
            return dict(np.load(nod_npz_path, allow_pickle=True))
        return None

    # First, try to load a previously saved NOD inference result
    nod_rdms = _try_load_nod_npz()

    if nod_rdms is None:
        # ---- 4) NOD inference: seeds 0..9 for all architectures ----
        nod_root = cfg["data"].get("nod_root")
        if nod_root:
            nod_path = Path(nod_root)
            if nod_path.exists():
                device = cfg["training"].get("device") or ("cuda:0" if torch.cuda.is_available() else "cpu")
                all_vecs, nod_rdms = iwe.run_nod_pipeline(
                    ckpt_root=ckpt_root,
                    nod_root=nod_path,
                    arch_list=arch_list,                      # all 5 archs
                    subsample_val_batches=None,
                    batch_size=cfg["training"]["batch_size"],
                    num_workers=cfg["misc"]["num_workers"],
                    device=device,
                    rdm_level=args.rdm_level,
                    seed_ids=range(8)
                )

                out_root = Path(cfg["paths"].get("output_root", ckpt_root / "outputs"))
                out_root.mkdir(parents=True, exist_ok=True)
                np.savez(out_root / f"nod_rdms_{args.rdm_level}_{arch_count}.npz", **nod_rdms)
                outputs["nod_rdms_shapes"] = {k: np.asarray(v).shape for k, v in nod_rdms.items()}
                print("[INF] NOD RDM shapes:", outputs["nod_rdms_shapes"])
            else:
                print(f"[WARN] NOD root does not exist: {nod_path}")

    # 1! seed roi
    if mode in (1, "all"):
        try:
            roi_results = iwe.run_mri_weighting_by_roi(fmri_path)
            roi_align   = iwe.compute_seed_roi_alignment(nod_rdms, roi_results, use_seed_weights=True)

            # ------------------------------------------------------------------
            # 1) Summary CSV: one row per (arch, ROI)
            # ------------------------------------------------------------------
            summary_csv = out_root / f"roi_alignments_summary_{args.rdm_level}.csv"
            with open(summary_csv, "w", newline="") as f:
                import csv
                w = csv.writer(f)
                w.writerow([
                    "arch","roi","n_seeds_used","vector_dim",
                    "mean_d_seed_vs_roi_simple","std_d_seed_vs_roi_simple",
                    "mean_d_seed_vs_roi_reweighted","std_d_seed_vs_roi_reweighted",
                    "best_seed_simple_index","best_seed_simple_distance",
                    "best_seed_reweighted_index","best_seed_reweighted_distance",
                    # legacy mean-vs-mean summaries
                    "d_seedmean_vs_roi_simple","d_weighted_seedmean_vs_roi_reweighted",
                    "d_seedmean_vs_roi_reweighted","d_weighted_seedmean_vs_roi_simple",
                ])
                for arch, roimap in roi_align.items():
                    # skip the special top-level key with flat records
                    if arch == "_flat_records":
                        continue
                    for roi, m in roimap.items():
                        w.writerow([
                            arch, roi,
                            int(m["n_seeds_used"]),
                            int(m["vector_dim"]),
                            f"{m['mean_d_seed_vs_roi_simple']:.6f}",
                            f"{m['std_d_seed_vs_roi_simple']:.6f}",
                            f"{m['mean_d_seed_vs_roi_reweighted']:.6f}",
                            f"{m['std_d_seed_vs_roi_reweighted']:.6f}",
                            int(m["best_seed_simple"]["seed_index"]),
                            f"{m['best_seed_simple']['distance']:.6f}",
                            int(m["best_seed_reweighted"]["seed_index"]),
                            f"{m['best_seed_reweighted']['distance']:.6f}",
                            f"{m['d_seedmean_vs_roi_simple']:.6f}",
                            f"{m['d_weighted_seedmean_vs_roi_reweighted']:.6f}",
                        ])

            print(f"[INF] Saved ROI-wise alignment (summary) CSV: {summary_csv}")
            outputs["roi_alignment_summary_csv"] = str(summary_csv)

            # ------------------------------------------------------------------
            # 2) Flat per-seed CSV: one row per (arch, ROI, seed)
            # ------------------------------------------------------------------
            flat = roi_align.get("_flat_records", [])
            flat_csv = out_root / f"roi_alignments_per_seed_{args.rdm_level}.csv"
            with open(flat_csv, "w", newline="") as f:
                import csv
                w = csv.writer(f)
                w.writerow(["arch","roi","seed_index","d_vs_roi_simple","d_vs_roi_reweighted","vector_dim"])
                for r in flat:
                    w.writerow([
                        r["arch"], r["roi"], int(r["seed_index"]),
                        f"{r['d_vs_roi_simple']:.6f}",
                        f"{r['d_vs_roi_reweighted']:.6f}",
                        int(r["vector_dim"]),
                    ])

            print(f"[INF] Saved ROI-wise alignment (per-seed) CSV: {flat_csv}")
            outputs["roi_alignment_per_seed_csv"] = str(flat_csv)

        except Exception as e:
            print(f"[WARN] ROI-wise alignment skipped: {e}")

    # 2! fmri roi
    # After nod_rdms is loaded/computed
    if mode in (2, "all"):
        try:
            subj_vs_seed_outputs = iwe.run_fmri_subject_vs_seed_means(cfg, args, ckpt_root, nod_rdms)
            outputs["fmri_subject_vs_seed_means"] = subj_vs_seed_outputs
        except Exception as e:
            print(f"[WARN] Subject-vs-seedMeans analysis skipped: {e}")

    # 3! seed reweighting whole
    # ---- 5) Seed reweighting (30 seeds) for resnet50, vit_b_16, alexnet ----
    if mode in (3, "all"):
        target_arches = ["resnet50", "alexnet"]

        seed_reweighting: Dict[str, Any] = {}
        for arch in target_arches:
            E_all = np.asarray(nod_rdms[arch], dtype=np.float32)  # (n_seeds_total, d)
            if E_all.shape[0] < 30:
                print(f"[WARN] [{arch}] only {E_all.shape[0]} seeds available in NPZs; reweighting on all.")
                E_30 = E_all
            else:
                E_30 = E_all[:30, :]  # first 30 rows correspond to seeds 0..29 given numeric sort in test

            print(f"[TEST][{arch}] Running iterative reweighting on {E_30.shape[0]} seeds...")
            rews = iwe.run_seeds_reweighting(
                all_vecs=E_30,
                arch=arch,
                fmri_simple_mean=simple_H,
                fmri_reweighted_mean=reweighted_H,
                do_plot=True
            )
            seed_reweighting[arch] = {
                "n_seeds_used": int(E_30.shape[0]),
                "spearman_to_fmri_simple": float(rews["spearman_to_fmri_simple"]),
                "spearman_to_fmri_reweighted": float(rews["spearman_to_fmri_reweighted"]),
            }

        outputs["seed_reweighting"] = seed_reweighting

    # Optional: write a small JSON summary
    summary_path = ckpt_root / f"brain_alignment_summary_{args.rdm_level}.json"
    try:
        payload = {
            "seed_outliers": {k: [int(x) for x in v] for k, v in results.get("seed_outliers", {}).items()},
            "seed_reweighting": seed_reweighting,
        }
        with open(summary_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"[INF] Saved summary: {summary_path}")
    except Exception as e:
        print(f"[WARN] Failed to save summary: {e}")

    return outputs


def run_inference_without_ridge(cfg, args, ckpt_root: Path, mode) -> dict[str, Any]:
    """
    1) Load per-layer NPZs from test step and concatenate per-arch seed vectors.
    2) Compute fMRI weighting (subjects) -> simple + reweighted means.
    3) Compare seed outliers to human reference (correct arg passing).
    4) Run NOD inference: seeds 0..9 for all architectures.
    5) Seed reweighting: seeds 0..29 for resnet50, vit_b_16, alexnet (using NPZ concatenated vectors).
    """
    fmri_path = Path(args.fmri_rdm_file or cfg["data"]["fmri_rdm_file"])
    if not fmri_path.exists():
        raise FileNotFoundError(f"fMRI RDM path not found: {fmri_path}")
    
    arch_list = cfg["training"]["architectures"]
    arch_count = len(arch_list)

    # ---- 2) Human (fMRI) weighting ----
    simple_H, reweighted_H, human_w, H_norm = iwe.run_mri_weighting(
        fmri_path, pca_dim=cfg["training"]["pca_dim"]
    )

    print(f"[INF] fMRI simple mean shape: {simple_H.shape} reweighted mean shape: {reweighted_H.shape}")

    outputs: Dict[str, Any] = {}

    nod_rdms = None
    out_root = Path(cfg["paths"].get("output_root", ckpt_root / "outputs"))
    out_root.mkdir(parents=True, exist_ok=True)

    # Preferred filename pattern includes rdm_level, #arch, and arch_tag.
    nod_npz_name = f"timm_pretrained_nod_rdms_1.npz"
    nod_npz_path = out_root / nod_npz_name

    def _try_load_nod_npz() -> Optional[dict]:
        # 1) exact match (new naming)
        if nod_npz_path.exists():
            print(f"[INF] Loading NOD NPZ: {nod_npz_path}")
            return dict(np.load(nod_npz_path, allow_pickle=True))
        return None

    # First, try to load a previously saved NOD inference result
    nod_rdms = _try_load_nod_npz()

    try:
        roi_results = iwe.run_mri_weighting_by_roi(fmri_path)
        roi_align   = iwe.compute_seed_roi_alignment(nod_rdms, roi_results, use_seed_weights=True)
        # ------------------------------------------------------------------
        # 1) Summary CSV: one row per (arch, ROI)
        # ------------------------------------------------------------------
        summary_csv = out_root / f"roi_alignments_summary.csv"
        with open(summary_csv, "w", newline="") as f:
            import csv
            w = csv.writer(f)
            w.writerow([
                "arch","roi","n_seeds_used","vector_dim",
                "mean_d_seed_vs_roi_simple","std_d_seed_vs_roi_simple",
                "mean_d_seed_vs_roi_reweighted","std_d_seed_vs_roi_reweighted",
                "best_seed_simple_index","best_seed_simple_distance",
                "best_seed_reweighted_index","best_seed_reweighted_distance",
                # legacy mean-vs-mean summaries
                "d_seedmean_vs_roi_simple","d_weighted_seedmean_vs_roi_reweighted",
                "d_seedmean_vs_roi_reweighted","d_weighted_seedmean_vs_roi_simple",
            ])
            for arch, roimap in roi_align.items():
                print(arch)
                # skip the special top-level key with flat records
                if arch == "_flat_records":
                    continue
                for roi, m in roimap.items():
                    print(m)
                    w.writerow([
                        arch, roi,
                        int(m["n_seeds_used"]),
                        int(m["vector_dim"]),
                        f"{m['mean_d_seed_vs_roi_simple']:.6f}",
                        f"{m['std_d_seed_vs_roi_simple']:.6f}",
                        f"{m['mean_d_seed_vs_roi_reweighted']:.6f}",
                        f"{m['std_d_seed_vs_roi_reweighted']:.6f}",
                        int(m["best_seed_simple"]["seed_index"]),
                        f"{m['best_seed_simple']['distance']:.6f}",
                        int(m["best_seed_reweighted"]["seed_index"]),
                        f"{m['best_seed_reweighted']['distance']:.6f}",
                        f"{m['d_seedmean_vs_roi_simple']:.6f}",
                        f"{m['d_weighted_seedmean_vs_roi_reweighted']:.6f}",
                    ])

        print(f"[INF] Saved ROI-wise alignment (summary) CSV: {summary_csv}")
        outputs["roi_alignment_summary_csv"] = str(summary_csv)

        # ------------------------------------------------------------------
        # 2) Flat per-seed CSV: one row per (arch, ROI, seed)
        # ------------------------------------------------------------------
        flat = roi_align.get("_flat_records", [])
        flat_csv = out_root / f"roi_alignments_per_seed.csv"
        with open(flat_csv, "w", newline="") as f:
            import csv
            w = csv.writer(f)
            w.writerow(["arch","roi","seed_index","d_vs_roi_simple","d_vs_roi_reweighted","vector_dim"])
            for r in flat:
                w.writerow([
                    r["arch"], r["roi"], int(r["seed_index"]),
                    f"{r['d_vs_roi_simple']:.6f}",
                    f"{r['d_vs_roi_reweighted']:.6f}",
                    int(r["vector_dim"]),
                ])

        print(f"[INF] Saved ROI-wise alignment (per-seed) CSV: {flat_csv}")
        outputs["roi_alignment_per_seed_csv"] = str(flat_csv)

    except Exception as e:
        print(f"[WARN] ROI-wise alignment skipped: {e}")

    return outputs


def run_inference_per_layer(cfg, args, ckpt_root: Path) -> dict[str, Any]:
    """
    Ridge-aware pipeline using per-layer RDMs.

    Steps:
      1) fMRI ROI weighting -> simple + reweighted means (run_mri_weighting_by_roi)
      2) NOD per-layer inference -> nod_rdms_per_layer: {arch: (n_seeds, L, D)}, per_arch metadata
      3) Per ROI & per seed: ridge across layers vs fMRI simple & reweighted means
      4) Per ROI & subject: distances to seed-ridge simple/weighted means
      5) CSV exports (ridge-based) mirroring prior reporting
    """
    device = cfg["training"].get("device") or ("cuda:0" if torch.cuda.is_available() else "cpu")
    fmri_path = Path(args.fmri_rdm_file or cfg["data"]["fmri_rdm_file"])
    if not fmri_path.exists():
        raise FileNotFoundError(f"fMRI RDM path not found: {fmri_path}")

    arch_list  = cfg["training"]["architectures"]
    arch_count = len(arch_list)
    out_root   = Path(cfg["paths"].get("output_root", ckpt_root / "outputs"))
    out_root.mkdir(parents=True, exist_ok=True)

    outputs: Dict[str, Any] = {}

    # 1) fMRI ROI weighting
    roi_results = iwe.run_mri_weighting_by_roi(fmri_path)
    
    roi_out_dir = Path(out_root) / "roi_results"
    roi_out_dir.mkdir(parents=True, exist_ok=True)

    index = []
    for roi, data in roi_results.items():
        if "reweighted_mean" not in data:
            raise KeyError(f"ROI '{roi}' missing 'reweighted_mean'")
        vec = np.asarray(data["reweighted_mean"], dtype=np.float32)
        np.save(roi_out_dir / f"{roi}.npy", vec)
        index.append({
            "roi": roi,
            "vector_dim": int(vec.size),
            "dtype": str(vec.dtype),
            "path": str(roi_out_dir / f"{roi}.npy"),
        })

    with open(roi_out_dir / "_index.json", "w") as f:
        json.dump(sorted(index, key=lambda x: x["roi"]), f, indent=2)

    # -------------------------------------------------------
    # 2) Load-or-compute NOD per-layer RDMs (with caching)
    # -------------------------------------------------------
    # Preferred filename pattern includes rdm_level and #arch
    # pl_npz_name  = f"pretrained_nod_rdms_per_layer_class_{arch_count}.npz"
    # pl_meta_name = f"pretrained_nod_rdms_per_layer_meta_class_{arch_count}.json"
    pl_npz_name  = f"timm_pretrained_nod_rdms_2.npz"
    pl_meta_name = f"timm_pretrained_nod_rdms_2.json"
    pl_npz_path  = out_root / pl_npz_name
    pl_meta_path = out_root / pl_meta_name


    def _try_load_nod_per_layer() -> Optional[tuple[dict, dict]]:
        """Return (nod_rdms_per_layer, per_arch_meta) if both files exist; else None."""
        if pl_npz_path.exists() and pl_meta_path.exists():
            print(f"[INF] Loading per-layer NOD NPZ: {pl_npz_path}")
            nod_rdms_per_layer = dict(np.load(pl_npz_path, allow_pickle=True))
            with open(pl_meta_path, "r") as f:
                per_arch_meta = json.load(f)  # {arch: {"layer_names": [...], "rdm_level": "..."}}
            return nod_rdms_per_layer, per_arch_meta
        return None

    loaded = _try_load_nod_per_layer()
    if loaded is not None:
        nod_rdms_per_layer, per_arch_meta = loaded
        # adapt meta into the shape expected by ridge caller
        per_arch = {
            arch: {"layer_names": meta.get("layer_names", []),
                   "rdm_level": meta.get("rdm_level", args.rdm_level)}
            for arch, meta in per_arch_meta.items()
        }
    else:
        # Need to compute
        print(f"[INF] Running per-layer NOD")
        nod_root = cfg["data"].get("nod_root")
        if not nod_root:
            raise FileNotFoundError("cfg['data']['nod_root'] is missing.")
        nod_path = Path(nod_root)
        if not nod_path.exists():
            raise FileNotFoundError(f"NOD root does not exist: {nod_path}")

        # Compute per-layer outputs
        nod_rdms_per_layer, per_arch = iwe.run_nod_pipeline_per_layer(
            ckpt_root=ckpt_root,
            nod_root=nod_path,
            arch_list=arch_list,
            subsample_val_batches=None,
            batch_size=cfg["training"]["batch_size"],
            num_workers=cfg["misc"]["num_workers"],
            device=device,
            rdm_level=args.rdm_level,
            seed_ids=range(10),  # adjust your seed subset here
        )

        # Save NPZ (arrays) and META (layer names, etc.)
        np.savez(pl_npz_path, **nod_rdms_per_layer)
        per_arch_meta = {
            arch: {
                "layer_names": rec.get("layer_names", []),
                "rdm_level": rec.get("rdm_level", args.rdm_level),
            }
            for arch, rec in per_arch.items()
        }
        with open(pl_meta_path, "w") as f:
            json.dump(per_arch_meta, f, indent=2)

    outputs["nod_rdms_per_layer_npz"]    = str(pl_npz_path)
    outputs["nod_rdms_per_layer_meta"]   = str(pl_meta_path)
    outputs["nod_rdms_per_layer_shapes"] = {k: np.asarray(v).shape for k, v in nod_rdms_per_layer.items()}


    # If all arches have same D, pass one into ROI check
    any_arch = next(iter(nod_rdms_per_layer))
    D = nod_rdms_per_layer[any_arch].shape[2]

    # -------------------------------------------------------------
    # 3) Per-seed layer ridge vs fMRI simple & reweighted means
    # -------------------------------------------------------------
    ridge_part1 = iwe.ridge_layers_per_seed_vs_fmri_means(
        nod_rdms_per_layer=nod_rdms_per_layer,  # <- first item of the tuple
        per_arch=per_arch,                      # <- second item of the tuple (has layer_names)
        roi_results=roi_results,
        alpha_min=1e-3, alpha_max=1e3, n_alphas=13, cv_folds=5,
        use_ridge=True
    )
    outputs["ridge_part1_arches"] = list(ridge_part1.keys())

    # # 4) Per-subject distances to seed-ridge simple/weighted means
    # subjects_part2 = iwe.subjects_vs_seedridge_means(
    #     ridge_part1=ridge_part1,
    #     roi_results=roi_results,
    # )
    # outputs["subjects_part2_arches"] = list(subjects_part2.keys())

    # ridge_four = iwe.ridge_four_combos_per_roi(
    # ridge_part1=ridge_part1,
    # roi_results=roi_results )

    # ridge_four_csv = iwe.export_ridge_four(ridge_four, out_root, args)
    # outputs["ridge_four_combos_csv"] = ridge_four_csv

    # 5A) Export ROI alignment CSVs using ridge predictions (simple-target)
    ridge_csv_simple = iwe.export_roi_alignment_from_ridge(
        ridge_part1=ridge_part1,
        roi_results=roi_results,
        out_root=out_root,
        args=args,
        use_simple_target=True,
    )
    # 5B) Export ROI alignment CSVs using ridge predictions (reweighted-target)
    ridge_csv_rew = iwe.export_roi_alignment_from_ridge(
        ridge_part1=ridge_part1,
        roi_results=roi_results,
        out_root=out_root,
        args=args,
        use_simple_target=False,
    )
    outputs["roi_alignment_csvs_simple"]    = ridge_csv_simple
    outputs["roi_alignment_csvs_reweighted"] = ridge_csv_rew

    # # 5C) Export per-subject CSVs
    # subjects_csv = iwe.export_subjects_part2_csv(
    #     subjects_part2=subjects_part2,
    #     out_root=out_root,
    #     args=args,
    # )
    # outputs["subjects_part2_csv"] = subjects_csv

    return outputs

# ------------- worker wrapper for DDP training -------------

def _train_worker(rank: int, args, cfg, data_root: str | Path, ckpt_root: Path):
    print(f"[WORKER] starting rank={rank}")
    run_train(cfg, args, data_root, ckpt_root, local_rank=rank)


# ------------- main -------------

def main():
    args, overrides = parse_cli()
    cfg = load_config(args.config, overrides)
    prepare_folders(cfg)

    # dataset roots
    dataset_type = cfg["data"].get("dataset", "imagenet_tiny")
    data_root = {
        "cifar10":       cfg["data"].get("cifar10_root"),
        "cifar100":      cfg["data"].get("cifar100_root"),
        "imagenet_tiny": cfg["data"].get("imagenet_tiny_root"),
    }.get(dataset_type, cfg["data"].get("imagenet_full_root"))
    if data_root is None:
        raise RuntimeError(f"Missing data root for dataset '{dataset_type}' in config.")

    ckpt_root = Path(cfg["paths"]["checkpoint_root"])

    # DDP world size auto
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if args.distributed and gpu_count > 0:
        args.world_size = gpu_count
    args.distributed = args.distributed and (args.world_size > 1)
    print(f"[MAIN] distributed={args.distributed} world_size={args.world_size} "
          f"gpu_count={gpu_count} CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")

    t0 = time.time()

    if args.mode in ("all", "train"):
        if args.distributed:
            try:
                mp.set_start_method("spawn", force=True)
            except RuntimeError:
                pass
            mp.spawn(_train_worker, args=(args, cfg, data_root, ckpt_root),
                     nprocs=args.world_size, join=True)
        else:
            run_train(cfg, args, data_root, ckpt_root, local_rank=0)

    if args.mode in ("all", "test"):
        # single-process test
        _ = run_test(cfg, args, data_root, ckpt_root)

    if args.mode in ("all", "inference"):
        _ = run_inference_without_ridge(cfg, args, ckpt_root, 2)

    if args.mode in ("all", "inference2"):
        _ = run_inference_per_layer(cfg, args, ckpt_root)

    # if args.mode in ("all", "metrics"):
    #     from eval_metrics import evaluate_all
        
    #     dataset = cfg["data"]["dataset"]
    #     data_root = {
    #         "cifar10":  cfg["data"].get("cifar10_root"),
    #         "cifar100": cfg["data"].get("cifar100_root"),
    #         "imagenet": cfg["data"].get("imagenet_full_root")
    #     }.get("imagenet" if dataset.startswith("imagenet") else dataset)

    #     out_csv = Path(cfg["paths"]["metrics_csv"])
    #     evaluate_all(
    #         archs=cfg["training"]["architectures"],
    #         ckpt_root=Path(cfg["paths"]["checkpoint_root"]),
    #         data_root=Path(data_root),
    #         dataset=("imagenet" if dataset.startswith("imagenet") else dataset),
    #         device=cfg["training"].get("device", "cuda:0"),
    #         batch_size=cfg["training"]["batch_size"],
    #         num_workers=cfg["misc"]["num_workers"],
    #         select_by="loss",
    #         out_csv=out_csv,
    #     )

    if args.mode == "pretrained":
        _ = run_pretrained_inference_all_layers(cfg, args, ckpt_root)

    print(f"[DONE] mode={args.mode} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
