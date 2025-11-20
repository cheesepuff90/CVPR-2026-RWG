#!/usr/bin/env python3
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
import imagenet_weighted_embeddings as iwe

# ---------------- CLI ----------------

def parse_cli() -> Tuple[argparse.Namespace, Dict[str, str]]:
    p = argparse.ArgumentParser("Seeds unified runner")
    p.add_argument("--config", "-c", type=str, default="configs/default.yaml")

    # run mode
    p.add_argument("--mode", type=str, default="all",
                   choices=["inference", "pretrained"],
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
    pl_npz_name  = f"timm_pretrained_nod_rdms_6.npz"
    pl_meta_name = f"timm_pretrained_nod_rdms_6.json"
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
    
    _ = iwe.export_layerwise_alignment_pre_ridge(nod_rdms_per_layer, per_arch, roi_results, out_root, args)

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
    
    weights_csv = iwe.export_ridge_layer_weights_long(
        ridge_part1=ridge_part1,
        per_arch=per_arch,
        out_root=out_root,
    )
    outputs["ridge_layer_weights_csv"] = weights_csv

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

    return outputs


# ------------- main -------------

def main():
    args, overrides = parse_cli()
    cfg = load_config(args.config, overrides)
    prepare_folders(cfg)

    ckpt_root = Path(cfg["paths"]["checkpoint_root"])

    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if args.distributed and gpu_count > 0:
        args.world_size = gpu_count
    args.distributed = args.distributed and (args.world_size > 1)
    print(f"[MAIN] distributed={args.distributed} world_size={args.world_size} "
          f"gpu_count={gpu_count} CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")

    t0 = time.time()

    if args.mode == "inference":
        _ = run_inference_per_layer(cfg, args, ckpt_root)

    if args.mode == "pretrained":
        _ = run_pretrained_inference_all_layers(cfg, args, ckpt_root)

    print(f"[DONE] mode={args.mode} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
