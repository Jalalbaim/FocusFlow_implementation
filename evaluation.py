import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from diffusers import FlowMatchEulerDiscreteScheduler, StableDiffusion3Pipeline
from transformers import CLIPModel, CLIPProcessor
import lpips

from FlowEdit import (
    decode_latents_to_image,
    encode_image_to_latents,
    flowedit,
    load_image,
)
from FocusFlow import focusflow
from utils.generate_edits import SDEditSD3
from utils.metrics import EditCase, clip_similarity, load_edits_yaml, lpips_distance


METHOD_FLOWEDIT = "flowedit"
METHOD_FOCUSFLOW = "focusflow"
METHOD_SDEDIT = "sdedit"
ALL_METHODS = (METHOD_FLOWEDIT, METHOD_FOCUSFLOW, METHOD_SDEDIT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate FlowEdit/FocusFlow/SDEdit on edits.yaml cases")

    # Input/output
    parser.add_argument("--edits_yaml", type=str, default="data/edits.yaml")
    parser.add_argument("--out_root", type=str, default="outputs_eval")
    parser.add_argument("--run_id", type=str, default=None, help="Default: seed<size> timestamp-free deterministic id")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--limit", type=int, default=None, help="Run only first N parsed edit cases")
    parser.add_argument("--only_methods", type=str, default="flowedit,focusflow,sdedit")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--save_masks", action="store_true", help="Save FocusFlow soft masks next to outputs")

    # Model setup
    parser.add_argument(
        "--model_id",
        type=str,
        default="stabilityai/stable-diffusion-3-medium-diffusers",
    )
    parser.add_argument("--drop_t5", action="store_true")

    # Common trajectory
    parser.add_argument("--T", type=int, default=60)

    # FlowEdit params
    parser.add_argument("--flowedit_nmax", type=int, default=33)
    parser.add_argument("--flowedit_nmin", type=int, default=0)
    parser.add_argument("--flowedit_navg", type=int, default=4)
    parser.add_argument("--flowedit_cfg_src", type=float, default=3.5)
    parser.add_argument("--flowedit_cfg_tar", type=float, default=16.5)

    # FocusFlow params
    parser.add_argument("--focus_nmax", type=int, default=60)
    parser.add_argument("--focus_nmin", type=int, default=15)
    parser.add_argument("--focus_navg", type=int, default=4)
    parser.add_argument("--focus_cfg_src", type=float, default=3.5)
    parser.add_argument("--focus_cfg_tar", type=float, default=21.5)
    parser.add_argument("--focus_negative_prompt", type=str, default="")

    # FocusFlow mask params
    parser.add_argument("--auto_mask", action="store_true", default=True)
    parser.add_argument("--mask_strength", type=float, default=0.5)
    parser.add_argument("--mask_n", type=int, default=10)
    parser.add_argument("--mask_guidance", type=float, default=6.0)
    parser.add_argument("--mask_blur_ks", type=int, default=5)
    parser.add_argument("--mask_dilate_ks", type=int, default=0)
    parser.add_argument("--mask_threshold", type=float, default=0.5)
    parser.add_argument("--mask_q_low", type=float, default=0.01)
    parser.add_argument("--mask_q_high", type=float, default=0.99)

    # SDEdit params
    parser.add_argument("--sdedit_nmax", type=int, default=25, help="Higher means stronger edit")
    parser.add_argument("--sdedit_nmin", type=int, default=0)
    parser.add_argument("--sdedit_navg", type=int, default=1)
    parser.add_argument("--sdedit_cfg_src", type=float, default=3.5)
    parser.add_argument("--sdedit_cfg_tar", type=float, default=13.5)
    parser.add_argument("--sdedit_negative_prompt", type=str, default="")

    # Metric params
    parser.add_argument("--clip_model", type=str, default="openai/clip-vit-large-patch14")
    parser.add_argument("--lpips_net", type=str, default="alex", choices=["alex", "vgg", "squeeze"])

    return parser.parse_args()


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA requested but unavailable; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_arg)


def _validate_hf_token() -> None:
    token = os.getenv("HF_TOKEN", "").strip()
    if not token:
        raise RuntimeError(
            "Missing HF_TOKEN environment variable. "
            "Set it before running, e.g. `export HF_TOKEN=...`"
        )


def _parse_methods(spec: str) -> List[str]:
    methods = [m.strip().lower() for m in spec.split(",") if m.strip()]
    bad = [m for m in methods if m not in ALL_METHODS]
    if bad:
        raise ValueError(f"Unknown method(s): {bad}. Allowed: {ALL_METHODS}")
    if not methods:
        raise ValueError("No methods selected.")
    return methods


def _build_run_dir(args: argparse.Namespace) -> Path:
    run_id = args.run_id or f"seed{args.seed}_size{args.size}"
    return Path(args.out_root) / run_id


def _method_output_path(run_dir: Path, method: str, case: EditCase) -> Path:
    stem = case.image_path.stem
    return run_dir / method / f"src_{stem}" / f"tar_{case.target_index}" / "output.png"


def _mask_output_path(output_path: Path) -> Path:
    return output_path.with_name("mask_soft.png")


def _meta_output_path(output_path: Path) -> Path:
    return output_path.with_name("meta.txt")


def _save_meta(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for k, v in payload.items():
            f.write(f"{k}: {v}\n")


def _save_soft_mask(mask_soft_hw: Optional[torch.Tensor], out_path: Path, out_size: int) -> None:
    if mask_soft_hw is None:
        return
    m = mask_soft_hw.float().cpu().numpy()
    m = np.clip(m, 0.0, 1.0)
    img = Image.fromarray((m * 255.0).round().astype(np.uint8), mode="L")
    img = img.resize((out_size, out_size), Image.BILINEAR)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def _load_pipe(args: argparse.Namespace, device: torch.device, dtype: torch.dtype) -> StableDiffusion3Pipeline:
    kwargs = {
        "torch_dtype": dtype if device.type == "cuda" else torch.float32,
        "variant": "fp16" if device.type == "cuda" else None,
        "use_safetensors": True,
    }
    if args.drop_t5:
        kwargs["text_encoder_3"] = None
        kwargs["tokenizer_3"] = None

    pipe = StableDiffusion3Pipeline.from_pretrained(args.model_id, **kwargs)
    pipe.enable_sequential_cpu_offload()
    pipe.enable_model_cpu_offload()
    pipe.enable_attention_slicing()
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    return pipe


def _run_flowedit(
    pipe: StableDiffusion3Pipeline,
    x_src: torch.Tensor,
    case: EditCase,
    args: argparse.Namespace,
) -> torch.Tensor:
    return flowedit(
        pipe=pipe,
        x_src=x_src,
        prompt_src=case.source_prompt,
        prompt_tar=case.target_prompt,
        T=args.T,
        nmax=args.flowedit_nmax,
        nmin=args.flowedit_nmin,
        navg=args.flowedit_navg,
        cfg_src=args.flowedit_cfg_src,
        cfg_tar=args.flowedit_cfg_tar,
        seed=args.seed,
    )


def _run_focusflow(
    pipe: StableDiffusion3Pipeline,
    x_src: torch.Tensor,
    case: EditCase,
    args: argparse.Namespace,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    out_latents, mask_soft_hw, _ = focusflow(
        pipe=pipe,
        x_src=x_src,
        prompt_src=case.source_prompt,
        prompt_tar=case.target_prompt,
        negative_prompt=args.focus_negative_prompt,
        T_steps=args.T,
        n_avg=args.focus_navg,
        src_guidance_scale=args.focus_cfg_src,
        tar_guidance_scale=args.focus_cfg_tar,
        n_min=args.focus_nmin,
        n_max=args.focus_nmax,
        auto_mask=args.auto_mask,
        mask_strength=args.mask_strength,
        mask_n=args.mask_n,
        mask_guidance=args.mask_guidance,
        mask_blur_ks=args.mask_blur_ks,
        mask_dilate_ks=args.mask_dilate_ks,
        mask_threshold=args.mask_threshold,
        mask_q_low=args.mask_q_low,
        mask_q_high=args.mask_q_high,
        mask_seed_base=args.seed,
    )
    return out_latents, mask_soft_hw


def _run_sdedit(
    pipe: StableDiffusion3Pipeline,
    x_src: torch.Tensor,
    case: EditCase,
    args: argparse.Namespace,
) -> torch.Tensor:
    scheduler = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config)
    return SDEditSD3(
        pipe=pipe,
        scheduler=scheduler,
        x0_src=x_src,
        src_prompt=case.source_prompt,
        tar_prompt=case.target_prompt,
        negative_prompt=args.sdedit_negative_prompt,
        T_steps=args.T,
        n_avg=args.sdedit_navg,
        cfg_src=args.sdedit_cfg_src,
        cfg_tgt=args.sdedit_cfg_tar,
        n_min=args.sdedit_nmin,
        n_max=args.sdedit_nmax,
    )


def _iter_cases(cases: Sequence[EditCase], limit: Optional[int]) -> Iterable[EditCase]:
    if limit is None:
        return cases
    return cases[: max(0, limit)]


def main() -> None:
    args = parse_args()
    _validate_hf_token()

    methods = _parse_methods(args.only_methods)
    device = _resolve_device(args.device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    run_dir = _build_run_dir(args)
    run_dir.mkdir(parents=True, exist_ok=True)

    cases = load_edits_yaml(Path(args.edits_yaml))
    cases = list(_iter_cases(cases, args.limit))
    if not cases:
        raise RuntimeError("No edit cases to run after applying --limit.")

    print(f"Loaded {len(cases)} cases from {args.edits_yaml}")
    print(f"Methods: {methods}")
    print(f"Run dir: {run_dir}")

    pipe = _load_pipe(args, device=device, dtype=dtype)
    exec_device = pipe._execution_device

    clip_model = CLIPModel.from_pretrained(args.clip_model).to(device).eval()
    clip_processor = CLIPProcessor.from_pretrained(args.clip_model)
    lpips_fn = lpips.LPIPS(net=args.lpips_net).to(device).eval()

    latent_cache: Dict[str, torch.Tensor] = {}
    source_img_cache: Dict[str, Image.Image] = {}

    rows: List[Dict[str, object]] = []

    for idx, case in enumerate(cases, start=1):
        src_key = str(case.image_path)
        src_path = Path(src_key)

        if src_key not in source_img_cache:
            source_img_cache[src_key] = load_image(str(src_path), size=args.size)
        src_img = source_img_cache[src_key]

        if src_key not in latent_cache:
            latent_cache[src_key] = encode_image_to_latents(pipe, src_img, device=exec_device, dtype=dtype)
        x_src = latent_cache[src_key]

        print(f"[{idx}/{len(cases)}] {case.image_path} :: target[{case.target_index}] {case.target_code}")

        for method in methods:
            output_path = _method_output_path(run_dir, method, case)
            meta_path = _meta_output_path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            if output_path.exists() and not args.overwrite:
                status = "skipped_exists"
                rows.append({
                    "method": method,
                    "src_image": str(case.image_path),
                    "target_index": case.target_index,
                    "target_code": case.target_code,
                    "clip_t": np.nan,
                    "lpips": np.nan,
                    "output_path": str(output_path),
                    "status": status,
                })
                continue

            try:
                mask_soft_hw = None
                print(f"[info] Executing method: {method}")
                if method == METHOD_FLOWEDIT:
                    print("[info] Running FlowEdit...")
                    out_latents = _run_flowedit(pipe, x_src, case, args)
                    print("[info] FlowEdit finished.")
                elif method == METHOD_FOCUSFLOW:
                    print("[info] Running FocusFlow...")
                    out_latents, mask_soft_hw = _run_focusflow(pipe, x_src, case, args)
                    print(f"[info] FocusFlow finished. mask_soft_hw={'present' if mask_soft_hw is not None else 'None'}")
                elif method == METHOD_SDEDIT:
                    print("[info] Running SDEdit...")
                    out_latents = _run_sdedit(pipe, x_src, case, args)
                    print("[info] SDEdit finished.")
                else:
                    raise ValueError(f"Unsupported method: {method}")

                if isinstance(out_latents, torch.Tensor):
                    try:
                        dev = out_latents.device
                        print(f"[info] out_latents shape={tuple(out_latents.shape)}, dtype={out_latents.dtype}, device={dev}")
                    except Exception:
                        print("[info] out_latents is a tensor (unable to show full details)")
                else:
                    print(f"[info] out_latents type: {type(out_latents)}")

                out_img = decode_latents_to_image(pipe, out_latents)
                out_img.save(output_path)

                if method == METHOD_FOCUSFLOW and args.save_masks and mask_soft_hw is not None:
                    _save_soft_mask(mask_soft_hw, _mask_output_path(output_path), out_size=args.size)

                clip_t = clip_similarity(clip_model, clip_processor, device, out_img, case.target_prompt)
                lpips_v = lpips_distance(lpips_fn, device, src_img, out_img)
                status = "ok"
            except Exception as exc:  # keep evaluation running across cases
                clip_t = np.nan
                lpips_v = np.nan
                status = f"error: {type(exc).__name__}"

            method_cfg = {
                "method": method,
                "seed": args.seed,
                "T": args.T,
                "src_prompt": case.source_prompt,
                "target_prompt": case.target_prompt,
                "target_code": case.target_code,
                "target_index": case.target_index,
            }
            if method == METHOD_FLOWEDIT:
                method_cfg.update({
                    "nmax": args.flowedit_nmax,
                    "nmin": args.flowedit_nmin,
                    "navg": args.flowedit_navg,
                    "cfg_src": args.flowedit_cfg_src,
                    "cfg_tar": args.flowedit_cfg_tar,
                })
            elif method == METHOD_FOCUSFLOW:
                method_cfg.update({
                    "nmax": args.focus_nmax,
                    "nmin": args.focus_nmin,
                    "navg": args.focus_navg,
                    "cfg_src": args.focus_cfg_src,
                    "cfg_tar": args.focus_cfg_tar,
                    "auto_mask": args.auto_mask,
                    "mask_strength": args.mask_strength,
                    "mask_n": args.mask_n,
                    "mask_guidance": args.mask_guidance,
                    "mask_blur_ks": args.mask_blur_ks,
                    "mask_dilate_ks": args.mask_dilate_ks,
                    "mask_threshold": args.mask_threshold,
                    "mask_q_low": args.mask_q_low,
                    "mask_q_high": args.mask_q_high,
                })
            elif method == METHOD_SDEDIT:
                method_cfg.update({
                    "nmax": args.sdedit_nmax,
                    "nmin": args.sdedit_nmin,
                    "navg": args.sdedit_navg,
                    "cfg_src": args.sdedit_cfg_src,
                    "cfg_tar": args.sdedit_cfg_tar,
                })
            _save_meta(meta_path, method_cfg)

            rows.append({
                "method": method,
                "src_image": str(case.image_path),
                "target_index": case.target_index,
                "target_code": case.target_code,
                "clip_t": clip_t,
                "lpips": lpips_v,
                "output_path": str(output_path),
                "status": status,
            })

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    per_case_csv = run_dir / "metrics_per_case.csv"
    agg_csv = run_dir / "metrics_agg.csv"

    fieldnames = ["method", "src_image", "target_index", "target_code", "clip_t", "lpips", "output_path", "status"]
    with per_case_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    agg_rows: List[Dict[str, object]] = []
    by_method: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for r in rows:
        by_method[str(r["method"])].append(r)

    for method, rlist in by_method.items():
        ok = [r for r in rlist if str(r["status"]) == "ok"]
        clip_vals = [float(r["clip_t"]) for r in ok if not np.isnan(r["clip_t"])]
        lpips_vals = [float(r["lpips"]) for r in ok if not np.isnan(r["lpips"])]
        agg_rows.append({
            "method": method,
            "n_cases": len(ok),
            "clip_t_mean": float(np.mean(clip_vals)) if clip_vals else np.nan,
            "lpips_mean": float(np.mean(lpips_vals)) if lpips_vals else np.nan,
        })

    with agg_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["method", "n_cases", "clip_t_mean", "lpips_mean"])
        w.writeheader()
        w.writerows(agg_rows)

    print("\nPer-method summary")
    print(f"{'method':<12} {'n_ok':>5} {'clip_t_mean':>12} {'lpips_mean':>12}")
    for r in sorted(agg_rows, key=lambda x: str(x["method"])):
        print(
            f"{r['method']:<12} {int(r['n_cases']):>5} "
            f"{float(r['clip_t_mean']):>12.4f} {float(r['lpips_mean']):>12.4f}"
        )

    print(f"\nSaved: {per_case_csv}")
    print(f"Saved: {agg_csv}")


if __name__ == "__main__":
    main()
