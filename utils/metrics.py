from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image

import yaml
from transformers import CLIPModel, CLIPProcessor

# LPIPS
import lpips


@dataclass(frozen=True)
class EditCase:
    image_path: Path
    source_prompt: str
    target_prompt: str
    target_code: str
    target_index: int


def _first_present(d: Dict[str, Any], keys: List[str]) -> Optional[Any]:
    for k in keys:
        if k in d:
            return d[k]
    return None


def load_edits_yaml(edits_path: Path) -> List[EditCase]:
    obj = yaml.safe_load(edits_path.read_text(encoding="utf-8"))

    if isinstance(obj, list):
        items = obj
    elif isinstance(obj, dict):
        items = obj.get("edits", obj.get("items", obj.get("data", [])))
        if not isinstance(items, list):
            raise ValueError("Unsupported edits YAML format (expected list under edits/items/data).")
    else:
        raise ValueError("Unsupported edits YAML format (expected list or dict).")

    cases: List[EditCase] = []
    for it in items:
        if not isinstance(it, dict):
            continue

        img = _first_present(it, ["image_path", "img_path", "image", "path", "image_file", "input_img", "input_image"])
        if img is None:
            continue

        src_prompt = _first_present(it, ["source_prompt", "src_prompt", "prompt_src", "source"])
        if src_prompt is None:
            src_prompt = ""

        t_prompts = _first_present(it, ["target_prompts", "tgt_prompts", "prompts_target"])
        t_codes   = _first_present(it, ["target_codes", "tgt_codes", "codes_target"])
        targets   = _first_present(it, ["targets", "target_list", "edits", "target"])

        if isinstance(targets, list) and all(isinstance(x, dict) for x in targets):
            for idx, t in enumerate(targets):
                tp = _first_present(t, ["prompt", "target_prompt", "text"])
                tc = _first_present(t, ["code", "target_code", "name", "tag"])
                if tp is None or tc is None:
                    continue
                cases.append(EditCase(Path(img), str(src_prompt), str(tp), str(tc), idx))
        elif isinstance(t_prompts, list) and isinstance(t_codes, list) and len(t_prompts) == len(t_codes):
            for idx, (tp, tc) in enumerate(zip(t_prompts, t_codes)):
                cases.append(EditCase(Path(img), str(src_prompt), str(tp), str(tc), idx))
        elif isinstance(t_prompts, list):
            for j, tp in enumerate(t_prompts):
                cases.append(EditCase(Path(img), str(src_prompt), str(tp), f"t{j:02d}", j))
        else:
            tp = _first_present(it, ["target_prompt", "tgt_prompt"])
            tc = _first_present(it, ["target_code", "tgt_code"])
            if tp is not None and tc is not None:
                cases.append(EditCase(Path(img), str(src_prompt), str(tp), str(tc), 0))

    if not cases:
        raise ValueError(f"No cases parsed from: {edits_path}")
    return cases


def pil_rgb(p: Path) -> Image.Image:
    return Image.open(p).convert("RGB")


def find_edited_image(run_dir: Path, case: EditCase) -> Optional[Path]:

    if not run_dir.exists():
        return None

    exts = (".png", ".jpg", ".jpeg", ".webp")
    stem = case.image_path.stem.lower()
    
    tar_folder_name = f"tar_{case.target_index}"
    src_pattern = f"src_{stem}"
    
    for candidate in run_dir.rglob("*"):
        if candidate.suffix.lower() not in exts:
            continue
        
        parent_path = str(candidate.parent).lower()
        
        if tar_folder_name in parent_path and src_pattern in parent_path:
            if "output" in candidate.name.lower():
                return candidate
    
    candidates = [p for p in run_dir.rglob("*") if p.suffix.lower() in exts]
    
    matches = [p for p in candidates 
              if stem in str(p.parent).lower() or stem in p.name.lower()]
    
    if not matches:
        matches = [p for p in candidates if p.parent != run_dir]

    if not matches:
        return None

    output_matches = [p for p in matches if "output" in p.name.lower()]
    if output_matches:
        matches = output_matches
    
    matches.sort(key=lambda p: (len(str(p)), str(p)))
    return matches[0]


@torch.no_grad()
def clip_similarity(
    model: CLIPModel,
    processor: CLIPProcessor,
    device: torch.device,
    image: Image.Image,
    text: str
) -> float:
    inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    out = model(**inputs)
    img_emb = out.image_embeds
    txt_emb = out.text_embeds
    img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
    txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)
    sim = (img_emb * txt_emb).sum(dim=-1).item()
    return float(sim)


@torch.no_grad()
def lpips_distance(lpips_fn: lpips.LPIPS, device: torch.device, src: Image.Image, edt: Image.Image) -> float:
    # LPIPS expects torch tensors in [-1,1], shape [1,3,H,W]
    def to_t(im: Image.Image) -> torch.Tensor:
        arr = np.asarray(im).astype(np.float32) / 255.0  # [0,1]
        t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        t = (t * 2.0) - 1.0
        return t.to(device)

    d = lpips_fn(to_t(src), to_t(edt)).item()
    return float(d)


def load_manifest(manifest_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(manifest_csv)
    needed = {"run_id", "model", "method", "label", "order_idx", "output_dir"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=str, required=True)
    ap.add_argument("--edits_yaml", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--clip_model", type=str, default="openai/clip-vit-large-patch14")
    ap.add_argument("--lpips_net", type=str, default="alex", choices=["alex", "vgg", "squeeze"])
    ap.add_argument("--out_dir", type=str, default=None, help="Default: same folder as manifest (usually outputs_fig7)")
    args = ap.parse_args()

    manifest_csv = Path(args.manifest).resolve()
    edits_yaml = Path(args.edits_yaml).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else manifest_csv.parent.resolve()

    df_runs = load_manifest(manifest_csv)
    cases = load_edits_yaml(edits_yaml)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    # CLIP
    clip_model = CLIPModel.from_pretrained(args.clip_model).to(device).eval()
    clip_proc = CLIPProcessor.from_pretrained(args.clip_model)

    # LPIPS
    lpips_fn = lpips.LPIPS(net=args.lpips_net).to(device).eval()

    rows: List[Dict[str, Any]] = []
    for _, r in df_runs.iterrows():
        run_id = str(r["run_id"])
        model = str(r["model"])
        method = str(r["method"])
        label = str(r["label"])
        order_idx = int(r["order_idx"])
        run_dir = Path(r["output_dir"]).resolve()

        for c in cases:
            src_path = (Path(".") / c.image_path).resolve()
            edt_path = find_edited_image(run_dir, c)

            if not src_path.exists() or edt_path is None or not edt_path.exists():
                rows.append({
                    "run_id": run_id,
                    "model": model,
                    "method": method,
                    "label": label,
                    "order_idx": order_idx,
                    "src_image": str(c.image_path),
                    "target_code": c.target_code,
                    "target_prompt": c.target_prompt,
                    "edited_image": None if edt_path is None else str(edt_path),
                    "clip_t": np.nan,
                    "lpips": np.nan,
                    "status": "missing"
                })
                continue

            src_img = pil_rgb(src_path)
            edt_img = pil_rgb(edt_path)

            ct = clip_similarity(clip_model, clip_proc, device, edt_img, c.target_prompt)
            ld = lpips_distance(lpips_fn, device, src_img, edt_img)

            rows.append({
                "run_id": run_id,
                "model": model,
                "method": method,
                "label": label,
                "order_idx": order_idx,
                "src_image": str(c.image_path),
                "target_code": c.target_code,
                "target_prompt": c.target_prompt,
                "edited_image": str(edt_path),
                "clip_t": ct,
                "lpips": ld,
                "status": "ok"
            })

    df = pd.DataFrame(rows)

    per_case_csv = out_dir / "metrics_per_case.csv"
    agg_csv = out_dir / "metrics_agg.csv"

    df.to_csv(per_case_csv, index=False)

    df_ok = df[df["status"] == "ok"].copy()
    agg = (
        df_ok.groupby(["run_id", "model", "method", "label", "order_idx"], as_index=False)
        .agg(
            clip_t_mean=("clip_t", "mean"),
            lpips_mean=("lpips", "mean"),
            n_cases=("clip_t", "count"),
        )
    )
    agg.to_csv(agg_csv, index=False)

    print(f"Saved: {per_case_csv}")
    print(f"Saved: {agg_csv}")


if __name__ == "__main__":
    main()