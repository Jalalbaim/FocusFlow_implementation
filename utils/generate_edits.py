import os
import argparse
import random
import numpy as np
from typing import Any, Dict, List, Tuple

import torch
from PIL import Image
from diffusers import StableDiffusion3Pipeline

from FlowEdit import flowedit



def sd3_sweep() -> List[Dict[str, Any]]:
    runs: List[Dict[str, Any]] = []

    # SDEdit
    nmax_list = [10, 15, 20, 25, 30, 35, 40]
    for i, nmax in enumerate(nmax_list):
        strength = 0.1 * (i + 2)  # label only
        runs.append({
            "model": "sd3",
            "method": "sdedit",
            "label": f"SDEdit_{strength:.1f}",
            "order_idx": i,
            "overrides": {
                "T": 50,
                "n_max": nmax,
                "cfg_tgt": 13.5,
            }
        })

    # FlowEdit (3 target CFG)
    for j, cfg_tgt in enumerate([13.5, 16.5, 19.5]):
        runs.append({
            "model": "sd3",
            "method": "flowedit",
            "label": f"Ours_cfg{cfg_tgt:g}",
            "order_idx": j,
            "overrides": {
                "T": 50,
                "n_max": 33,
                "cfg_src": 3.5,
                "cfg_tgt": cfg_tgt,
            }
        })

    return runs


@torch.no_grad()
def _encode_prompt_sd3(
    pipe,
    prompt: str,
    negative_prompt: str,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    try:
        out = pipe.encode_prompt(
            prompt=prompt,
            prompt_2=None,
            prompt_3=None,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
            negative_prompt=negative_prompt,
            negative_prompt_2=None,
            negative_prompt_3=None,
        )
    except TypeError:
        out = pipe.encode_prompt(
            prompt=prompt,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
            negative_prompt=negative_prompt,
        )

    if isinstance(out, (tuple, list)) and len(out) == 4:
        prompt_embeds, neg_prompt_embeds, pooled, neg_pooled = out
        return prompt_embeds, neg_prompt_embeds, pooled, neg_pooled

    raise RuntimeError("encode_prompt() did not return 4 tensors. Print 'out' and adjust unpacking.")


@torch.no_grad()
def _add_noise_sd3_fm(scheduler, x0: torch.Tensor, noise: torch.Tensor, i: int) -> torch.Tensor:

    if not hasattr(scheduler, "sigmas"):
        raise RuntimeError("Scheduler has no sigmas; can't add noise manually.")

    sigma = scheduler.sigmas[i]
    if not torch.is_tensor(sigma):
        sigma = torch.tensor(sigma, device=x0.device, dtype=x0.dtype)
    else:
        sigma = sigma.to(device=x0.device, dtype=x0.dtype)

    # sigma is scalar; broadcast over latent
    return x0 + sigma * noise



def _make_prompt_pack(pipe, prompt: str, negative_prompt: str, device: torch.device) -> Dict[str, torch.Tensor]:
    cond_emb, uncond_emb, cond_pool, uncond_pool = _encode_prompt_sd3(pipe, prompt, negative_prompt, device)
    return {
        "cond_emb": cond_emb,
        "uncond_emb": uncond_emb,
        "cond_pool": cond_pool,
        "uncond_pool": uncond_pool,
    }


@torch.no_grad()
def _transformer_forward_sd3(
    pipe,
    latents: torch.Tensor,
    t_in: torch.Tensor,
    emb_in: torch.Tensor,
    pool_in: torch.Tensor,
):

    # Most common
    try:
        return pipe.transformer(
            hidden_states=latents,
            timestep=t_in,
            encoder_hidden_states=emb_in,
            pooled_projections=pool_in,
            return_dict=True,
        )
    except TypeError:
        pass

    # Alternate pooled name
    try:
        return pipe.transformer(
            hidden_states=latents,
            timestep=t_in,
            encoder_hidden_states=emb_in,
            pooled_prompt_embeds=pool_in,
            return_dict=True,
        )
    except TypeError:
        pass

    # Some versions use "timesteps"
    try:
        return pipe.transformer(
            hidden_states=latents,
            timesteps=t_in,
            encoder_hidden_states=emb_in,
            pooled_projections=pool_in,
            return_dict=True,
        )
    except TypeError as e:
        raise RuntimeError(f"SD3 transformer call failed with your diffusers version: {e}")


@torch.no_grad()
def _cfg_velocity_sd3(
    pipe,
    latents: torch.Tensor,
    t: torch.Tensor,
    prompt_pack: Dict[str, torch.Tensor],
    guidance_scale: float,
) -> torch.Tensor:

    cond_emb = prompt_pack["cond_emb"]
    uncond_emb = prompt_pack["uncond_emb"]
    cond_pool = prompt_pack["cond_pool"]
    uncond_pool = prompt_pack["uncond_pool"]

    lat_in = torch.cat([latents, latents], dim=0)
    emb_in = torch.cat([uncond_emb, cond_emb], dim=0)
    pool_in = torch.cat([uncond_pool, cond_pool], dim=0)

    if not isinstance(t, torch.Tensor):
        t = torch.tensor(t, device=latents.device)
    if t.ndim == 0:
        t = t[None]
    t_in = t.repeat(2)

    out = _transformer_forward_sd3(pipe, lat_in, t_in, emb_in, pool_in)

    v = out.sample if hasattr(out, "sample") else out[0]
    v_uncond, v_cond = v.chunk(2, dim=0)
    return v_uncond + guidance_scale * (v_cond - v_uncond)


@torch.no_grad()
def SDEditSD3(
    pipe,
    scheduler,
    x0_src: torch.Tensor,
    src_prompt: str,
    tar_prompt: str,
    negative_prompt: str,
    T_steps: int,
    n_avg: int,
    cfg_src: float,
    cfg_tgt: float,
    n_min: int,
    n_max: int,
) -> torch.Tensor:
    device = x0_src.device
    scheduler.set_timesteps(T_steps, device=device)

    # start_i: bigger n_max => stronger => earlier (higher sigma)
    start_i = max(0, min(T_steps - 1, T_steps - int(n_max)))

    noise = torch.randn_like(x0_src)
    x_t = _add_noise_sd3_fm(scheduler, x0_src, noise, start_i)

    pack_tgt = _make_prompt_pack(pipe, tar_prompt, negative_prompt, device)

    lat = x_t
    for i in range(start_i, T_steps):
        t = scheduler.timesteps[i]
        with torch.autocast("cuda", enabled=torch.cuda.is_available()), torch.inference_mode():
            v = _cfg_velocity_sd3(pipe, lat, t, pack_tgt, cfg_tgt)

        step_out = scheduler.step(v, t, lat, return_dict=True)
        lat = step_out.prev_sample if hasattr(step_out, "prev_sample") else step_out[0]

    return lat

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_sd3_pipe(device_number: int) -> StableDiffusion3Pipeline:
    if "HF_TOKEN" not in os.environ:
        raise RuntimeError("Missing HF_TOKEN in environment. Do: export HF_TOKEN=...")

    pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers",
        torch_dtype=torch.float16,
        token=os.environ["HF_TOKEN"],
    )

    if torch.cuda.is_available():
        pipe.enable_model_cpu_offload()
        if hasattr(pipe, "vae"):
            if hasattr(pipe.vae, "enable_slicing"):
                pipe.vae.enable_slicing()
            if hasattr(pipe.vae, "enable_tiling"):
                pipe.vae.enable_tiling()
    else:
        pipe = pipe.to("cpu")

    return pipe


def encode_image_to_latent(pipe: StableDiffusion3Pipeline, image_path: str, device: torch.device) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    image = image.crop((0, 0, image.width - image.width % 16, image.height - image.height % 16))

    image_src = pipe.image_processor.preprocess(image).to(device).half()

    with torch.autocast("cuda", enabled=torch.cuda.is_available()), torch.inference_mode():
        x0_src_denorm = pipe.vae.encode(image_src).latent_dist.mode()

    x0_src = (x0_src_denorm - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
    return x0_src.to(device)


def decode_latent_to_pil(pipe: StableDiffusion3Pipeline, x0: torch.Tensor) -> Image.Image:
    x0_denorm = (x0 / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
    with torch.autocast("cuda", enabled=torch.cuda.is_available()), torch.inference_mode():
        img = pipe.vae.decode(x0_denorm, return_dict=False)[0]
    pil_list = pipe.image_processor.postprocess(img)
    return pil_list[0]


def run_one(
    pipe,
    scheduler,
    method: str,
    x0_src: torch.Tensor,
    src_prompt: str,
    tgt_prompt: str,
    negative_prompt: str,
    cfg: Dict[str, Any],
) -> torch.Tensor:
    T_steps = int(cfg["T"])
    n_avg = int(cfg["n_avg"])
    n_min = int(cfg["n_min"])
    n_max = int(cfg["n_max"])
    cfg_src = float(cfg["cfg_src"])
    cfg_tgt = float(cfg["cfg_tgt"])

    if method == "flowedit":
        return flowedit(
            pipe, scheduler, x0_src,
            src_prompt, tgt_prompt, negative_prompt,
            T_steps, n_avg, cfg_src, cfg_tgt, n_min, n_max
        )

    if method == "sdedit":
        return SDEditSD3(
            pipe, scheduler, x0_src,
            src_prompt, tgt_prompt, negative_prompt,
            T_steps, n_avg, cfg_src, cfg_tgt, n_min, n_max
        )

    raise ValueError(f"Unknown method: {method}")

class Args:
    device_number = 0
    input_img = "example_images/cat_ginger.png"
    src_prompt = "An orange and white cat sitting."
    tgt_prompt = "A tiger sitting."
    seed = 0
    out_dir = "outputs_sweep"
    n_avg = 1
    n_min = 0
    n_max = 33
    cfg_src = 3.5
    cfg_tgt = 13.5
    T = 50

args = Args()


def main():
    torch.set_grad_enabled(False)

    device = torch.device(f"cuda:{args.device_number}" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    set_seed(args.seed)

    pipe = load_sd3_pipe(args.device_number)
    scheduler = pipe.scheduler
    print("Model loaded")

    x0_src = encode_image_to_latent(pipe, args.input_img, device)

    runs = sd3_sweep()
    method_order = {"sdedit": 0, "ode_inv": 1, "irfds": 2, "flowedit": 3}
    runs = sorted(runs, key=lambda r: (method_order.get(r["method"], 99), r.get("order_idx", 0)))

    negative_prompt = ""

    base_cfg = {
        "T": args.T,
        "n_avg": args.n_avg,
        "n_min": args.n_min,
        "n_max": args.n_max,
        "cfg_src": args.cfg_src,
        "cfg_tgt": args.cfg_tgt,
    }

    src_name = os.path.splitext(os.path.basename(args.input_img))[0]

    for run in runs:
        cfg = dict(base_cfg)
        cfg.update(run.get("overrides", {}))

        label = run["label"]
        method = run["method"]

        print(f"[RUN] {label}  method={method}  cfg={cfg}")

        x0_out = run_one(
            pipe=pipe,
            scheduler=scheduler,
            method=method,
            x0_src=x0_src,
            src_prompt=args.src_prompt,
            tgt_prompt=args.tgt_prompt,
            negative_prompt=negative_prompt,
            cfg=cfg,
        )

        img_out = decode_latent_to_pil(pipe, x0_out)

        save_dir = os.path.join(args.out_dir, f"src_{src_name}", "tar_0", label)
        os.makedirs(save_dir, exist_ok=True)

        fname = (
            f"out_method_{method}_T{cfg['T']}_navg{cfg['n_avg']}_"
            f"cfgsrc{cfg.get('cfg_src','NA')}_cfgtgt{cfg.get('cfg_tgt','NA')}_"
            f"nmin{cfg.get('n_min','NA')}_nmax{cfg.get('n_max','NA')}_seed{args.seed}.png"
        )
        img_out.save(os.path.join(save_dir, fname))

        with open(os.path.join(save_dir, "prompts.txt"), "w", encoding="utf-8") as f:
            f.write(f"Source prompt: {args.src_prompt}\n")
            f.write(f"Target prompt: {args.tgt_prompt}\n")
            f.write(f"Seed: {args.seed}\n")
            f.write(f"Method: {method}\n")
            f.write(f"Label: {label}\n")
            f.write(f"Config: {cfg}\n")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("Done")

if __name__ == "__main__":
    main()