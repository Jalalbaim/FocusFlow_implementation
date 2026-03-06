import gc
import argparse
from typing import Optional, Union, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from diffusers import StableDiffusion3Pipeline, FlowMatchEulerDiscreteScheduler
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps

from dotenv import load_dotenv
import os
from huggingface_hub import login

# Charger les variables d'environnement
load_dotenv()

# Récupérer le token
huggingface_token = os.getenv('HF_TOKEN')

# Utiliser avec Hugging Face
try:
    login(token=huggingface_token)
    print("Logged in to Hugging Face with provided token.")
except Exception as e:
    print("Failed to log in to Hugging Face:", e)


def load_image(path: str, size: int = 512) -> Image.Image:
    img = Image.open(path).convert("RGB")
    img = img.resize((size, size), Image.BICUBIC)
    return img


@torch.no_grad()
def encode_image_to_latents(pipe: StableDiffusion3Pipeline, image: Image.Image, device, dtype) -> torch.Tensor:
    arr = np.array(image).astype(np.float32) / 255.0
    x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    x = (x * 2.0 - 1.0).to(device=device, dtype=dtype)

    latents = pipe.vae.encode(x).latent_dist.sample()
    latents = latents * pipe.vae.config.scaling_factor
    return latents


@torch.no_grad()
def decode_latents_to_image(pipe: StableDiffusion3Pipeline, latents: torch.Tensor) -> Image.Image:
    latents = latents / pipe.vae.config.scaling_factor
    img = pipe.vae.decode(latents).sample
    img = (img / 2 + 0.5).clamp(0, 1)
    img = img[0].permute(1, 2, 0).float().cpu().numpy()
    img = (img * 255).round().astype(np.uint8)
    return Image.fromarray(img)


def randn_like_with_seed(x: torch.Tensor, seed: int) -> torch.Tensor:
    # Deterministic noise even if x is on GPU by sampling on CPU.
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    noise_cpu = torch.randn(
        x.shape,
        generator=gen,
        device="cpu",
        dtype=torch.float32,
    )
    return noise_cpu.to(device=x.device, dtype=x.dtype)


def scale_noise(
    scheduler,
    sample: torch.FloatTensor,
    timestep: Union[float, torch.FloatTensor],
    noise: Optional[torch.FloatTensor] = None,
) -> torch.FloatTensor:
    """
    Match the FocusFlow notebook logic: compute sigma via scheduler's step_index and blend
    sample/noise accordingly.
    """
    scheduler._init_step_index(timestep)
    sigma = scheduler.sigmas[scheduler.step_index]
    return sigma * noise + (1.0 - sigma) * sample


def _encode_prompts_sd3(
    pipe: StableDiffusion3Pipeline,
    device,
    src_prompt: str,
    tar_prompt: str,
    negative_prompt: str,
    src_guidance_scale: float,
    tar_guidance_scale: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns concatenated prompt embeds and pooled embeds in the order:
    [src_uncond, src_text, tar_uncond, tar_text]
    """
    # Ensure SD3 pipeline treats this as classifier-free guidance.
    pipe._guidance_scale = float(max(src_guidance_scale, tar_guidance_scale, 1.0001))

    # Source embeds
    (src_prompt_embeds, src_negative_prompt_embeds,
     src_pooled_prompt_embeds, src_negative_pooled_prompt_embeds) = pipe.encode_prompt(
        prompt=src_prompt,
        prompt_2=None,
        prompt_3=None,
        negative_prompt=negative_prompt,
        do_classifier_free_guidance=True,
        device=device,
    )

    # Target embeds
    (tar_prompt_embeds, tar_negative_prompt_embeds,
     tar_pooled_prompt_embeds, tar_negative_pooled_prompt_embeds) = pipe.encode_prompt(
        prompt=tar_prompt,
        prompt_2=None,
        prompt_3=None,
        negative_prompt=negative_prompt,
        do_classifier_free_guidance=True,
        device=device,
    )

    src_tar_prompt_embeds = torch.cat(
        [src_negative_prompt_embeds, src_prompt_embeds,
         tar_negative_prompt_embeds, tar_prompt_embeds],
        dim=0
    )
    src_tar_pooled_prompt_embeds = torch.cat(
        [src_negative_pooled_prompt_embeds, src_pooled_prompt_embeds,
         tar_negative_pooled_prompt_embeds, tar_pooled_prompt_embeds],
        dim=0
    )

    return src_tar_prompt_embeds, src_tar_pooled_prompt_embeds


def calc_v_sd3(
    pipe: StableDiffusion3Pipeline,
    src_tar_latent_model_input: torch.Tensor,
    src_tar_prompt_embeds: torch.Tensor,
    src_tar_pooled_prompt_embeds: torch.Tensor,
    src_guidance_scale: float,
    tar_guidance_scale: float,
    t: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (v_src, v_tar) at timestep t for the given latent batch.
    Expects the latent batch size to be 4 * B in the order:
      [src_uncond, src_text, tar_uncond, tar_text]
    """
    # broadcast timestep
    timestep = t.expand(src_tar_latent_model_input.shape[0])

    with torch.no_grad():
        noise_pred_src_tar = pipe.transformer(
            hidden_states=src_tar_latent_model_input,
            timestep=timestep,
            encoder_hidden_states=src_tar_prompt_embeds,
            pooled_projections=src_tar_pooled_prompt_embeds,
            joint_attention_kwargs=None,
            return_dict=False,
        )[0]

        src_u, src_c, tar_u, tar_c = noise_pred_src_tar.chunk(4)
        v_src = src_u + float(src_guidance_scale) * (src_c - src_u)
        v_tar = tar_u + float(tar_guidance_scale) * (tar_c - tar_u)

    return v_src, v_tar


def _clip_extremes_percentile(x: torch.Tensor, q_low: float = 0.01, q_high: float = 0.99) -> torch.Tensor:
    flat = x.flatten()
    lo = torch.quantile(flat, float(q_low))
    hi = torch.quantile(flat, float(q_high))
    return x.clamp(lo, hi)


def _soften_mask(mask01_hw: torch.Tensor, blur_ks: int = 5, dilate_ks: int = 0) -> torch.Tensor:
    """
    Smooth edges to avoid seams when blending velocities.
    """
    M = mask01_hw[None, None, ...]  # (1,1,H,W)

    if dilate_ks and dilate_ks > 1:
        pad = dilate_ks // 2
        M = F.max_pool2d(M, kernel_size=dilate_ks, stride=1, padding=pad)

    if blur_ks and blur_ks > 1:
        pad = blur_ks // 2
        M = F.avg_pool2d(M, kernel_size=blur_ks, stride=1, padding=pad)

    return M.squeeze(0).squeeze(0).clamp(0.0, 1.0)


def _prep_mask_for_latents(mask_hw: torch.Tensor, x_src: torch.Tensor) -> torch.Tensor:
    """
    Convert mask (H,W) or (1,1,H,W) into (B,C,H,W) with x_src's shape.
    """
    device = x_src.device
    dtype = x_src.dtype
    B, C, H, W = x_src.shape

    M = mask_hw
    if not torch.is_tensor(M):
        M = torch.tensor(M)

    M = M.to(device=device, dtype=torch.float32)

    if M.ndim == 2:
        M = M[None, None, ...]  # (1,1,H,W)
    elif M.ndim == 3:
        M = M[None, ...]        # (1,*,H,W)

    if M.shape[-2:] != (H, W):
        M = F.interpolate(M, size=(H, W), mode="bilinear", align_corners=False)

    M = M.clamp(0.0, 1.0)

    if M.shape[0] == 1 and B > 1:
        M = M.expand(B, -1, -1, -1)
    if M.shape[1] == 1 and C > 1:
        M = M.expand(-1, C, -1, -1)

    return M.to(dtype=dtype)


@torch.no_grad()
def create_diffedit_mask_sd3(
    pipe: StableDiffusion3Pipeline,
    scheduler,
    x_src: torch.Tensor,
    src_prompt: str,
    tar_prompt: str,
    negative_prompt: str,
    T_steps: int = 50,
    strength: float = 0.5,
    n: int = 10,
    guidance_mask: float = 5.0,
    q_low: float = 0.01,
    q_high: float = 0.99,
    threshold: float = 0.5,
    blur_ks: int = 5,
    dilate_ks: int = 0,
    seed_base: int = 0,
):
    device = x_src.device
    timesteps, _ = retrieve_timesteps(scheduler, T_steps, device, timesteps=None)

    init_timestep = int(T_steps * float(strength))
    init_timestep = max(1, min(init_timestep, len(timesteps)))
    t_mask = timesteps[-init_timestep]

    src_tar_prompt_embeds, src_tar_pooled_prompt_embeds = _encode_prompts_sd3(
        pipe=pipe,
        device=device,
        src_prompt=src_prompt,
        tar_prompt=tar_prompt,
        negative_prompt=negative_prompt,
        src_guidance_scale=float(guidance_mask),
        tar_guidance_scale=float(guidance_mask),
    )

    B, C, H, W = x_src.shape
    acc = torch.zeros((H, W), device=device, dtype=torch.float32)

    for k in range(int(n)):
        fwd_noise = randn_like_with_seed(x_src, int(seed_base) + 100 * k)
        x_t = scale_noise(scheduler, x_src, t_mask, noise=fwd_noise)

        latent_in = torch.cat([x_t, x_t, x_t, x_t], dim=0)
        v_src, v_tar = calc_v_sd3(
            pipe,
            latent_in,
            src_tar_prompt_embeds,
            src_tar_pooled_prompt_embeds,
            float(guidance_mask),
            float(guidance_mask),
            t_mask,
        )

        diff_hw = (v_tar - v_src).abs().mean(dim=1).float()  # (B,H,W)
        acc += diff_hw.mean(dim=0)

    mask = acc / float(n)
    mask = _clip_extremes_percentile(mask, q_low=float(q_low), q_high=float(q_high))

    mmin, mmax = mask.min(), mask.max()
    mask01 = (mask - mmin) / (mmax - mmin + 1e-8)

    mask01 = _soften_mask(mask01, blur_ks=int(blur_ks), dilate_ks=int(dilate_ks))
    mask_bin = (mask01 >= float(threshold)).to(torch.uint8)

    return mask01.detach().cpu(), mask_bin.detach().cpu(), t_mask


@torch.no_grad()
def focusflow(
    pipe: StableDiffusion3Pipeline,
    x_src: torch.Tensor,
    prompt_src: str,
    prompt_tar: str,
    negative_prompt: str = "",
    T_steps: int = 50,
    n_avg: int = 1,
    src_guidance_scale: float = 3.5,
    tar_guidance_scale: float = 13.5,
    n_min: int = 0,
    n_max: int = 15,
    # mask options
    mask_soft_hw: Optional[torch.Tensor] = None,  # (H,W) in [0,1]
    auto_mask: bool = True,
    mask_strength: float = 0.5,
    mask_n: int = 10,
    mask_guidance: float = 5.0,
    mask_blur_ks: int = 5,
    mask_dilate_ks: int = 0,
    mask_threshold: float = 0.5,
    mask_q_low: float = 0.01,
    mask_q_high: float = 0.99,
    mask_seed_base: int = 0,
):
    """
    FocusFlow = masked FlowEdit for SD3:
      - If mask_soft_hw is None and auto_mask=True, builds a DiffEdit-style mask from velocity differences.
      - ODE-like phase (T_steps-n_max .. T_steps-n_min): integrates masked delta velocity.
      - Sampling-like phase (last n_min steps): blends v_tar inside mask and v_src outside mask.
    """
    device = x_src.device

    # Scheduler: use the same class as SD3 FlowEdit implementations.
    scheduler = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config)
    timesteps, T_steps = retrieve_timesteps(scheduler, T_steps, device, timesteps=None)
    pipe._num_timesteps = len(timesteps)

    # Ensure CFG is active for the batched prompt embedding logic.
    pipe._guidance_scale = float(max(src_guidance_scale, tar_guidance_scale, mask_guidance, 1.0001))

    src_tar_prompt_embeds, src_tar_pooled_prompt_embeds = _encode_prompts_sd3(
        pipe=pipe,
        device=device,
        src_prompt=prompt_src,
        tar_prompt=prompt_tar,
        negative_prompt=negative_prompt,
        src_guidance_scale=float(src_guidance_scale),
        tar_guidance_scale=float(tar_guidance_scale),
    )

    print("Mask generation...")
    if mask_soft_hw is None and auto_mask:
        mask_soft_hw, mask_bin_hw, _ = create_diffedit_mask_sd3(
            pipe=pipe,
            scheduler=scheduler,
            x_src=x_src,
            src_prompt=prompt_src,
            tar_prompt=prompt_tar,
            negative_prompt=negative_prompt,
            T_steps=T_steps,
            strength=float(mask_strength),
            n=int(mask_n),
            guidance_mask=float(mask_guidance),
            q_low=float(mask_q_low),
            q_high=float(mask_q_high),
            threshold=float(mask_threshold),
            blur_ks=int(mask_blur_ks),
            dilate_ks=int(mask_dilate_ks),
            seed_base=int(mask_seed_base),
        )
    elif mask_soft_hw is None:
        mask_bin_hw = None
    else:
        if not torch.is_tensor(mask_soft_hw):
            mask_soft_hw = torch.tensor(mask_soft_hw)
        mask_bin_hw = (mask_soft_hw >= float(mask_threshold)).to(torch.uint8)

    M = None
    if mask_soft_hw is not None:
        M = _prep_mask_for_latents(mask_soft_hw.to(device=device), x_src)

    print("FocusFlow (masked FlowEdit)")
    zt_edit = x_src.clone()

    # Main integration loop (faithful to the notebook logic).
    for i, t in enumerate(timesteps):
        if T_steps - i > int(n_max):
            continue

        t_i = t / 1000
        if i + 1 < len(timesteps):
            t_im1 = timesteps[i + 1] / 1000
        else:
            t_im1 = torch.zeros_like(t_i).to(t_i.device)

        if T_steps - i > int(n_min):
            # ODE phase: masked delta velocity
            V_delta_avg = torch.zeros_like(x_src)

            for _ in range(int(n_avg)):
                fwd_noise = torch.randn_like(x_src)

                zt_src = (1 - t_i) * x_src + t_i * fwd_noise
                zt_tar = zt_edit + zt_src - x_src

                latent_in = torch.cat([zt_src, zt_src, zt_tar, zt_tar], dim=0)
                Vt_src, Vt_tar = calc_v_sd3(
                    pipe,
                    latent_in,
                    src_tar_prompt_embeds,
                    src_tar_pooled_prompt_embeds,
                    float(src_guidance_scale),
                    float(tar_guidance_scale),
                    t,
                )

                delta = (Vt_tar - Vt_src)
                if M is not None:
                    delta = M * delta

                V_delta_avg += delta / float(n_avg)

            zt_edit = zt_edit.to(torch.float32)
            zt_edit = zt_edit + (t_im1 - t_i) * V_delta_avg.to(torch.float32)
            zt_edit = zt_edit.to(dtype=x_src.dtype)

        else:
            # sampling-like phase: blend velocities
            if i == T_steps - int(n_min):
                fwd_noise = torch.randn_like(x_src)
                xt_src = scale_noise(scheduler, x_src, t, noise=fwd_noise)
                xt_tar = zt_edit + xt_src - x_src

            latent_in = torch.cat([xt_tar, xt_tar, xt_tar, xt_tar], dim=0)
            Vt_src, Vt_tar = calc_v_sd3(
                pipe,
                latent_in,
                src_tar_prompt_embeds,
                src_tar_pooled_prompt_embeds,
                float(src_guidance_scale),
                float(tar_guidance_scale),
                t,
            )

            if M is None:
                V_final = Vt_tar
            else:
                V_final = M * Vt_tar + (1 - M) * Vt_src

            xt_tar = xt_tar.to(torch.float32)
            xt_tar = (xt_tar + (t_im1 - t_i) * V_final.to(torch.float32)).to(dtype=x_src.dtype)
            #xt_tar = (xt_tar + (t_im1 - t_i) * (V_final - Vt_src).to(torch.float32)).to(dtype=x_src.dtype)

    out_latents = zt_edit if int(n_min) == 0 else xt_tar
    return out_latents, mask_soft_hw, mask_bin_hw


def _save_mask_png(mask_hw: torch.Tensor, out_path: str, out_size: int = 512):
    """
    Save a soft mask (H,W) in [0,1] as a grayscale PNG, upsampled to out_size.
    """
    if mask_hw is None:
        return
    m = mask_hw.float().cpu()[None, None]  # (1,1,h,w)
    m = F.interpolate(m, size=(out_size, out_size), mode="bilinear", align_corners=False)[0, 0]
    m = m.clamp(0, 1).numpy()
    img = (m * 255).round().astype(np.uint8)
    Image.fromarray(img, mode="L").save(out_path)


def main():
    parser = argparse.ArgumentParser(description="FocusFlow (masked FlowEdit) for SD3")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image")
    parser.add_argument("--prompt_src", type=str, required=True, help="Source prompt describing the input image")
    parser.add_argument("--prompt_tar", type=str, required=True, help="Target prompt for the edited image")
    parser.add_argument("--negative_prompt", type=str, default="", help="Negative prompt (default: '')")

    # Main edit params (mirrors my_FlowEdit.py style)
    parser.add_argument("--T", type=int, default=50, help="Number of timesteps (default: 50)")
    parser.add_argument("--nmax", type=int, default=15, help="Maximum n for FocusFlow loop (default: 15)")
    parser.add_argument("--nmin", type=int, default=8, help="Minimum n for sampling-like phase (default: 8)")
    parser.add_argument("--navg", type=int, default=1, help="Number of averages (default: 1)")
    parser.add_argument("--cfg_src", type=float, default=3.5, help="CFG scale for source (default: 3.5)")
    parser.add_argument("--cfg_tar", type=float, default=13.5, help="CFG scale for target (default: 13.5)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")

    # Mask params
    parser.add_argument("--auto_mask", action="store_true", default=True, help="Enable auto mask (default: True)")
    parser.add_argument("--mask_strength", type=float, default=0.5, help="DiffEdit strength for mask timestep (default: 0.5)")
    parser.add_argument("--mask_n", type=int, default=10, help="Number of samples for mask averaging (default: 10)")
    parser.add_argument("--mask_guidance", type=float, default=5.0, help="Guidance scale used for mask generation (default: 5.0)")
    parser.add_argument("--mask_blur_ks", type=int, default=5, help="Avg-pool blur kernel size (default: 5)")
    parser.add_argument("--mask_dilate_ks", type=int, default=0, help="Max-pool dilation kernel size (default: 0)")
    parser.add_argument("--mask_threshold", type=float, default=0.5, help="Mask threshold (default: 0.5)")
    parser.add_argument("--mask_q_low", type=float, default=0.01, help="Lower percentile clip for mask (default: 0.01)")
    parser.add_argument("--mask_q_high", type=float, default=0.99, help="Upper percentile clip for mask (default: 0.99)")
    parser.add_argument("--mask_seed_base", type=int, default=0, help="Base seed for deterministic mask noise (default: 0)")

    # Pipeline / output
    parser.add_argument(
        "--model_id",
        type=str,
        default="stabilityai/stable-diffusion-3-medium-diffusers",
        help="Model ID (default: stabilityai/stable-diffusion-3-medium-diffusers)",
    )
    parser.add_argument("--drop_t5", action="store_true", help="Drop T5 text encoder (SD3 text_encoder_3)")
    parser.add_argument("--output", type=str, default="focusflow_out.png", help="Output image path (default: focusflow_out.png)")
    parser.add_argument("--save_mask", type=str, default="", help="Optional path to save the soft mask PNG")
    parser.add_argument("--size", type=int, default=512, help="Input/output resolution (default: 512)")

    args = parser.parse_args()

    print("-" * 60)
    print(f"seed={args.seed}  T={args.T}  nmax={args.nmax}  nmin={args.nmin}  navg={args.navg}")
    print(f"cfg_src={args.cfg_src}  cfg_tar={args.cfg_tar}")
    print(f"auto_mask={args.auto_mask}  mask_strength={args.mask_strength}  mask_n={args.mask_n}  mask_guidance={args.mask_guidance}")
    print("-" * 60)

    # Reproducibility
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    cuda = torch.cuda.is_available()
    dtype = torch.float16 if cuda else torch.float32

    print("Loading pipeline...")
    if args.drop_t5:
        pipe = StableDiffusion3Pipeline.from_pretrained(
            args.model_id,
            text_encoder_3=None,
            tokenizer_3=None,
            torch_dtype=dtype if cuda else torch.float32,
            variant="fp16" if cuda else None,
            use_safetensors=True,
        )
    else:
        pipe = StableDiffusion3Pipeline.from_pretrained(
            args.model_id,
            torch_dtype=dtype if cuda else torch.float32,
            variant="fp16" if cuda else None,
            use_safetensors=True,
        )

    # Same offload/slicing strategy as my_FlowEdit.py
    pipe.enable_sequential_cpu_offload()
    pipe.enable_model_cpu_offload()
    pipe.enable_attention_slicing()
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    exec_device = pipe._execution_device
    print("Execution device:", exec_device)

    img = load_image(args.image, size=int(args.size))
    x_src = encode_image_to_latents(pipe, img, device=exec_device, dtype=dtype)
    print("Encoded latents:", tuple(x_src.shape), x_src.dtype, x_src.device)

    out_latents, mask_soft_hw, _ = focusflow(
        pipe=pipe,
        x_src=x_src,
        prompt_src=args.prompt_src,
        prompt_tar=args.prompt_tar,
        negative_prompt=args.negative_prompt,
        T_steps=int(args.T),
        n_avg=int(args.navg),
        src_guidance_scale=float(args.cfg_src),
        tar_guidance_scale=float(args.cfg_tar),
        n_min=int(args.nmin),
        n_max=int(args.nmax),
        mask_soft_hw=None,
        auto_mask=bool(args.auto_mask),
        mask_strength=float(args.mask_strength),
        mask_n=int(args.mask_n),
        mask_guidance=float(args.mask_guidance),
        mask_blur_ks=int(args.mask_blur_ks),
        mask_dilate_ks=int(args.mask_dilate_ks),
        mask_threshold=float(args.mask_threshold),
        mask_q_low=float(args.mask_q_low),
        mask_q_high=float(args.mask_q_high),
        mask_seed_base=int(args.mask_seed_base),
    )

    out_img = decode_latents_to_image(pipe, out_latents)
    out_img.save(args.output)
    print(f"Saved: {args.output}")

    if args.save_mask and mask_soft_hw is not None:
        _save_mask_png(mask_soft_hw, args.save_mask, out_size=int(args.size))
        print(f"Saved mask: {args.save_mask}")

    # cleanup
    del pipe
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()