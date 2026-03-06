import argparse
import gc
import os
import sys
from typing import Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from diffusers import FlowMatchEulerDiscreteScheduler, StableDiffusion3Pipeline
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps


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


def scale_noise(
    scheduler,
    sample: torch.FloatTensor,
    timestep: Union[float, torch.FloatTensor],
    noise: Optional[torch.FloatTensor] = None,
) -> torch.FloatTensor:
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
    pipe._guidance_scale = float(max(src_guidance_scale, tar_guidance_scale, 1.0001))

    (
        src_prompt_embeds,
        src_negative_prompt_embeds,
        src_pooled_prompt_embeds,
        src_negative_pooled_prompt_embeds,
    ) = pipe.encode_prompt(
        prompt=src_prompt,
        prompt_2=None,
        prompt_3=None,
        negative_prompt=negative_prompt,
        do_classifier_free_guidance=True,
        device=device,
    )

    (
        tar_prompt_embeds,
        tar_negative_prompt_embeds,
        tar_pooled_prompt_embeds,
        tar_negative_pooled_prompt_embeds,
    ) = pipe.encode_prompt(
        prompt=tar_prompt,
        prompt_2=None,
        prompt_3=None,
        negative_prompt=negative_prompt,
        do_classifier_free_guidance=True,
        device=device,
    )

    src_tar_prompt_embeds = torch.cat(
        [src_negative_prompt_embeds, src_prompt_embeds, tar_negative_prompt_embeds, tar_prompt_embeds],
        dim=0,
    )
    src_tar_pooled_prompt_embeds = torch.cat(
        [src_negative_pooled_prompt_embeds, src_pooled_prompt_embeds, tar_negative_pooled_prompt_embeds, tar_pooled_prompt_embeds],
        dim=0,
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


def _prep_mask_for_latents(mask_hw: torch.Tensor, x_src: torch.Tensor) -> torch.Tensor:
    device = x_src.device
    dtype = x_src.dtype
    bsz, channels, h, w = x_src.shape

    m = mask_hw if torch.is_tensor(mask_hw) else torch.tensor(mask_hw)
    m = m.to(device=device, dtype=torch.float32)

    if m.ndim == 2:
        m = m[None, None, ...]
    elif m.ndim == 3:
        m = m[None, ...]

    if m.shape[-2:] != (h, w):
        m = F.interpolate(m, size=(h, w), mode="bilinear", align_corners=False)

    m = m.clamp(0.0, 1.0)

    if m.shape[0] == 1 and bsz > 1:
        m = m.expand(bsz, -1, -1, -1)
    if m.shape[1] == 1 and channels > 1:
        m = m.expand(-1, channels, -1, -1)

    return m.to(dtype=dtype)


def _soften_mask(mask01_hw: torch.Tensor, blur_ks: int = 7, dilate_ks: int = 3) -> torch.Tensor:
    m = mask01_hw[None, None, ...]

    if dilate_ks and dilate_ks > 1:
        pad = dilate_ks // 2
        m = F.max_pool2d(m, kernel_size=dilate_ks, stride=1, padding=pad)

    if blur_ks and blur_ks > 1:
        pad = blur_ks // 2
        m = F.avg_pool2d(m, kernel_size=blur_ks, stride=1, padding=pad)

    return m.squeeze(0).squeeze(0).clamp(0.0, 1.0)


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
    mask_soft_hw: Optional[torch.Tensor] = None,
    auto_mask: bool = False,
    mask_threshold: float = 0.5,
):
    del auto_mask, mask_threshold
    device = x_src.device

    scheduler = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config)
    timesteps, T_steps = retrieve_timesteps(scheduler, T_steps, device, timesteps=None)
    pipe._num_timesteps = len(timesteps)

    pipe._guidance_scale = float(max(src_guidance_scale, tar_guidance_scale, 1.0001))

    src_tar_prompt_embeds, src_tar_pooled_prompt_embeds = _encode_prompts_sd3(
        pipe=pipe,
        device=device,
        src_prompt=prompt_src,
        tar_prompt=prompt_tar,
        negative_prompt=negative_prompt,
        src_guidance_scale=float(src_guidance_scale),
        tar_guidance_scale=float(tar_guidance_scale),
    )

    m = None
    if mask_soft_hw is not None:
        m = _prep_mask_for_latents(mask_soft_hw.to(device=device), x_src)

    zt_edit = x_src.clone()

    for i, t in enumerate(timesteps):
        if T_steps - i > int(n_max):
            continue

        t_i = t / 1000
        if i + 1 < len(timesteps):
            t_im1 = timesteps[i + 1] / 1000
        else:
            t_im1 = torch.zeros_like(t_i).to(t_i.device)

        if T_steps - i > int(n_min):
            v_delta_avg = torch.zeros_like(x_src)

            for _ in range(int(n_avg)):
                fwd_noise = torch.randn_like(x_src)

                zt_src = (1 - t_i) * x_src + t_i * fwd_noise
                zt_tar = zt_edit + zt_src - x_src

                latent_in = torch.cat([zt_src, zt_src, zt_tar, zt_tar], dim=0)
                vt_src, vt_tar = calc_v_sd3(
                    pipe,
                    latent_in,
                    src_tar_prompt_embeds,
                    src_tar_pooled_prompt_embeds,
                    float(src_guidance_scale),
                    float(tar_guidance_scale),
                    t,
                )

                delta = vt_tar - vt_src
                if m is not None:
                    delta = m * delta

                v_delta_avg += delta / float(n_avg)

            zt_edit = zt_edit.to(torch.float32)
            zt_edit = zt_edit + (t_im1 - t_i) * v_delta_avg.to(torch.float32)
            zt_edit = zt_edit.to(dtype=x_src.dtype)

        else:
            if i == T_steps - int(n_min):
                fwd_noise = torch.randn_like(x_src)
                xt_src = scale_noise(scheduler, x_src, t, noise=fwd_noise)
                xt_tar = zt_edit + xt_src - x_src

            latent_in = torch.cat([xt_tar, xt_tar, xt_tar, xt_tar], dim=0)
            vt_src, vt_tar = calc_v_sd3(
                pipe,
                latent_in,
                src_tar_prompt_embeds,
                src_tar_pooled_prompt_embeds,
                float(src_guidance_scale),
                float(tar_guidance_scale),
                t,
            )

            if m is None:
                v_final = vt_tar
            else:
                v_final = m * vt_tar + (1 - m) * vt_src

            xt_tar = xt_tar.to(torch.float32)
            xt_tar = (xt_tar + (t_im1 - t_i) * v_final.to(torch.float32)).to(dtype=x_src.dtype)

    out_latents = zt_edit if int(n_min) == 0 else xt_tar
    return out_latents


def _parse_point(point_str: str) -> Tuple[float, float]:
    try:
        x, y = [float(v.strip()) for v in point_str.split(",")]
    except Exception as exc:
        raise ValueError("--sam_point must be provided as 'x,y' with normalized values in [0,1].") from exc

    if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
        raise ValueError("--sam_point values must be in [0,1].")
    return x, y


def _choose_mask_index(masks: np.ndarray, scores: np.ndarray, px: int, py: int) -> int:
    contains_point = masks[:, py, px].astype(np.float32)
    rank = scores + 10.0 * contains_point
    return int(np.argmax(rank))


def _sam_mask_segment_anything(
    image_pil: Image.Image,
    model_type: str,
    checkpoint_path: str,
    point_xy_norm: Tuple[float, float],
    multimask: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    try:
        from segment_anything import SamPredictor, sam_model_registry
    except ImportError as exc:
        raise RuntimeError("segment_anything is not installed. Install with: pip install segment-anything") from exc

    if not checkpoint_path:
        raise RuntimeError("--sam_checkpoint is required when using segment_anything backend.")
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"SAM checkpoint not found: {checkpoint_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    image_np = np.array(image_pil)
    h, w = image_np.shape[:2]
    px = int(round(point_xy_norm[0] * (w - 1)))
    py = int(round(point_xy_norm[1] * (h - 1)))

    predictor.set_image(image_np)
    masks, scores, logits = predictor.predict(
        point_coords=np.array([[px, py]]),
        point_labels=np.array([1]),
        multimask_output=bool(multimask),
    )

    idx = _choose_mask_index(masks, scores, px, py) if multimask else 0
    dog_mask_hw = masks[idx].astype(np.uint8)
    dog_mask_soft_hw = torch.sigmoid(torch.from_numpy(logits[idx]).float()).numpy()
    dog_mask_soft_hw = np.clip(dog_mask_soft_hw, 0.0, 1.0)
    return dog_mask_hw, dog_mask_soft_hw


def _sam_mask_transformers(
    image_pil: Image.Image,
    sam_hf_id: str,
    point_xy_norm: Tuple[float, float],
    multimask: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    try:
        from transformers import SamModel, SamProcessor
    except ImportError as exc:
        raise RuntimeError(
            "transformers SAM backend is not installed. Install with: pip install transformers"
        ) from exc

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SamModel.from_pretrained(sam_hf_id).to(device)
    processor = SamProcessor.from_pretrained(sam_hf_id)

    w, h = image_pil.size
    px = point_xy_norm[0] * (w - 1)
    py = point_xy_norm[1] * (h - 1)

    inputs = processor(
        images=image_pil,
        input_points=[[[float(px), float(py)]]],
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, multimask_output=bool(multimask))

    pred_masks = processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu(),
    )[0]

    iou_scores = outputs.iou_scores[0].detach().cpu().numpy()
    masks = pred_masks[0].detach().cpu().numpy() > 0.0

    px_i = int(round(px))
    py_i = int(round(py))
    idx = _choose_mask_index(masks, iou_scores, px_i, py_i) if multimask else 0

    dog_mask_hw = masks[idx].astype(np.uint8)
    dog_mask_soft_hw = torch.sigmoid(pred_masks[idx]).detach().cpu().numpy()
    dog_mask_soft_hw = np.clip(dog_mask_soft_hw, 0.0, 1.0)
    return dog_mask_hw, dog_mask_soft_hw


def generate_mask_with_sam(
    image_pil: Image.Image,
    sam_backend: str,
    sam_checkpoint: str,
    sam_model_type: str,
    sam_hf_id: str,
    sam_point: Tuple[float, float],
    sam_multimask: bool,
) -> Tuple[np.ndarray, np.ndarray, str]:
    backend = sam_backend

    if backend == "auto":
        try:
            import segment_anything  # noqa: F401

            if sam_checkpoint and os.path.isfile(sam_checkpoint):
                backend = "segment_anything"
            else:
                backend = "transformers"
        except Exception:
            backend = "transformers"

    if backend == "segment_anything":
        dog_mask_hw, dog_mask_soft_hw = _sam_mask_segment_anything(
            image_pil=image_pil,
            model_type=sam_model_type,
            checkpoint_path=sam_checkpoint,
            point_xy_norm=sam_point,
            multimask=sam_multimask,
        )
        return dog_mask_hw, dog_mask_soft_hw, backend

    if backend == "transformers":
        dog_mask_hw, dog_mask_soft_hw = _sam_mask_transformers(
            image_pil=image_pil,
            sam_hf_id=sam_hf_id,
            point_xy_norm=sam_point,
            multimask=sam_multimask,
        )
        return dog_mask_hw, dog_mask_soft_hw, backend

    raise RuntimeError(
        "No valid SAM backend available. Install one of:\n"
        "  - segment-anything + checkpoint (.pth), or\n"
        "  - transformers (with a SAM checkpoint on Hugging Face)."
    )


def _make_overlay(image_np: np.ndarray, mask01: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    overlay = image_np.astype(np.float32).copy()
    red = np.zeros_like(overlay)
    red[..., 0] = 255.0
    
    # Redimensionner le masque à la taille de l'image si nécessaire
    if mask01.shape[:2] != image_np.shape[:2]:
        mask_pil = Image.fromarray((mask01 * 255).astype(np.uint8))
        mask_pil = mask_pil.resize((image_np.shape[1], image_np.shape[0]), Image.BILINEAR)
        mask01 = np.array(mask_pil).astype(np.float32) / 255.0
    
    mask3 = np.repeat(mask01[..., None], 3, axis=2)
    out = overlay * (1.0 - alpha * mask3) + red * (alpha * mask3)
    return np.clip(out, 0, 255).astype(np.uint8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FocusFlow + SAM background editing for SD3")

    parser.add_argument("--demo_dog_snow", action="store_true", help="Run preset demo on data/imgs/dog_snow.png")
    parser.add_argument("--image", type=str, default="", help="Input image path")
    parser.add_argument("--prompt_src", type=str, default="", help="Source prompt")
    parser.add_argument("--prompt_tar", type=str, default="", help="Target prompt")
    parser.add_argument("--negative_prompt", type=str, default="")

    parser.add_argument(
        "--model_id",
        type=str,
        default="stabilityai/stable-diffusion-3-medium-diffusers",
    )
    parser.add_argument("--T", type=int, default=50)
    parser.add_argument("--nmax", type=int, default=15)
    parser.add_argument("--nmin", type=int, default=8)
    parser.add_argument("--navg", type=int, default=1)
    parser.add_argument("--cfg_src", type=float, default=3.5)
    parser.add_argument("--cfg_tar", type=float, default=13.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--size", type=int, default=512)

    parser.add_argument("--sam_backend", choices=["segment_anything", "transformers", "auto"], default="segment_anything")
    parser.add_argument("--sam_checkpoint", type=str, default="checkpoint/sam_vit_h_4b8939.pth")
    parser.add_argument("--sam_model_type", type=str, default="vit_h")
    parser.add_argument("--sam_hf_id", type=str, default="facebook/sam-vit-base")
    parser.add_argument("--sam_point", type=str, default="0.5,0.5")
    parser.add_argument("--sam_multimask", action="store_true")

    parser.add_argument("--mask_blur_ks", type=int, default=7)
    parser.add_argument("--mask_dilate_ks", type=int, default=3)

    parser.add_argument("--out_img", type=str, default="outputs/dog_snow_focusflow_sam_edit.png")
    parser.add_argument("--out_plot", type=str, default="outputs/dog_snow_focusflow_sam_subplot.png")

    if len(sys.argv) == 1:
        return parser.parse_args(["--demo_dog_snow"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.demo_dog_snow:
        args.image = "data/imgs/dog_snow.png"
        args.prompt_src = "A snowy field with a dog."
        args.prompt_tar = "A green grassy field with a dog."

    if not args.image or not args.prompt_src or not args.prompt_tar:
        raise ValueError("You must provide --image, --prompt_src, and --prompt_tar, or use --demo_dog_snow.")

    sam_point = _parse_point(args.sam_point)

    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    os.makedirs(os.path.dirname(args.out_img) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.out_plot) or ".", exist_ok=True)

    img = load_image(args.image, size=int(args.size))

    dog_mask_hw, dog_mask_soft_hw, used_backend = generate_mask_with_sam(
        image_pil=img,
        sam_backend=args.sam_backend,
        sam_checkpoint=args.sam_checkpoint,
        sam_model_type=args.sam_model_type,
        sam_hf_id=args.sam_hf_id,
        sam_point=sam_point,
        sam_multimask=bool(args.sam_multimask),
    )
    print(f"SAM backend used: {used_backend}")

    dog_soft_t = torch.from_numpy(dog_mask_soft_hw).float()
    dog_soft_t = _soften_mask(dog_soft_t, blur_ks=int(args.mask_blur_ks), dilate_ks=0)

    background_mask_soft_hw = (1.0 - dog_soft_t).clamp(0.0, 1.0)
    background_mask_soft_hw = _soften_mask(
        background_mask_soft_hw,
        blur_ks=int(args.mask_blur_ks),
        dilate_ks=int(args.mask_dilate_ks),
    )

    cuda = torch.cuda.is_available()
    dtype = torch.float16 if cuda else torch.float32

    pipe = StableDiffusion3Pipeline.from_pretrained(
        args.model_id,
        torch_dtype=dtype if cuda else torch.float32,
        variant="fp16" if cuda else None,
        use_safetensors=True,
    )

    pipe.enable_sequential_cpu_offload()
    pipe.enable_model_cpu_offload()
    pipe.enable_attention_slicing()
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    exec_device = pipe._execution_device
    x_src = encode_image_to_latents(pipe, img, device=exec_device, dtype=dtype)

    out_latents = focusflow(
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
        mask_soft_hw=background_mask_soft_hw,
        auto_mask=False,
    )

    out_img = decode_latents_to_image(pipe, out_latents)
    out_img.save(args.out_img)

    img_np = np.array(img)
    mask_np = background_mask_soft_hw.detach().cpu().numpy()
    overlay_np = _make_overlay(img_np, mask_np, alpha=0.4)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(img_np)
    axes[0].set_title("image_orig")
    axes[0].axis("off")

    axes[1].imshow(mask_np, cmap="gray", vmin=0.0, vmax=1.0)
    axes[1].set_title("mask (background editable)")
    axes[1].axis("off")

    axes[2].imshow(overlay_np)
    axes[2].set_title("mask_overlay")
    axes[2].axis("off")

    axes[3].imshow(np.array(out_img))
    axes[3].set_title("edited_img")
    axes[3].axis("off")

    plt.tight_layout()
    fig.savefig(args.out_plot, dpi=150)
    plt.close(fig)

    print(f"Edited image saved to: {args.out_img}")
    print(f"Subplot figure saved to: {args.out_plot}")

    del pipe
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

# How to run:
#   python FocusFlow_SAM.py --demo_dog_snow
#   python FocusFlow_SAM.py --image data/imgs/dog_snow.png --prompt_src "A snowy field with a dog." --prompt_tar "A green grassy field with a dog." --sam_backend auto
