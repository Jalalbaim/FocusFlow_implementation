import gc
import argparse
import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusion3Pipeline
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

def load_image(path, size=512):
    img = Image.open(path).convert("RGB")
    img = img.resize((size, size), Image.BICUBIC)
    return img


@torch.no_grad()
def encode_image_to_latents(pipe, image, device, dtype):
    arr = np.array(image).astype(np.float32) / 255.0
    x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    x = (x * 2.0 - 1.0).to(device=device, dtype=dtype)

    latents = pipe.vae.encode(x).latent_dist.sample()
    latents = latents * pipe.vae.config.scaling_factor
    return latents


@torch.no_grad()
def decode_latents_to_image(pipe, latents):
    latents = latents / pipe.vae.config.scaling_factor
    img = pipe.vae.decode(latents).sample
    img = (img / 2 + 0.5).clamp(0, 1)
    img = img[0].permute(1, 2, 0).float().cpu().numpy()
    img = (img * 255).round().astype(np.uint8)
    return Image.fromarray(img)


def _as_1d_float_timestep(t, device):
    if not torch.is_tensor(t):
        t = torch.tensor([t], device=device, dtype=torch.float32)
    else:
        t = t.to(device=device, dtype=torch.float32)
        if t.ndim == 0:
            t = t[None]
        elif t.ndim != 1:
            t = t.view(-1)
    return t


@torch.no_grad()
def get_text_embeddings(pipe, prompt, negative_prompt=""):
    device = pipe._execution_device
    return pipe.encode_prompt(
        prompt=prompt,
        prompt_2=prompt,
        prompt_3=None,
        negative_prompt=negative_prompt,
        negative_prompt_2=negative_prompt,
        negative_prompt_3=None,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=True,
    )


@torch.no_grad()
def sd3_velocity(pipe, latents, timestep, text_embeds_bundle, cfg_scale, low_vram_cfg=True):

    prompt_embeds, neg_prompt_embeds, pooled, neg_pooled = text_embeds_bundle

    t = _as_1d_float_timestep(timestep, latents.device)


    v_uncond = pipe.transformer(
        hidden_states=latents,
        timestep=t,
        encoder_hidden_states=neg_prompt_embeds,
        pooled_projections=neg_pooled,
        return_dict=False,
    )[0]

    v_cond = pipe.transformer(
        hidden_states=latents,
        timestep=t,
        encoder_hidden_states=prompt_embeds,
        pooled_projections=pooled,
        return_dict=False,
    )[0]

    return v_uncond + cfg_scale * (v_cond - v_uncond)



def _euler_update(sample, v, dt):
    return (sample.float() + dt * v.float()).to(sample.dtype)


@torch.no_grad()
def flowedit(
    pipe,
    x_src,
    prompt_src,
    prompt_tar,
    T=50,
    nmax=33,
    nmin=0,
    navg=1,
    cfg_src=3.5,
    cfg_tar=13.5,
    seed=0,
    low_vram_cfg=True,
    empty_cache_every=8,
):
    device = pipe._execution_device
    x_src = x_src.to(device)
    g = torch.Generator(device=device).manual_seed(seed)

    pipe.scheduler.set_timesteps(T, device=device)
    timesteps = pipe.scheduler.timesteps
    sigmas = pipe.scheduler.sigmas.to(device)

    print("Scheduler:", type(pipe.scheduler).__name__)
    print("len(timesteps) =", len(timesteps))
    print("len(sigmas)    =", len(sigmas))

    emb_src = get_text_embeddings(pipe, prompt_src or "", negative_prompt="")
    emb_tar = get_text_embeddings(pipe, prompt_tar, negative_prompt="")

    z_fe = x_src.clone()

    for k, i in enumerate(range(nmax, nmin, -1), start=1):
        assert 0 <= i < len(timesteps)
        assert i + 1 < len(sigmas)

        t_i = _as_1d_float_timestep(timesteps[i], device)
        sigma_i = sigmas[i]

        v_delta_accum = None

        for _ in range(navg):
            n = randn_like_gen(x_src, g)

            # FlowMatch forward noising
            z_src_hat = pipe.scheduler.scale_noise(x_src, t_i, noise=n)
            z_tar_hat = z_fe + z_src_hat - x_src

            v_tar = sd3_velocity(pipe, z_tar_hat, t_i, emb_tar, cfg_tar, low_vram_cfg=low_vram_cfg)
            v_src = sd3_velocity(pipe, z_src_hat, t_i, emb_src, cfg_src, low_vram_cfg=low_vram_cfg)

            v_delta = (v_tar - v_src)
            v_delta_accum = v_delta if v_delta_accum is None else (v_delta_accum + v_delta)

        v_delta = v_delta_accum / float(navg)

        dt = (sigmas[i + 1] - sigmas[i]).item()
        z_fe = _euler_update(z_fe, v_delta, dt)

        if empty_cache_every and (k % empty_cache_every == 0) and torch.cuda.is_available():
            torch.cuda.empty_cache()

    if nmin == 0:
        return z_fe

    t_m = _as_1d_float_timestep(timesteps[nmin], device)
    n = randn_like_gen(x_src, g)
    z_src_hat = pipe.scheduler.scale_noise(x_src, t_m, noise=n)
    z_tar = z_fe + z_src_hat - x_src

    for i in range(nmin, 0, -1):
        t_i = _as_1d_float_timestep(timesteps[i], device)
        v = sd3_velocity(pipe, z_tar, t_i, emb_tar, cfg_tar, low_vram_cfg=low_vram_cfg)
        dt = (sigmas[i + 1] - sigmas[i]).item()
        z_tar = _euler_update(z_tar, v, dt)

    return z_tar


def randn_like_gen(x, g=None):
    if g is None:
        return torch.randn(x.shape, device=x.device, dtype=x.dtype)
    return torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=g)


def main():
    parser = argparse.ArgumentParser(description="FlowEdit Image Editing")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image")
    parser.add_argument("--prompt_src", type=str, required=True, help="Source prompt describing the input image")
    parser.add_argument("--prompt_tar", type=str, required=True, help="Target prompt for the edited image")
    parser.add_argument("--T", type=int, default=50, help="Number of timesteps (default: 50)")
    parser.add_argument("--nmax", type=int, default=33, help="Maximum n for FlowEdit loop (default: 33)")
    parser.add_argument("--nmin", type=int, default=8, help="Minimum n for FlowEdit loop (default: 8)")
    parser.add_argument("--navg", type=int, default=4, help="Number of averages (default: 4)")
    parser.add_argument("--cfg_src", type=float, default=2.0, help="CFG scale for source (default: 2.0)")
    parser.add_argument("--cfg_tar", type=float, default=10.0, help="CFG scale for target (default: 10.0)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")
    parser.add_argument("--model_id", type=str, default="stabilityai/stable-diffusion-3-medium-diffusers", 
                        help="Model ID (default: stabilityai/stable-diffusion-3-medium-diffusers)")
    parser.add_argument("--drop_t5", action="store_true", help="Drop T5 text encoder")
    parser.add_argument("--low_vram_cfg", action="store_true", default=True, help="Use low VRAM CFG mode (default: True)")
    parser.add_argument("--output", type=str, default="flowedit_out.png", help="Output image path (default: flowedit_out.png)")
    
    args = parser.parse_args()
    
    model_id = args.model_id
    test_img = args.image
    prompt_src = args.prompt_src
    prompt_tar = args.prompt_tar
    T = args.T
    nmax = args.nmax
    nmin = args.nmin
    navg = args.navg
    cfg_src = args.cfg_src
    cfg_tar = args.cfg_tar
    seed = args.seed
    drop_t5 = args.drop_t5
    low_vram_cfg = args.low_vram_cfg

    print("-" * 60)
    print(f"seed={seed}  T={T}  nmax={nmax}  nmin={nmin}  navg={navg}  cfg_src={cfg_src}  cfg_tar={cfg_tar}")
    print(f"DROP_T5={drop_t5}  LOW_VRAM_CFG={low_vram_cfg}")
    print("-" * 60)

    cuda = torch.cuda.is_available()
    dtype = torch.float16 if cuda else torch.float32

    print("Loading pipeline...")
    
    pipe = StableDiffusion3Pipeline.from_pretrained(
        model_id,
        text_encoder_3=None,
        tokenizer_3=None,
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
    print("Execution device:", exec_device)

    img = load_image(test_img, size=512)
    x_src = encode_image_to_latents(pipe, img, device=exec_device, dtype=dtype)
    print("Encoded latents:", tuple(x_src.shape), x_src.dtype, x_src.device)

    emb_tar = get_text_embeddings(pipe, prompt_tar, negative_prompt="")
    
    z_out = flowedit(
        pipe=pipe,
        x_src=x_src,
        prompt_src=prompt_src,
        prompt_tar=prompt_tar,
        T=T,
        nmax=nmax,
        nmin=nmin,
        navg=navg,
        cfg_src=cfg_src,
        cfg_tar=cfg_tar,
        seed=seed,
        low_vram_cfg=low_vram_cfg,
    )

    out_img = decode_latents_to_image(pipe, z_out)
    out_img.save(args.output)
    print(f"Saved: {args.output}")

    # cleanup
    del pipe
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()