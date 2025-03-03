from huggingface_hub import login

import torch
from diffusers import StableDiffusionPipeline
from safetensors.torch import load_file

# Load base Stable Diffusion model (change to your version if needed)
base_model = "stabilityai/stable-diffusion-2"  # Change if using SDXL or another version

# Load pre-trained Stable Diffusion pipeline
pipe = StableDiffusionPipeline.from_pretrained(base_model, torch_dtype=torch.float16)
pipe.to("cuda" if torch.cuda.is_available() else "cpu")

# Load LoRA weights from Hugging Face (replace with your LoRA model)
lora_weights = "ButterChicken98/plantVillage-stableDiffusion-2-lora_rank_8_cond_concat"  # Change this to your LoRA file or Hugging Face repo
pipe.unet.load_attn_procs(lora_weights, strict=False)

image_list = pipe(
        prompt=f"a healthy tomato leaf",
        negative_prompt="bad quality, blurry",
        num_inference_steps=50,
        height=256,
        width=256,
        guidence_scale =7.5,
        num_images_per_prompt=1
    ).images[0]
image_list.save('/home/saranshvashistha/workspace/diffusers/out/output.png')


