from contextlib import nullcontext
import os
from accelerate.logging import get_logger
import torch
from torch_fidelity import calculate_metrics

logger = get_logger(__name__, log_level="INFO")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"}

def count_images(directory:str)-> int:
    if not os.path.isdir(directory):
        print("Invalid directory path.")
        return 0
    
    count = 0
    for filename in os.listdir(directory):
        if any(filename.lower().endswith(ext) for ext in IMAGE_EXTENSIONS):
            count += 1
    
    return count

def generate_images(pipeline, args, global_step, accelerator, epoch, num_images):
    logger.info(
        f"generating {num_images} images with prompt:"
        f" {args.validation_prompt}. for the calculation of the FID scores"
    )

    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)
    generator = torch.Generator(device=accelerator.device)
    if args.seed is not None:
        generator = generator.manual_seed(args.seed)
    if torch.backends.mps.is_available():
        autocast_ctx = nullcontext()
    else:
        autocast_ctx = torch.autocast(accelerator.device.type)
    with autocast_ctx:
        for i in range(args.num_images):
            image = pipeline(
                prompt = args.validation_prompt,
                negative_prompt = "bad quality, blurry",
                num_inference_steps=50, 
                generator=generator,
                height=args.resolution,
                width = args.resolution,
                ).images[0]
            image.save(os.path.join(args.gen_image_path, f"{global_step}", f"generated_{i + 1:03d}.png"))
            
def compute_fid(args, global_step)-> float:
    metrics = calculate_metrics(
        input1=args.original_image_path,
        input2=os.path.join(args.gen_image_path,f"{global_step}"),
        cuda=True,
        isc=False,
        fid=True,
    )
    logger.info(f"FID: { metrics['frechet_inception_distance']} at the step {global_step}")
    return metrics['frechet_inception_distance']