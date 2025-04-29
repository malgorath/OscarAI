import sys
import torch
import datetime
import os
from diffusers import StableDiffusionXLPipeline

filename_base = "image"
now = datetime.datetime.now()
timestamp = now.strftime("%Y%m%d%H%M%S")
# model_id = "stabilityai/stable-diffusion-xl-base-1.0"
model_id = "runwayml/stable-diffusion-v1-5"

# Load the pipeline with model offloading and increased CPU memory limits
pipeline = StableDiffusionXLPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,  # Use float16 precision
    device_map="balanced",  # Use balanced strategy for CPU/GPU offloading
    max_memory={0: "4GiB", "cpu": "120GiB"}  # Limit GPU to 4GiB and allow CPU to use up to 120GiB
)
pipeline.enable_attention_slicing()  # Enable attention slicing to reduce VRAM usage
pipeline.enable_xformers_memory_efficient_attention()  # Enable memory-efficient attention
pipeline.safety_checker = None  # Disable safety checker to save VRAM

if torch.cuda.is_available():
    torch.cuda.empty_cache()  # Clear unused GPU memory
    torch.cuda.reset_peak_memory_stats()  # Reset memory stats
    print("Using CUDA with model offloading for image generation.")
else:
    print("Using CPU for image generation. This might be slower.")

if len(sys.argv) >= 2:
    prompt = sys.argv[1]
    save_dir = sys.argv[2] if len(sys.argv) > 2 else "."  # Default to current directory
    os.makedirs(save_dir, exist_ok=True)  # Ensure the directory exists
    filename = os.path.join(save_dir, f"{filename_base}-{timestamp}.png")
    
    # Generate image with guidance scale
    image = pipeline(prompt, guidance_scale=7.5).images[0]
    image.save(filename)
    print(f"Image generated and saved as {filename}")
else:
    print("Usage: python script.py <prompt> [save_directory]")