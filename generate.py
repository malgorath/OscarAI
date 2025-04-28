import sys
import torch
import datetime
from diffusers import StableDiffusionXLPipeline


filename_base = "image"
now = datetime.datetime.now()
timestamp = now.strftime("%Y%m%d%H%M%S")
model_id = "stabilityai/stable-diffusion-xl-base-1.0"
pipeline = StableDiffusionXLPipeline.from_pretrained(model_id)
filename = f"{filename_base}-{timestamp}.png"

if torch.cuda.is_available():
    pipeline = pipeline.to("cuda")
    print("Using CUDA for image generation.")
else:
    print("Using CPU for image generation. This might be slower.")
if len(sys.argv) == 2:
    prompt = sys.argv[1]
    image = pipeline(prompt).images[0]
    image.save(filename)
    print("Image generated and saved as {filename}")
else:
    print ("Usage: python script.py <prompt>")
