import argparse
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import os
import time

# --- !!! IMPORTANT DISCLAIMER !!! ---
# By running this script with the safety checker disabled,
# you acknowledge and accept full responsibility for the
# content generated. Ensure your use complies with all
# applicable laws, terms of service, and ethical standards.
# Do not use this script to generate harmful, illegal,
# or unethical content.
# --- !!! IMPORTANT DISCLAIMER !!! ---

def main():
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(
        description="Generate an image from a text prompt using Stable Diffusion on CPU (Safety Checker Disabled).",
        epilog="""WARNING: The safety checker is DISABLED. You are responsible for the generated content."""
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Text prompt for image generation."
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output file path including filename and extension (e.g., 'output/my_image.png', 'generated.jpg')."
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="runwayml/stable-diffusion-v1-5", # This model can still be used, we just disable the checker
        help="Model ID from Hugging Face Hub (e.g., 'runwayml/stable-diffusion-v1-5')."
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=25, # Slightly increased default steps
        help="Number of diffusion steps (higher means potentially better quality but slower)."
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="Guidance scale (higher means image closer to prompt, but potentially less diverse)."
    )
    parser.add_argument(
        "--safety_checker",
        type=bool,
        default=False,
        help="Enable safety checker (default is disabled for this script)."
    )

    args = parser.parse_args()

    # --- Device Setup ---
    device = "cpu"
    print(f"Using device: {device}")
    print("--------------------------------------------------")
    print("!! WARNING: Image generation on CPU is VERY slow !!")
    print("!!         (minutes to potentially hours)       !!")
    print("--------------------------------------------------")
    
    if args.safety_checker == False:
        print("**************************************************")
        print("*** SAFETY CHECKER IS DISABLED ***")
        print("*** You are responsible for the output content. ***")
        print("**************************************************")

    # --- Load Model (with safety checker disabled) ---
    print(f"Loading model: {args.model_id}...")
    start_load_time = time.time()
    try:
        if args.safety_checker == False:
            pipe = StableDiffusionPipeline.from_pretrained(
                args.model_id,
                torch_dtype=torch.float32,
                safety_checker=None,          # <<< --- KEY CHANGE: Disable safety checker
                requires_safety_checker=False # <<< --- KEY CHANGE: Indicate checker isn't required
            )
        else:
            pipe = StableDiffusionPipeline.from_pretrained(
                args.model_id,
                torch_dtype=torch.float32,
            )

        pipe = pipe.to(device)
    except Exception as e:
        print(f"\nError loading model: {e}")
        print("Ensure the model ID is correct and you have enough RAM.")
        print("If the error persists, the model might require specific configurations.")
        return # Exit if model loading fails
    load_time = time.time() - start_load_time
    print(f"Model loaded in {load_time:.2f} seconds.")

    # --- Generate Image ---
    print(f"\nGenerating image for prompt: '{args.prompt}'")
    print(f"Using {args.steps} steps and guidance scale {args.guidance_scale}.")
    print("This will take a significant amount of time...")
    start_gen_time = time.time()
    try:
        # Generate the image
        with torch.no_grad():
             generated_output = pipe(
                 prompt=args.prompt,
                 num_inference_steps=args.steps,
                 guidance_scale=args.guidance_scale
             )
        image = generated_output.images[0]

        # Check if the image is potentially black/blank (sometimes happens on checker removal with certain prompts)
        # Note: This is a *very basic* check and not foolproof for detecting failed generations.
        if image is None or (hasattr(image, 'width') and image.width == 1):
             print("\nWarning: Generated image appears to be blank or invalid.")
             print("This can sometimes happen when the safety checker is removed or with certain prompts.")
             # Optionally, you could exit here or try to generate again.
             # For now, we'll just warn and proceed to save whatever was returned (if anything).
             if image is None: return # Exit if image is truly None

    except Exception as e:
        print(f"\nError during image generation: {e}")
        print("This could be due to memory constraints or issues with the model/parameters.")
        return # Exit if generation fails
    gen_time = time.time() - start_gen_time
    print(f"Image generated in {gen_time:.2f} seconds.")

    # --- Save Image ---
    print(f"\nSaving image to {args.output}...")
    try:
        # Ensure output directory exists
        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            print(f"Created directory: {output_dir}")

        # Determine file format from extension
        file_extension = os.path.splitext(args.output)[1].lower()
        if file_extension == ".jpg" or file_extension == ".jpeg":
            file_format = "JPEG"
            if image.mode == 'RGBA':
                 print("Info: Image has alpha channel, converting to RGB for JPEG saving.")
                 image = image.convert('RGB')
        elif file_extension == ".png":
            file_format = "PNG"
        else:
            if not file_extension:
                 args.output += ".png"
                 print(f"Warning: No file extension provided. Defaulting to PNG format. Saving as {args.output}")
            else:
                 print(f"Warning: Unknown file extension '{file_extension}'. Attempting to save as PNG.")
            file_format = "PNG"

        # Save the image
        image.save(args.output, format=file_format)
        print(f"Image successfully saved to {args.output}")

    except AttributeError:
         print("\nError saving image: The generated image object is invalid (possibly None or blank).")
    except Exception as e:
        print(f"\nError saving image: {e}")
        print("Please ensure the output path is valid and you have write permissions.")


if __name__ == "__main__":
    main()