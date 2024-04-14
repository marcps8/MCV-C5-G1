import os
import torch
from diffusers import (
    AutoPipelineForText2Image,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    StableDiffusionPipeline,
)

def generate_image(
    prompt,
    model_id="stabilityai/stable-diffusion-2-1",
    output_folder="./generated_images",
    cfg_strength=0.5,
    num_denoising_steps=50,
):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
    if model_id == "stabilityai/stable-diffusion-2-1":
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id, torch_dtype=torch.float16
        )
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    elif model_id == "stabilityai/stable-diffusion-xl-base-1.0":
        pipe = DiffusionPipeline.from_pretrained(
            model_id, torch_dtype=torch.float16, use_safetensors=True, variant="fp16"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = pipe.to(device)

    # Generate image
    if model_id == "stabilityai/stable-diffusion-2-1":
        image = pipe(
            prompt=prompt,
            cfg_strength=cfg_strength,
            num_denoising_steps=num_denoising_steps,
        ).images[0]
    elif model_id == "stabilityai/stable-diffusion-xl-base-1.0":
        image = pipe(
            prompt=prompt,
            cfg_strength=cfg_strength,
            num_denoising_steps=num_denoising_steps,
        ).images[0]

    # Save the generated image with a unique ID
    file_id = len(os.listdir(output_folder)) + 1
    image_path = os.path.join(output_folder, f"train_{file_id}.png")
    image.save(image_path)

    return image_path

if __name__ == "__main__":
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Generate image using Stable Diffusion model"
    )
    parser.add_argument("--prompt", type=str, help="Prompt for generating the image")
    parser.add_argument("--model", type=int, help="Index of the model for generating the image")
    args = parser.parse_args()

    models = [
        "stabilityai/stable-diffusion-2-1",
        "stabilityai/sd-turbo",
        "stabilityai/stable-diffusion-xl-base-1.0",
        "stabilityai/sdxl-turbo",
    ]
    
    # Configuration values for testing
    cfg_strength = 0.7
    num_denoising_steps = 50

    # Generate image using the provided prompt and model
    print(f"Model: {models[args.model]}")
    generated_image_path = generate_image(
        args.prompt,
        model_id=models[args.model],
        cfg_strength=cfg_strength,
        num_denoising_steps=num_denoising_steps,
    )
    print(f"Generated image saved at: {generated_image_path}")
