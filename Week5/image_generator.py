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
):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
    if model_id == "stabilityai/stable-diffusion-2-1":
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id, torch_dtype=torch.float16
        )
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    elif model_id == "stabilityai/sd-turbo":
        pipe = AutoPipelineForText2Image.from_pretrained(
            model_id, torch_dtype=torch.float16, variant="fp16"
        )
    elif model_id == "stabilityai/stable-diffusion-xl-base-1.0":
        pipe = DiffusionPipeline.from_pretrained(
            model_id, torch_dtype=torch.float16, use_safetensors=True, variant="fp16"
        )
    elif model_id == "stabilityai/sdxl-turbo":
        pipe = AutoPipelineForText2Image.from_pretrained(
            model_id, torch_dtype=torch.float16, variant="fp16"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = pipe.to(device)

    # Generate image
    if model_id == "stabilityai/stable-diffusion-2-1":
        image = pipe(prompt).images[0]
    elif model_id == "stabilityai/sd-turbo":
        image = pipe(prompt=prompt, num_inference_steps=1, guidance_scale=0.0).images[0]
    elif model_id == "stabilityai/stable-diffusion-xl-base-1.0":
        image = pipe(prompt=prompt).images[0]
    elif model_id == "stabilityai/sdxl-turbo":
        image = pipe(prompt=prompt, num_inference_steps=1, guidance_scale=0.0).images[0]

    # Save the generated image with a unique ID
    file_id = len(os.listdir(output_folder)) + 1
    image_path = os.path.join(output_folder, f"train_{model_id.split('/')[-1]}_2.png")
    image.save(image_path)

    return image_path


if __name__ == "__main__":
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Generate image using Stable Diffusion model"
    )
    parser.add_argument("--prompt", type=str, help="Prompt for generating the image")
    parser.add_argument("--model", type=str, help="Model for generating the image")
    args = parser.parse_args()
    models = [
        "stabilityai/stable-diffusion-2-1",
        "stabilityai/sd-turbo",
        "stabilityai/stable-diffusion-xl-base-1.0",
        "stabilityai/sdxl-turbo",
    ]
    # Generate image using the provided prompt
    print(f"Model: {models[int(args.model)]}")
    generated_image_path = generate_image(args.prompt, model_id=models[int(args.model)])
    print(f"Generated image saved at: {generated_image_path}")
