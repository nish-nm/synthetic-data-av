import torch
from diffusers import StableDiffusionPipeline

# Load the Stable Diffusion pipeline (v1.5 is a common choice)
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float32)

# Force the pipeline to use CPU. Remove this if you have a GPU.
pipe.to("cpu")

def generate_image(
    prompt: str,
    height: int = 512,
    width: int = 512,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 50,
    seed: int = 0
):
    # Create a CPU generator for reproducibility
    generator = torch.Generator("cpu").manual_seed(seed)
    # Generate the image using the pipeline
    output = pipe(
        prompt,
        height=height,
        width=width,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=generator
    )
    return output.images[0]

if __name__ == "__main__":
    test_prompt = "A futuristic autonomous vehicle navigating a neon-lit urban street at night"
    image = generate_image(test_prompt)
    image.save("generated_sd_image.png")
    print("Image saved as generated_sd_image.png")
