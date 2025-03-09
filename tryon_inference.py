import argparse
import torch
from diffusers.utils import load_image, check_min_version
from diffusers import FluxPriorReduxPipeline, FluxFillPipeline
from diffusers import FluxTransformer2DModel
from src.flux.custom_attention_processor import FluxAttnProcessorWithViz
from diffusers.models.attention_processor import Attention
import numpy as np
from torchvision import transforms
from attention_map import visualize_all_heads_all_layers

def run_inference(
    image_path,
    mask_path,
    garment_path,
    size=(576, 768),
    num_steps=50,
    guidance_scale=30,
    seed=42,
    pipe=None
):
    # Build pipeline
    if pipe is None:
        transformer = FluxTransformer2DModel.from_pretrained(
            "xiaozaa/catvton-flux-beta", 
            torch_dtype=torch.bfloat16
        )
        transformer.set_attn_processor(FluxAttnProcessorWithViz())
        
        # for name, module in transformer.named_modules():
        #     if "attn" in name:
        #         print(name, module)
        
        pipe = FluxFillPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            transformer=transformer,
            torch_dtype=torch.bfloat16
        ).to("cuda")
    else:
        pipe.to("cuda")

    pipe.transformer.to(torch.bfloat16)

    # Add transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # For RGB images
    ])
    mask_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # Load and process images
    # print("image_path", image_path)
    image = load_image(image_path).convert("RGB").resize(size)
    mask = load_image(mask_path).convert("RGB").resize(size)
    garment = load_image(garment_path).convert("RGB").resize(size)

    # Transform images using the new preprocessing
    image_tensor = transform(image)
    mask_tensor = mask_transform(mask)[:1]  # Take only first channel
    garment_tensor = transform(garment)

    # Create concatenated images
    inpaint_image = torch.cat([garment_tensor, image_tensor], dim=2)  # Concatenate along width
    garment_mask = torch.zeros_like(mask_tensor)
    extended_mask = torch.cat([garment_mask, mask_tensor], dim=2)

    prompt = f"The pair of images highlights a clothing and its styling on a model, high resolution, 4K, 8K; " \
            f"[IMAGE1] Detailed product shot of a clothing" \
            f"[IMAGE2] The same cloth is worn by a model in a lifestyle setting."

    generator = torch.Generator(device="cuda").manual_seed(seed)
    max_sequence_length = 512
    height = size[1]
    width = size[0] * 2
    result = pipe(
        height=height,
        width=width,
        image=inpaint_image,
        mask_image=extended_mask,
        num_inference_steps=num_steps,
        generator=generator,
        max_sequence_length=max_sequence_length,
        guidance_scale=guidance_scale,
        prompt=prompt,
    ).images[0]
    
    all_attention_probs = []
    for name, module in pipe.transformer.named_modules():
        if isinstance(module, Attention) and hasattr(module, 'processor') and \
           isinstance(module.processor, FluxAttnProcessorWithViz):
            all_attention_probs.append(module.processor.last_attention_probs[:, max_sequence_length:, max_sequence_length:])
            print(name, module)
    
    visualize_all_heads_all_layers(
        all_attention_probs,  # List of attention tensors from all layers
        original_image=inpaint_image,
        latent_height=height//16,  # Adjust to match your model dimensions
        latent_width=width//16,   # Adjust to match your model dimensions
        save_path="all_heads_all_layers.png",
        colormap='plasma',
        brightness_factor=2.5,
        threshold=0.6,
    )
    
    # Split and save results
    width = size[0]
    garment_result = result.crop((0, 0, width, size[1]))
    tryon_result = result.crop((width, 0, width * 2, size[1]))
    
    return garment_result, tryon_result

def main():
    parser = argparse.ArgumentParser(description='Run FLUX virtual try-on inference')
    parser.add_argument('--image', required=True, help='Path to the model image')
    parser.add_argument('--mask', required=True, help='Path to the agnostic mask')
    parser.add_argument('--garment', required=True, help='Path to the garment image')
    parser.add_argument('--output_garment', default='flux_inpaint_garment.png', help='Output path for garment result')
    parser.add_argument('--output_tryon', default='flux_inpaint_tryon.png', help='Output path for try-on result')
    parser.add_argument('--steps', type=int, default=50, help='Number of inference steps')
    parser.add_argument('--guidance_scale', type=float, default=30, help='Guidance scale')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--width', type=int, default=576, help='Width')
    parser.add_argument('--height', type=int, default=768, help='Height')
    
    args = parser.parse_args()
    
    check_min_version("0.30.2")
    
    garment_result, tryon_result = run_inference(
        image_path=args.image,
        mask_path=args.mask,
        garment_path=args.garment,
        num_steps=args.steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        size=(args.width, args.height)
    )
    output_tryon_path=args.output_tryon
    
    tryon_result.save(output_tryon_path)
    
    print("Successfully saved garment and try-on images")

if __name__ == "__main__":
    main()