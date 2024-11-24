import gradio as gr
from tryon_inference import run_inference
import os
import numpy as np
from PIL import Image
import tempfile

def gradio_inference(
    image_data, 
    garment, 
    num_steps=50, 
    guidance_scale=30.0, 
    seed=-1, 
    size=(768,1024)
):
    """Wrapper function for Gradio interface"""
    # Use temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Save inputs to temp directory
        temp_image = os.path.join(tmp_dir, "image.png")
        temp_mask = os.path.join(tmp_dir, "mask.png")
        temp_garment = os.path.join(tmp_dir, "garment.png")
        
        # Extract image and mask from ImageEditor data
        image = image_data["background"]
        mask = image_data["layers"][0]  # First layer contains the mask
        
        # Convert to numpy array and process mask
        mask_array = np.array(mask)
        is_black = np.all(mask_array < 10, axis=2)
        mask = Image.fromarray(((~is_black) * 255).astype(np.uint8))
        
        # Save files to temp directory
        image.save(temp_image)
        mask.save(temp_mask)
        garment.save(temp_garment)
        
        try:
            # Run inference
            _, tryon_result = run_inference(
                image_path=temp_image,
                mask_path=temp_mask,
                garment_path=temp_garment,
                num_steps=num_steps,
                guidance_scale=guidance_scale,
                seed=seed,
                size=size
            )
            return tryon_result
        except Exception as e:
            raise gr.Error(f"Error during inference: {str(e)}")

def create_demo():
    with gr.Blocks() as demo:
        gr.Markdown("""
        # CATVTON FLUX Virtual Try-On Demo
        Upload a model image, an agnostic mask, and a garment image to generate virtual try-on results.
        
        [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/xiaozaa/catvton-flux-alpha)
        [![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)](https://github.com/nftblackmagic/catvton-flux)
        """)
        
        with gr.Column():
            with gr.Row():
                with gr.Column():
                    image_input = gr.ImageMask(
                        label="Model Image (Draw mask where garment should go)", 
                        type="pil",
                        height=600,
                    )
                    gr.Examples(
                        examples=[
                            ["./example/person/00008_00.jpg"],
                            ["./example/person/00055_00.jpg"],
                            ["./example/person/00057_00.jpg"],
                            ["./example/person/00067_00.jpg"],
                            ["./example/person/00069_00.jpg"],
                        ],
                        inputs=[image_input],
                        label="Person Images",
                    )
                with gr.Column():
                    garment_input = gr.Image(label="Garment Image", type="pil", height=600)
                    gr.Examples(
                        examples=[
                            ["./example/garment/04564_00.jpg"],
                            ["./example/garment/00055_00.jpg"],
                            ["./example/garment/00057_00.jpg"],
                            ["./example/garment/00067_00.jpg"],
                            ["./example/garment/00069_00.jpg"],
                        ],
                        inputs=[garment_input],
                        label="Garment Images",
                    ) 
                
            with gr.Row():
                num_steps = gr.Slider(
                    minimum=1, 
                    maximum=100, 
                    value=50, 
                    step=1, 
                    label="Number of Steps"
                )
                guidance_scale = gr.Slider(
                    minimum=1.0, 
                    maximum=50.0, 
                    value=30.0, 
                    step=0.5, 
                    label="Guidance Scale"
                )
                seed = gr.Slider(
                    minimum=-1,
                    maximum=2147483647,
                    step=1,
                    value=-1,
                    label="Seed (-1 for random)"
                )
            
            submit_btn = gr.Button("Generate Try-On", variant="primary")
            
            with gr.Column():
                tryon_output = gr.Image(label="Try-On Result")
                
        with gr.Row():
            gr.Markdown("""
            ### Notes:
            - The model image should be a full-body photo
            - The mask should indicate the region where the garment will be placed
            - The garment image should be on a clean background
            """)
        
        submit_btn.click(
            fn=gradio_inference,
            inputs=[
                image_input,
                garment_input,
                num_steps,
                guidance_scale,
                seed
            ],
            outputs=[tryon_output],
            api_name="try-on"
        )
    
    return demo

if __name__ == "__main__":
    demo = create_demo()
    demo.queue()  # Enable queuing for multiple users
    demo.launch(
        share=True,
        server_name="0.0.0.0"  # Makes the server accessible from other machines
    ) 