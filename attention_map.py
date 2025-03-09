import matplotlib.pyplot as plt
import numpy as np
import torch
import math
from PIL import Image
import torchvision.transforms as T
import matplotlib.cm as cm
# import imageio
from tempfile import TemporaryDirectory

def create_stylish_attention_heatmap(
    attention_probs,
    original_image, 
    latent_height,
    latent_width,
    head_idx=0,
    batch_idx=0,
    save_path="stylish_attention_heatmap.png",
    colormap='plasma',  # 'plasma', 'viridis', 'inferno', etc.
    brightness_factor=2.0,  # Controls how bright the hotspots appear
    threshold=0.5  # Only show attention above this threshold (0-1)
):
    """
    Creates a clean, stylish attention heatmap overlay on the original image,
    similar to the example image with bright colored spots on clothing.
    """
    # Reshape if needed
    if len(attention_probs.shape) == 3:
        num_batch_and_heads = attention_probs.shape[0]
        seq_len = attention_probs.shape[1]
        batch_size = 1
        num_heads = num_batch_and_heads // batch_size
        attention_probs = attention_probs.reshape(batch_size, num_heads, seq_len, seq_len)
    
    # Extract attention map for the specified head, batch item
    attn_map = attention_probs[batch_idx, head_idx].cpu().detach().numpy()
    
    # Compute the average attention received by each token
    avg_attention_received = attn_map.mean(axis=0)
    
    # Reshape to latent space dimensions
    attention_latent = avg_attention_received.reshape(latent_height, latent_width)
    
    # Convert original image to numpy if it's a tensor
    if isinstance(original_image, torch.Tensor):
        img_np = original_image.cpu().permute(1, 2, 0).numpy()
        img_np = np.clip(img_np, 0, 1)
    else:
        # If it's a PIL image, convert to numpy
        img_np = np.array(original_image) / 255.0
    
    # Resize attention map to match original image dimensions
    img_height, img_width = img_np.shape[:2]
    attention_resized = torch.nn.functional.interpolate(
        torch.tensor(attention_latent).unsqueeze(0).unsqueeze(0),
        size=(img_height, img_width),
        mode='bicubic'
    ).squeeze().numpy()
    
    # Normalize the attention values
    attention_normalized = (attention_resized - attention_resized.min()) / (attention_resized.max() - attention_resized.min())
    
    # Apply threshold to only show high attention areas
    thresholded_attention = np.copy(attention_normalized)
    thresholded_attention[thresholded_attention < threshold] = 0
    
    # Create the colored heatmap using the specified colormap
    cmap = plt.get_cmap(colormap)
    colored_heatmap = cmap(thresholded_attention)
    
    # Make the background transparent where attention is below threshold
    colored_heatmap[..., 3] = thresholded_attention * brightness_factor
    
    # Create a figure with dark background for better visualization
    plt.figure(figsize=(10, 10))
    plt.style.use('dark_background')
    
    # Add subtle background glow effect
    plt.imshow(np.ones_like(img_np) * 0.2, alpha=0.3)  # Subtle background
    
    # Plot the original image
    plt.imshow(img_np)
    
    # Overlay the heatmap
    plt.imshow(colored_heatmap, alpha=0.8)
    
    # Remove axes and create a tight layout
    plt.axis('off')
    plt.tight_layout(pad=0)
    
    # Save with high DPI for better quality
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    print(f"Stylish attention heatmap saved to {save_path}")
    
    return colored_heatmap

# Function to visualize multiple layers with this style
def create_multi_layer_stylish_heatmap(
    all_layers_attention_probs,
    original_image,
    latent_height,
    latent_width,
    head_idx=0,
    batch_idx=0,
    save_path="multi_layer_stylish_heatmap.png",
    colormap='plasma',
    brightness_factor=2.0,
    threshold=0.5,
    layer_weights=None  # Weights for each layer (None for equal weighting)
):
    """
    Creates a combined stylish heatmap from multiple layers
    """
    num_layers = len(all_layers_attention_probs)
    
    # Default to equal weighting if not specified
    if layer_weights is None:
        layer_weights = [1.0/num_layers] * num_layers
    
    # Convert original image to numpy if it's a tensor
    if isinstance(original_image, torch.Tensor):
        img_np = original_image.cpu().permute(1, 2, 0).numpy()
        img_np = np.clip(img_np, 0, 1)
    else:
        img_np = np.array(original_image) / 255.0
    
    img_height, img_width = img_np.shape[:2]
    combined_attention = np.zeros((img_height, img_width))
    
    # Process each layer
    for layer_idx, attention_probs in enumerate(all_layers_attention_probs):
        if len(attention_probs.shape) == 3:
            num_batch_and_heads = attention_probs.shape[0]
            seq_len = attention_probs.shape[1]
            batch_size = 1
            num_heads = num_batch_and_heads // batch_size
            attention_probs = attention_probs.reshape(batch_size, num_heads, seq_len, seq_len)
        
        # Extract attention map
        attn_map = attention_probs[batch_idx, head_idx].cpu().detach().numpy()
        avg_attention = attn_map.mean(axis=0)
        attention_latent = avg_attention.reshape(latent_height, latent_width)
        
        # Resize to match original image
        attention_resized = torch.nn.functional.interpolate(
            torch.tensor(attention_latent).unsqueeze(0).unsqueeze(0),
            size=(img_height, img_width),
            mode='bicubic'
        ).squeeze().numpy()
        
        # Normalize
        attention_normalized = (attention_resized - attention_resized.min()) / (attention_resized.max() - attention_resized.min())
        
        # Add to combined attention with layer weight
        combined_attention += attention_normalized * layer_weights[layer_idx]
    
    # Re-normalize the combined attention
    combined_normalized = (combined_attention - combined_attention.min()) / (combined_attention.max() - combined_attention.min())
    
    # Apply threshold
    thresholded_attention = np.copy(combined_normalized)
    thresholded_attention[thresholded_attention < threshold] = 0
    
    # Create colored heatmap
    cmap = plt.get_cmap(colormap)
    colored_heatmap = cmap(thresholded_attention)
    colored_heatmap[..., 3] = thresholded_attention * brightness_factor
    
    # Create figure with dark background
    plt.figure(figsize=(10, 10))
    plt.style.use('dark_background')
    
    # Add subtle glow effect
    plt.imshow(np.ones_like(img_np) * 0.2, alpha=0.3)
    
    # Plot the original image
    plt.imshow(img_np)
    
    # Overlay the heatmap
    plt.imshow(colored_heatmap, alpha=0.8)
    
    # Remove axes and create tight layout
    plt.axis('off')
    plt.tight_layout(pad=0)
    
    # Save with high DPI
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    print(f"Multi-layer stylish heatmap saved to {save_path}")
    
    return colored_heatmap

def visualize_all_heads_all_layers(
    all_layers_attention_probs,  # List of attention tensors, one per layer
    original_image,              # PIL Image or tensor
    latent_height,               # Height in latent space
    latent_width,                # Width in latent space
    save_path="all_heads_all_layers.png",
    colormap='plasma',
    brightness_factor=2.0,
    threshold=0.5,
    layer_weights=None,          # Optional weights for layers
    head_weights=None            # Optional weights for heads
):
    """
    Visualizes the combined attention from ALL heads across ALL layers.
    
    Parameters:
    - all_layers_attention_probs: List of attention tensors, one per layer
    - original_image: The original image to overlay attention on
    - latent_height/width: Dimensions in latent space
    - layer_weights: Optional list of weights per layer (higher = more importance)
    - head_weights: Optional list of weights per head (higher = more importance)
    """
    num_layers = len(all_layers_attention_probs)
    
    # Convert original image to numpy if it's a tensor
    if isinstance(original_image, torch.Tensor):
        img_np = original_image.cpu().permute(1, 2, 0).numpy()
        img_np = np.clip(img_np, 0, 1)
    else:
        img_np = np.array(original_image) / 255.0
    
    img_height, img_width = img_np.shape[:2]
    combined_attention = np.zeros((img_height, img_width))
    
    # Process each layer
    for layer_idx, attention_probs in enumerate(all_layers_attention_probs):
        # Reshape attention if needed (batch*heads, seq_len, seq_len) → (batch, heads, seq_len, seq_len)
        if len(attention_probs.shape) == 3:
            num_heads = attention_probs.shape[0]  # Since batch is always 1
            seq_len = attention_probs.shape[1]
            attention_probs = attention_probs.reshape(1, num_heads, seq_len, seq_len)
        
        # Get layer weight
        layer_weight = 1.0
        if layer_weights is not None and layer_idx < len(layer_weights):
            layer_weight = layer_weights[layer_idx]
            
        # Process each head in this layer
        num_heads = attention_probs.shape[1]
        
        for head_idx in range(num_heads):
            # Get head weight
            head_weight = 1.0
            if head_weights is not None and head_idx < len(head_weights):
                head_weight = head_weights[head_idx]
            
            # Extract attention map for this head (batch index is 0)
            attn_map = attention_probs[0, head_idx].cpu().to(torch.float32).detach().numpy()
            
            # Compute average attention received by each token
            avg_attention = attn_map.mean(axis=0)
            
            # Reshape to latent space dimensions
            attention_latent = avg_attention.reshape(latent_height, latent_width)
            
            # Resize to match original image
            attention_resized = torch.nn.functional.interpolate(
                torch.tensor(attention_latent).unsqueeze(0).unsqueeze(0),
                size=(img_height, img_width),
                mode='bicubic'
            ).squeeze().numpy()
            
            # Normalize
            attention_normalized = (attention_resized - attention_resized.min()) / (attention_resized.max() - attention_resized.min() + 1e-8)
            
            # Apply the head and layer weights and add to combined attention
            combined_attention += attention_normalized * head_weight * layer_weight
    
    # Normalize the combined attention
    combined_normalized = (combined_attention - combined_attention.min()) / (combined_attention.max() - combined_attention.min() + 1e-8)
    
    # Apply threshold
    thresholded_attention = np.copy(combined_normalized)
    thresholded_attention[thresholded_attention < threshold] = 0
    
    # Create colored heatmap
    cmap = plt.get_cmap(colormap)
    colored_heatmap = cmap(thresholded_attention)
    colored_heatmap[..., 3] = thresholded_attention * brightness_factor
    
    # Create figure with dark background
    plt.figure(figsize=(10, 10))
    plt.style.use('dark_background')
    
    # Add subtle glow effect
    plt.imshow(np.ones_like(img_np) * 0.2, alpha=0.3)
    
    # Plot the original image
    plt.imshow(img_np)
    
    # Overlay the heatmap
    plt.imshow(colored_heatmap, alpha=0.8)
    
    # Remove axes and create tight layout
    plt.axis('off')
    plt.tight_layout(pad=0)
    
    # Save with high DPI
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    print(f"Combined visualization of all heads across all layers saved to {save_path}")
    
    return colored_heatmap

def create_animated_attention_visualization(
    all_layers_attention_probs,
    original_image,
    latent_height,
    latent_width,
    output_gif="attention_animation.gif",
    fps=2,              # Frames per second
    colormap='plasma',
    brightness_factor=2.0,
    threshold=0.5
):
    """
    Creates an animated GIF showing attention patterns across different layers.
    """
    with TemporaryDirectory() as temp_dir:
        frame_paths = []
        
        # Convert original image to numpy if needed
        if isinstance(original_image, torch.Tensor):
            img_np = original_image.cpu().permute(1, 2, 0).numpy()
            img_np = np.clip(img_np, 0, 1)
        else:
            img_np = np.array(original_image) / 255.0
        
        # Create frames for each layer
        for layer_idx, attention_probs in enumerate(all_layers_attention_probs):
            # Create a frame showing all heads in this layer combined
            frame_path = f"{temp_dir}/layer_{layer_idx:02d}.png"
            
            # Reshape if needed
            if len(attention_probs.shape) == 3:
                num_heads = attention_probs.shape[0]  # Since batch is always 1
                seq_len = attention_probs.shape[1]
                attention_probs = attention_probs.reshape(1, num_heads, seq_len, seq_len)
            
            # Process all heads in this layer
            layer_combined = np.zeros((img_np.shape[0], img_np.shape[1]))
            
            for head_idx in range(attention_probs.shape[1]):
                attn_map = attention_probs[0, head_idx].cpu().detach().numpy()
                avg_attention = attn_map.mean(axis=0)
                attention_latent = avg_attention.reshape(latent_height, latent_width)
                
                # Resize to match original image
                attention_resized = torch.nn.functional.interpolate(
                    torch.tensor(attention_latent).unsqueeze(0).unsqueeze(0),
                    size=(img_np.shape[0], img_np.shape[1]),
                    mode='bicubic'
                ).squeeze().numpy()
                
                # Normalize
                attention_normalized = (attention_resized - attention_resized.min()) / (attention_resized.max() - attention_resized.min() + 1e-8)
                
                # Add to layer combined
                layer_combined += attention_normalized
            
            # Normalize the combined layer attention
            layer_combined = (layer_combined - layer_combined.min()) / (layer_combined.max() - layer_combined.min() + 1e-8)
            
            # Apply threshold
            thresholded = np.copy(layer_combined)
            thresholded[thresholded < threshold] = 0
            
            # Create colored heatmap
            cmap = plt.get_cmap(colormap)
            colored_heatmap = cmap(thresholded)
            colored_heatmap[..., 3] = thresholded * brightness_factor
            
            # Create frame
            plt.figure(figsize=(10, 10))
            plt.style.use('dark_background')
            plt.imshow(np.ones_like(img_np) * 0.2, alpha=0.3)
            plt.imshow(img_np)
            plt.imshow(colored_heatmap, alpha=0.8)
            plt.title(f"Layer {layer_idx}", color='white', fontsize=20)
            plt.axis('off')
            plt.tight_layout(pad=0)
            plt.savefig(frame_path, dpi=100, bbox_inches='tight', pad_inches=0)
            plt.close()
            
            frame_paths.append(frame_path)
        
        # # Create GIF from frames
        # with imageio.get_writer(output_gif, mode='I', fps=fps) as writer:
        #     for frame_path in frame_paths:
        #         image = imageio.imread(frame_path)
        #         writer.append_data(image)
        
        # print(f"Animated attention visualization saved to {output_gif}")

