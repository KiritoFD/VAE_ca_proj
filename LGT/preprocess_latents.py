"""
Data Preprocessing: Pre-encode images to VAE latents

Critical: NEVER run VAE encoder in training loop!
All latents must be pre-computed and loaded from disk.
"""

import os
import json
import torch
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm


def load_vae():
    """Load Stable Diffusion VAE model."""
    try:
        from diffusers import AutoencoderKL
        print("Loading VAE from stabilityai/sd-vae-ft-mse...")
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
        return vae
    except ImportError:
        print("Error: diffusers not installed. Run: pip install diffusers")
        return None
    except Exception as e:
        print(f"Error loading VAE: {e}")
        return None


def preprocess_image(image_path, target_size=256):
    """
    Preprocess single image.
    
    Args:
        image_path: Path to image
        target_size: Target size (square)
    
    Returns:
        image_tensor: [3, H, W] in range [-1, 1]
    """
    try:
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Resize (maintain aspect ratio)
        w, h = image.size
        if w < h:
            new_w = target_size
            new_h = int(h * target_size / w)
        else:
            new_h = target_size
            new_w = int(w * target_size / h)
        
        image = image.resize((new_w, new_h), Image.LANCZOS)
        
        # Center crop to square
        left = (new_w - target_size) // 2
        top = (new_h - target_size) // 2
        image = image.crop((left, top, left + target_size, top + target_size))
        
        # Convert to tensor: [0, 255] → [0, 1] → [-1, 1]
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)  # [H, W, 3] → [3, H, W]
        image = image * 2.0 - 1.0
        
        return image
    
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None


def encode_to_latent(vae, image_tensor, device):
    """
    Encode image to VAE latent.
    
    Args:
        vae: VAE model
        image_tensor: [3, H, W] in range [-1, 1]
        device: torch device
    
    Returns:
        latent: [4, H//8, W//8]
    """
    vae = vae.to(device)
    vae.eval()
    
    with torch.no_grad():
        # Add batch dimension
        image_batch = image_tensor.unsqueeze(0).to(device)  # [1, 3, H, W]
        
        # Encode
        latent_dist = vae.encode(image_batch).latent_dist
        latent = latent_dist.sample()  # [1, 4, H//8, W//8]
        
        # Scale (Stable Diffusion standard: 0.18215)
        latent = latent * vae.config.scaling_factor
        
        # Remove batch dimension and move to CPU
        latent = latent.squeeze(0).cpu()  # [4, H//8, W//8]
    
    return latent


def process_style_folder(vae, style_dir, output_dir, style_name, device, target_size=256):
    """
    Process all images in a style folder.
    
    Args:
        vae: VAE model
        style_dir: Source directory with images
        output_dir: Output directory for latents
        style_name: Name for output subdirectory
        device: torch device
        target_size: Target image size
    """
    style_dir = Path(style_dir)
    output_dir = Path(output_dir) / style_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all images
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(style_dir.glob(f"*{ext}"))
        image_files.extend(style_dir.glob(f"*{ext.upper()}"))
    
    if len(image_files) == 0:
        print(f"⚠ No images found in {style_dir}")
        return
    
    print(f"\nProcessing {style_name}: {style_dir}")
    print(f"Found {len(image_files)} images")
    
    # Process each image
    success_count = 0
    for img_path in tqdm(image_files, desc=style_name):
        try:
            # Preprocess
            image_tensor = preprocess_image(img_path, target_size)
            if image_tensor is None:
                continue
            
            # Encode to latent
            latent = encode_to_latent(vae, image_tensor, device)
            
            # Save latent
            output_path = output_dir / f"{img_path.stem}.pt"
            torch.save(latent, output_path)
            
            success_count += 1
        
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue
    
    print(f"✓ {style_name}: {success_count}/{len(image_files)} images encoded")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocess images to VAE latents')
    parser.add_argument('--config', type=str, default='config.json', 
                        help='Config file path')
    parser.add_argument('--raw_data_root', type=str, required=True,
                        help='Root directory with style folders')
    parser.add_argument('--output_root', type=str, default='data/latents',
                        help='Output directory for latents')
    parser.add_argument('--target_size', type=int, default=256, 
                        help='Target image size')
    parser.add_argument('--device', type=str, default=None, 
                        help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Load config if exists
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
        style_subdirs = config['data']['style_subdirs']
    else:
        # Default: assume style0, style1, ...
        style_subdirs = None
    
    raw_data_root = Path(args.raw_data_root)
    output_root = Path(args.output_root)
    
    print("=" * 80)
    print("LGT VAE Latent Preprocessing")
    print("=" * 80)
    print(f"Raw data root: {raw_data_root}")
    print(f"Output root: {output_root}")
    print(f"Target size: {args.target_size}")
    print("=" * 80)
    
    # Set device
    if args.device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"\nUsing device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load VAE
    vae = load_vae()
    if vae is None:
        print("Failed to load VAE. Exiting.")
        return
    
    print("✓ VAE loaded successfully")
    
    # Create output directory
    output_root.mkdir(parents=True, exist_ok=True)
    
    # Process each style folder
    if style_subdirs is None:
        # Auto-detect style folders
        style_subdirs = [d.name for d in raw_data_root.iterdir() if d.is_dir()]
        print(f"\nAuto-detected style folders: {style_subdirs}")
    
    for style_name in style_subdirs:
        style_dir = raw_data_root / style_name
        
        if not style_dir.exists():
            print(f"⚠ Warning: Style directory not found: {style_dir}")
            continue
        
        process_style_folder(
            vae=vae,
            style_dir=style_dir,
            output_dir=output_root,
            style_name=style_name,
            device=device,
            target_size=args.target_size
        )
    
    print("\n" + "=" * 80)
    print("✓ Preprocessing completed!")
    print(f"Latents saved to: {output_root}")
    print("=" * 80)
    
    # Show statistics
    print("\nDataset statistics:")
    for style_name in style_subdirs:
        style_dir = output_root / style_name
        if style_dir.exists():
            num_files = len(list(style_dir.glob("*.pt")))
            print(f"  {style_name}: {num_files} samples")
    
    # Update config if needed
    if os.path.exists(args.config):
        print(f"\nReminder: Update '{args.config}' with:")
        print(f'  "data_root": "{output_root}"')
        print(f'  "style_subdirs": {style_subdirs}')
        print(f'  "num_styles": {len(style_subdirs)}')


if __name__ == "__main__":
    main()
