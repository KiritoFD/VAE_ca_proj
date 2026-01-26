import torch
import torch.nn.functional as F
import numpy as np
import cv2
import argparse
import sys
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import your LGT model modules
from src.inference import LGTInference, load_vae, decode_latent 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--latents", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--style_id", type=int, default=2) # 2 = VanGogh
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--smooth_sigma", type=float, default=1.0)
    parser.add_argument("--tile_overlap", type=int, default=0, help="Overlap pixels between tiles (default 16 = 50%)")
    parser.add_argument("--num_steps_inv", type=int, default=10, help="Inversion steps")
    parser.add_argument("--num_steps_gen", type=int, default=20, help="Generation steps (increase for stronger style)")
    parser.add_argument("--blend_type", type=str, default="gaussian", choices=["linear", "gaussian", "cosine"], help="Blend mask type for tile overlaps")
    parser.add_argument("--vae_tiling", action="store_true", help="Enable VAE decoder tiling (reduce checkerboard artifacts)")
    parser.add_argument("--blend_debug", action="store_true", help="Print blend mask statistics")
    args = parser.parse_args()
    
    device = 'cuda'
    
    # 1. Load Data (Memory Mapped if huge, but 4070 RAM is fine)
    data = np.load(args.latents)
    latents_raw = data['latents']
    fps = data['fps']
    
    # Convert to float32 for scipy compatibility (gaussian_filter1d doesn't support float16)
    latents_raw = latents_raw.astype(np.float32)
    
    # 2. Temporal Smoothing (The Anti-Flicker Magic)
    # Applied on CPU, very fast. Filters high-freq sensor noise in Z-space.
    print(f"Applying Temporal Smoothing (sigma={args.smooth_sigma})...")
    latents_smooth = gaussian_filter1d(latents_raw, sigma=args.smooth_sigma, axis=0)
    latents_tensor = torch.from_numpy(latents_smooth).to(device)
    
    # 3. Load Model
    lgt = LGTInference(args.checkpoint, device=device, num_steps=4) # Fast ODE
    vae = load_vae(device)
    
    # 4. Spatial Tiling Configuration
    TILE_SIZE = 32  # Training size
    TILE_OVERLAP = args.tile_overlap  # Overlap in pixels (50% = 16 for 32x32)
    TILE_STEP = TILE_SIZE - TILE_OVERLAP  # Stride
    
    def create_blend_mask(tile_size, overlap=16, mask_type="gaussian", device='cuda'):
        """
        Create smooth blending mask for tile edges.
        
        Args:
            tile_size: size of tile
            overlap: overlap pixels on each side
            mask_type: 'linear', 'gaussian', or 'cosine'
            device: torch device
            
        Returns:
            mask: [1, 1, tile_size, tile_size]
        """
        mask = torch.ones((tile_size, tile_size), device=device, dtype=torch.float32)
        
        if mask_type == "linear":
            # Linear fade: [0...1...0]
            for i in range(overlap):
                alpha = (i + 1) / (overlap + 1)
                mask[i, :] *= alpha
                mask[-(i+1), :] *= alpha
                mask[:, i] *= alpha
                mask[:, -(i+1)] *= alpha
        
        elif mask_type == "gaussian":
            # Gaussian fade: smoother bell curve
            # Create 1D gaussian
            x = torch.linspace(-3, 3, tile_size, device=device)
            gauss_1d = torch.exp(-0.5 * x**2)
            gauss_1d = gauss_1d / gauss_1d.max()  # Normalize to [0, 1]
            
            # Apply 2D gaussian
            gauss_2d = gauss_1d.unsqueeze(1) * gauss_1d.unsqueeze(0)
            # Scale to emphasize center
            gauss_2d = (1 - gauss_2d) * 0.5 + 0.5  # Map to [0.5, 1]
            mask = mask * gauss_2d
        
        elif mask_type == "cosine":
            # Cosine window: smooth and symmetric
            for i in range(overlap):
                # Cosine fade from 0 to 1
                alpha = (1 - torch.cos(torch.tensor(3.14159 * (i + 1) / (overlap + 1), device=device))) / 2
                mask[i, :] *= alpha
                mask[-(i+1), :] *= alpha
                mask[:, i] *= alpha
                mask[:, -(i+1)] *= alpha
        
        return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, tile_size, tile_size]
    
    def process_with_tiling(latent_batch, lgt_model, style_id, tile_size=32, tile_overlap=16, blend_type="gaussian", num_steps_inv=10, num_steps_gen=4):
        """
        Process latent using spatial tiling with overlap and blending.
        
        Args:
            latent_batch: [B, 4, H, W] latent tensor
            lgt_model: LGT inference model
            style_id: target style ID
            tile_size: size of each tile (must match model training size)
            tile_overlap: overlap pixels between tiles
            blend_type: 'linear', 'gaussian', or 'cosine' blending
            num_steps_inv: inversion steps
            num_steps_gen: generation steps
        
        Returns:
            output: [B, 4, H, W] processed latent
        """
        B, C, H, W = latent_batch.shape
        device = latent_batch.device
        tile_step = tile_size - tile_overlap
        
        # If input is already <= tile_size, no tiling needed
        if H <= tile_size and W <= tile_size:
            print(f"  Input {H}×{W} <= {tile_size}, processing directly without tiling")
            with torch.no_grad():
                z_T = lgt_model.inversion(latent_batch, source_style_id=1, num_steps=num_steps_inv)
                output = lgt_model.generation(z_T, target_style_id=style_id, num_steps=num_steps_gen)
            return output
        
        print(f"  Tiling {H}×{W} with {tile_size}×{tile_size} tiles (stride {tile_step}, overlap {tile_overlap}px, mask={blend_type})")
        
        # Output accumulator and weight map
        output = torch.zeros_like(latent_batch)
        weight_map = torch.zeros((B, 1, H, W), device=device, dtype=torch.float32)
        
        # Create blend mask with proper overlap
        blend_mask = create_blend_mask(tile_size, overlap=tile_overlap, mask_type=blend_type, device=device)
        
        if args.blend_debug:
            print(f"  [DEBUG] Blend mask - min: {blend_mask.min():.4f}, max: {blend_mask.max():.4f}, mean: {blend_mask.mean():.4f}")
        
        # Generate tile positions with overlap
        y_positions = list(range(0, max(1, H - tile_size + 1), tile_step))
        x_positions = list(range(0, max(1, W - tile_size + 1), tile_step))
        
        # Ensure we cover the entire image - add final position if needed
        if H > tile_size and (y_positions[-1] + tile_size) < H:
            y_positions.append(H - tile_size)
        if W > tile_size and (x_positions[-1] + tile_size) < W:
            x_positions.append(W - tile_size)
        
        total_tiles = len(y_positions) * len(x_positions)
        print(f"  Processing {total_tiles} tiles ({len(y_positions)}×{len(x_positions)})...")
        
        tile_idx = 0
        # Spatial loop over tiles
        for y_start in y_positions:
            for x_start in x_positions:
                y_end = min(y_start + tile_size, H)
                x_end = min(x_start + tile_size, W)
                
                # Extract tile (may be smaller at edges)
                tile = latent_batch[:, :, y_start:y_end, x_start:x_end]
                tile_h = y_end - y_start
                tile_w = x_end - x_start
                
                # Pad tile to tile_size if at edges
                if tile_h < tile_size or tile_w < tile_size:
                    pad_h = tile_size - tile_h
                    pad_w = tile_size - tile_w
                    tile_padded = F.pad(tile, (0, pad_w, 0, pad_h), mode='reflect')
                else:
                    tile_padded = tile
                
                # Process tile: inversion → generation
                with torch.no_grad():
                    z_T = lgt_model.inversion(tile_padded, source_style_id=1, num_steps=num_steps_inv)
                    z_0 = lgt_model.generation(z_T, target_style_id=style_id, num_steps=num_steps_gen)
                
                # Remove padding from output
                z_0 = z_0[:, :, :tile_h, :tile_w]
                
                # Get corresponding mask (cropped if at edges)
                mask_crop = blend_mask[:, :, :tile_h, :tile_w]
                
                # Apply blend mask and accumulate (weighted averaging in overlap regions)
                output[:, :, y_start:y_end, x_start:x_end] += z_0 * mask_crop
                weight_map[:, :, y_start:y_end, x_start:x_end] += mask_crop
                
                tile_idx += 1
                if tile_idx % max(1, total_tiles // 4) == 0:
                    print(f"    Processed {tile_idx}/{total_tiles} tiles")
        
        # Normalize by weight map (handles overlap regions via weighted averaging)
        output = output / (weight_map + 1e-8)
        
        print(f"  ✓ Tiling complete")
        return output
    
    # Check if tiling is needed
    need_tiling = latents_tensor.shape[2] > TILE_SIZE or latents_tensor.shape[3] > TILE_SIZE
    
    if need_tiling:
        print(f"⚠️  Input size {latents_tensor.shape[2]}×{latents_tensor.shape[3]} > {TILE_SIZE}×{TILE_SIZE}")
        print(f"   Enabling spatial tiling with overlap blending")
    
    chunk_size = 1 if need_tiling else 16  # Process one frame at a time if tiling
    
    print("Running LGT ODE...")
    print(f"Input latent shape: {latents_tensor.shape}")  # Debug: check input shape
    
    # Don't force resize - let tiling handle any size
    # If smaller than 32×32, tiling will just process it directly
    
    with torch.no_grad():
        # Process with tiling if needed
        target_latents_list = []
        
        for i in tqdm(range(0, len(latents_tensor), chunk_size), desc="Stylizing"):
            batch = latents_tensor[i : i + chunk_size]
            
            if need_tiling:
                # Use spatial tiling for large latents
                styled_batch = process_with_tiling(
                    batch,
                    lgt,
                    args.style_id,
                    tile_size=TILE_SIZE,
                    tile_overlap=TILE_OVERLAP,
                    blend_type=args.blend_type,
                    num_steps_inv=args.num_steps_inv,
                    num_steps_gen=args.num_steps_gen
                )
            else:
                # Direct processing for small latents
                z_T = lgt.inversion(batch, source_style_id=1, num_steps=args.num_steps_inv)
                styled_batch = lgt.generation(z_T, target_style_id=args.style_id, num_steps=args.num_steps_gen)
            
            target_latents_list.append(styled_batch)
        
        target_latents = torch.cat(target_latents_list)    
    print(f"\nDecoding {len(target_latents)} frames to video...")
    
    # Ensure fps is a Python float (not numpy type)
    fps = float(fps) if hasattr(fps, 'item') else float(fps)
    
    print(f"Video fps: {fps}")
    
    # Decode first batch to get actual image dimensions
    print("Detecting actual decoded image resolution...")
    with torch.no_grad():
        first_batch = target_latents[0:1]
        first_imgs = decode_latent(vae, first_batch, device)
        first_imgs_np = first_imgs.permute(0, 2, 3, 1).cpu().numpy()
        actual_height, actual_width = first_imgs_np.shape[1:3]
    
    print(f"Actual decoded image dimensions: ({actual_width}, {actual_height})")
    
    # Try multiple codecs in case mp4v is not available
    codecs = ['mp4v', 'H264', 'XVID', 'MJPG']
    writer = None
    
    for codec in codecs:
        try:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            test_writer = cv2.VideoWriter(args.output, fourcc, fps, (actual_width, actual_height))
            
            if test_writer.isOpened():
                writer = test_writer
                print(f"✓ VideoWriter initialized successfully with codec: {codec}")
                break
            else:
                test_writer.release()
                print(f"✗ Codec {codec} failed to open writer")
        except Exception as e:
            print(f"✗ Codec {codec} error: {e}")
            continue
    
    if writer is None:
        print("ERROR: Failed to open VideoWriter with any codec!")
        print("Trying fallback: writing frames as PNGs instead...")
        
        # Fallback: save as PNG sequence
        output_dir = Path(args.output).parent / (Path(args.output).stem + "_frames")
        output_dir.mkdir(exist_ok=True)
        
        with torch.no_grad():
            for frame_idx, i in enumerate(tqdm(range(0, len(target_latents), 8), desc="Decoding")):
                batch = target_latents[i : i+8]
                imgs = decode_latent(vae, batch, device) # [B, 3, H, W]
                
                # Tensor to uint8 numpy
                imgs = imgs.permute(0, 2, 3, 1).cpu().numpy()
                imgs = np.clip(imgs * 255, 0, 255).astype(np.uint8)
                
                for batch_idx, img in enumerate(imgs):
                    frame_path = output_dir / f"frame_{frame_idx * 8 + batch_idx:06d}.png"
                    cv2.imwrite(str(frame_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        
        print(f"Frames saved to: {output_dir}")
        return
    
    # Normal video writing path
    # Write first batch
    first_imgs_uint8 = np.clip(first_imgs_np * 255, 0, 255).astype(np.uint8)
    
    print("Writing frames to video...")
    for img in first_imgs_uint8:
        success = writer.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        if not success:
            print("WARNING: Failed to write frame!")
    
    # Write remaining batches
    with torch.no_grad():
        for i in tqdm(range(8, len(target_latents), 8), desc="Saving"):
            batch = target_latents[i : i+8]
            imgs = decode_latent(vae, batch, device) # [B, 3, H, W]
            
            # Tensor to uint8 numpy
            imgs = imgs.permute(0, 2, 3, 1).cpu().numpy()
            imgs = np.clip(imgs * 255, 0, 255).astype(np.uint8)
            
            for img in imgs:
                success = writer.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                if not success:
                    print("WARNING: Failed to write frame!")
                
    writer.release()
    
    # Print summary
    try:
        output_size = Path(args.output).stat().st_size / (1024*1024)  # MB
        print(f"\n✓ Video saved successfully: {args.output}")
        print(f"  File size: {output_size:.1f} MB")
        print(f"  Resolution: {actual_width}×{actual_height}")
        print(f"  Frames: {len(target_latents)}")
        print(f"  FPS: {fps}")
    except Exception as e:
        print(f"✓ Video saved: {args.output}")
        
    print("\n💡 Tip: If you want higher resolution output (e.g., 512×512 → 1080p):")
    print("   Install Real-ESRGAN: pip install realesrgan")
    print("   Re-run the script to enable automatic 2x upscaling")

if __name__ == "__main__":
    main()