"""
Run evaluation for a given checkpoint:
- For each source style (one image per style in test_image_dir), transfer to all target styles
- Compute LPIPS and VGG perceptual distances:
    - content_lpips: LPIPS(generated, source)
    - style_lpips: LPIPS(generated, target_ref)
    - content_vgg: L2 distance between VGG features (generated vs source)
    - style_vgg: L2 distance between VGG features (generated vs target_ref)
- Save per-transfer metrics to CSV and per-experiment summary

Usage:
    python run_evaluation.py --checkpoint checkpoints/baseline/latest.pt --config experiments/baseline/config.json --output experiments/baseline/eval

"""

import argparse
import json
from pathlib import Path
import torch
import numpy as np
from PIL import Image
import csv
import os

# LPIPS
try:
    import lpips
except Exception:
    lpips = None

import torchvision.transforms as T
import torchvision.models as models
import torch.nn.functional as F

from inference import LGTInference, load_vae, encode_image, decode_latent, tensor_to_pil
import base64
from io import BytesIO
from PIL import Image as PILImage


# VGG feature extractor
class VGGFeatureExtractor(torch.nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        vgg = models.vgg16(pretrained=True).features.eval().to(device)
        for p in vgg.parameters():
            p.requires_grad = False
        self.device = device
        # We'll use features at relu2_2 (index ~8) and relu3_3 (~15)
        self.vgg = vgg
        self.layer_ids = [8, 15]
    
    def forward(self, x):
        feats = []
        h = x
        for i, layer in enumerate(self.vgg):
            h = layer(h)
            if i in self.layer_ids:
                feats.append(h)
        return feats


def load_image_to_tensor(path, size=256, device='cuda'):
    img = Image.open(path).convert('RGB')
    img = img.resize((size, size))
    t = T.ToTensor()(img).unsqueeze(0).to(device)  # [0,1]
    return t


def to_lpips_input(img_tensor):
    # LPIPS expects [-1,1]
    return img_tensor * 2.0 - 1.0


def compute_vgg_distance(vgg, img1, img2):
    # img in [0,1]
    # Normalize by ImageNet mean/std
    mean = torch.tensor([0.485, 0.456, 0.406], device=img1.device).view(1,3,1,1)
    std = torch.tensor([0.229, 0.224, 0.225], device=img1.device).view(1,3,1,1)
    x1 = (img1 - mean) / std
    x2 = (img2 - mean) / std
    feats1 = vgg(x1)
    feats2 = vgg(x2)
    dists = []
    for f1, f2 in zip(feats1, feats2):
        # L2 distance averaged over spatial and channels
        d = F.mse_loss(f1, f2, reduction='mean')
        dists.append(d.item())
    return np.mean(dists)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--config', type=str, required=False, default=None)
    parser.add_argument('--test_dir', type=str, default=None)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_steps', type=int, default=20)
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'

    # Load checkpoint and config
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # If config provided, use it; else load config embedded in checkpoint
    if args.config:
        with open(args.config, 'r') as f:
            cfg = json.load(f)
    else:
        # try to load from checkpoint
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        cfg = ckpt.get('config', None)
        if cfg is None:
            raise RuntimeError("Config not provided and not found in checkpoint")

    test_dir = Path(args.test_dir) if args.test_dir else Path(cfg['training'].get('test_image_dir', '/mnt/f/monet2photo/test'))
    style_subdirs = cfg['data'].get('style_subdirs', [])

    # Prepare output
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load inference engine
    lgt = LGTInference(str(checkpoint_path), device=device, num_steps=args.num_steps)

    # Load VAE for encoding/decoding
    vae = load_vae(device)

    # Prepare LPIPS
    if lpips is None:
        print("Warning: lpips package not installed. Install with 'pip install lpips' for LPIPS metric.")
    else:
        loss_fn = lpips.LPIPS(net='vgg').to(device)

    # VGG extractor
    vgg = VGGFeatureExtractor(device=device)

    # CLIP model for semantic similarity (image and text)
    try:
        from transformers import CLIPModel, CLIPProcessor
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        clip_model.eval()
        has_clip = True
    except Exception as e:
        print(f"Warning: Failed to load CLIP model: {e}. CLIP metrics will be skipped.")
        clip_model = None
        clip_processor = None
        has_clip = False

    # Load style prompts from config (optional)
    style_prompts = cfg.get('data', {}).get('style_prompts', None)
    if style_prompts is not None and len(style_prompts) != len(style_subdirs):
        print("Warning: 'style_prompts' length does not match 'style_subdirs'. Ignoring prompts.")
        style_prompts = None

    # Gather test images: use ALL files in each style directory
    test_images = {}
    for style_id, style_name in enumerate(style_subdirs):
        style_dir = test_dir / style_name
        if not style_dir.exists():
            print(f"Warning: style test dir not found: {style_dir}")
            continue
        images = sorted([p for p in style_dir.iterdir() if p.suffix.lower() in ['.jpg', '.png', '.jpeg', '.bmp', '.webp']])
        if not images:
            print(f"Warning: no images in {style_dir}")
            continue
        # store list of images for this style
        test_images[style_id] = (style_name, images)

    if not test_images:
        raise RuntimeError("No test images found. Check test_image_dir and style subdirs.")

    results = []

    # For each source style and each source image
    for src_id, (src_name, src_list) in test_images.items():
        for src_path in src_list:
            print(f"Source: {src_name} -> {src_path.name}")
            img_src = load_image_to_tensor(src_path, size=256, device=device)  # [1,3,H,W]
            latent_src = encode_image(vae, img_src, device)
            # Ensure dtype compatibility: model parameters are float32; latent from VAE may be float16
            latent_src = latent_src.to(device).to(torch.float32)

            # Inversion once per source image
            try:
                latent_x0 = lgt.inversion(latent_src, src_id, num_steps=15)
            except Exception as e:
                print(f"Error during inversion for {src_name} image {src_path.name}: {e}")
                continue

            # For each target style: generate ONCE, then evaluate against ALL references
            for tgt_id, (tgt_name, tgt_list) in test_images.items():
                if tgt_id == src_id:
                    continue
                
                print(f"  -> Generating for target style: {tgt_name}")
                
                # Generation: only once per (src_image, tgt_style_id)
                try:
                    latent_tgt = lgt.generation(latent_x0, tgt_id, num_steps=15)
                except Exception as e:
                    print(f"Error during generation for {src_name}/{src_path.name} -> {tgt_name}: {e}")
                    continue

                # Decode generated image
                img_gen = decode_latent(vae, latent_tgt, device)  # [B,3,H,W] in [0,1]
                
                # Save generated image (once per src/tgt_style pair)
                out_img_name = f"{src_name}_{src_path.stem}_to_{tgt_name}.jpg"
                out_img_path = out_dir / out_img_name
                from torchvision.utils import save_image
                save_image(img_gen, out_img_path)
                
                # Now evaluate against ALL reference images of this target style
                for tgt_path in tgt_list:
                    print(f"    Evaluating against ref: {tgt_path.name}")

                    # Load target reference image (for metric computation only, NOT as model input)
                    img_tgt_ref = load_image_to_tensor(tgt_path, size=256, device=device)

                    # Compute LPIPS
                    content_lpips = None
                    style_lpips = None
                    if lpips is not None:
                        a = to_lpips_input(img_gen)
                        b_src = to_lpips_input(img_src)
                        b_tgt = to_lpips_input(img_tgt_ref)
                        with torch.no_grad():
                            content_lpips = float(loss_fn(a, b_src).item())
                            style_lpips = float(loss_fn(a, b_tgt).item())

                    # Compute VGG distances
                    content_vgg = compute_vgg_distance(vgg, img_gen, img_src)
                    style_vgg = compute_vgg_distance(vgg, img_gen, img_tgt_ref)

                    # Compute CLIP similarity (cosine on image embeddings)
                    clip_content = None
                    clip_style = None
                    clip_text = None
                    if has_clip:
                        # Convert tensors to PIL for the processor
                        from torchvision.transforms import ToPILImage
                        to_pil = ToPILImage()

                        pil_gen = to_pil(img_gen.squeeze(0).cpu())
                        pil_src = to_pil(img_src.squeeze(0).cpu())
                        pil_tgt = to_pil(img_tgt_ref.squeeze(0).cpu())

                        inputs_gen = clip_processor(images=pil_gen, return_tensors='pt').to(device)
                        inputs_src = clip_processor(images=pil_src, return_tensors='pt').to(device)
                        inputs_tgt = clip_processor(images=pil_tgt, return_tensors='pt').to(device)

                        with torch.no_grad():
                            emb_gen = clip_model.get_image_features(**inputs_gen)
                            emb_src = clip_model.get_image_features(**inputs_src)
                            emb_tgt = clip_model.get_image_features(**inputs_tgt)

                            emb_gen = emb_gen / emb_gen.norm(p=2, dim=-1, keepdim=True)
                            emb_src = emb_src / emb_src.norm(p=2, dim=-1, keepdim=True)
                            emb_tgt = emb_tgt / emb_tgt.norm(p=2, dim=-1, keepdim=True)

                            clip_content = float(torch.nn.functional.cosine_similarity(emb_gen, emb_src).item())
                            clip_style = float(torch.nn.functional.cosine_similarity(emb_gen, emb_tgt).item())

                        # If style text prompts provided, compute CLIP image-text similarity
                        if style_prompts is not None:
                            prompt = style_prompts[tgt_id]
                            try:
                                text_inputs = clip_processor(text=[prompt], return_tensors='pt', padding=True).to(device)
                                with torch.no_grad():
                                    text_emb = clip_model.get_text_features(**text_inputs)
                                    text_emb = text_emb / text_emb.norm(p=2, dim=-1, keepdim=True)
                                    clip_text = float(torch.nn.functional.cosine_similarity(emb_gen, text_emb).item())
                            except Exception as e:
                                print(f"Warning: CLIP text embedding failed for prompt '{prompt}': {e}")
                                clip_text = None

                    entry = {
                        'src_style': src_name,
                        'tgt_style': tgt_name,
                        'src_image': str(src_path),
                        'tgt_ref_image': str(tgt_path),
                        'gen_image': str(out_img_path),
                        'content_lpips': content_lpips,
                        'style_lpips': style_lpips,
                        'content_vgg': content_vgg,
                        'style_vgg': style_vgg,
                        'clip_content': clip_content,
                        'clip_style': clip_style,
                        'clip_text': clip_text
                    }
                    results.append(entry)

    # Write CSV
    csv_path = out_dir / 'metrics.csv'
    keys = ['src_style','tgt_style','src_image','tgt_ref_image','gen_image','content_lpips','style_lpips','content_vgg','style_vgg','clip_content','clip_style','clip_text']
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    # Summary
    def mean_ignore_none(arr):
        vals = [x for x in arr if x is not None]
        return float(np.mean(vals)) if vals else None

    content_lpips_mean = mean_ignore_none([r['content_lpips'] for r in results])
    style_lpips_mean = mean_ignore_none([r['style_lpips'] for r in results])
    content_vgg_mean = mean_ignore_none([r['content_vgg'] for r in results])
    style_vgg_mean = mean_ignore_none([r['style_vgg'] for r in results])
    clip_content_mean = mean_ignore_none([r.get('clip_content') for r in results])
    clip_style_mean = mean_ignore_none([r.get('clip_style') for r in results])

    clip_text_mean = mean_ignore_none([r.get('clip_text') for r in results])

    summary = {
        'n_pairs': len(results),
        'content_lpips_mean': content_lpips_mean,
        'style_lpips_mean': style_lpips_mean,
        'content_vgg_mean': content_vgg_mean,
        'style_vgg_mean': style_vgg_mean,
        'clip_content_mean': clip_content_mean,
        'clip_style_mean': clip_style_mean,
        'clip_text_mean': clip_text_mean
    }
    with open(out_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print('\nEvaluation finished. Summary:')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
