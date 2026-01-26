"""
LGT Evaluation Pro: Double Persistence, Zero-Overhead Resume & Structured Analytics

Optimizations:
1. Result Persistence: Checks metrics.csv. Skips processed pairs instantly.
2. Feature Persistence: Caches extracted VGG/CLIP features to 'ref_features.pt'.
3. Structured Reporting: Generates a hierarchical JSON (Matrix, Transfer vs Identity).

Target Hardware: RTX 4070 Laptop (8GB VRAM)
"""

import argparse
import json
from pathlib import Path
import torch
import numpy as np
import csv
import os
import random
from tqdm import tqdm
import time
from collections import defaultdict

# LPIPS
try:
    import lpips
except Exception:
    lpips = None

import torchvision.transforms as T
import torchvision.models as models
import torch.nn.functional as F
from PIL import Image

# Import your modules
from inference import LGTInference, load_vae, encode_image, decode_latent
from torchvision.transforms import ToPILImage

# ==========================================
# Optimized Feature Extractors
# ==========================================

class VGGFeatureExtractor(torch.nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        vgg = models.vgg16(pretrained=True).features.eval().to(device)
        for p in vgg.parameters():
            p.requires_grad = False
        self.device = device
        self.vgg = vgg
        self.layer_ids = [8, 15] 
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406], device=device).view(1,3,1,1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225], device=device).view(1,3,1,1))

    def get_features(self, x):
        if x.device != self.mean.device:
            x = x.to(self.mean.device)
        x = (x - self.mean) / self.std
        feats = []
        h = x
        for i, layer in enumerate(self.vgg):
            h = layer(h)
            if i in self.layer_ids:
                feats.append(h.detach().cpu())
        return feats

def compute_distance_cpu_gpu_hybrid(feats_gen_gpu, feats_ref_cpu, device):
    dists = []
    for f_gen, f_ref in zip(feats_gen_gpu, feats_ref_cpu):
        f_ref_gpu = f_ref.to(device)
        d = F.mse_loss(f_gen, f_ref_gpu, reduction='mean')
        dists.append(d.item())
    return np.mean(dists)

def load_image_to_tensor(path, size=256, device='cuda'):
    try:
        img = Image.open(path).convert('RGB')
        img = img.resize((size, size))
        t = T.ToTensor()(img).unsqueeze(0).to(device)
        return t
    except Exception:
        return None

def to_lpips_input(img_tensor):
    return img_tensor * 2.0 - 1.0

# ==========================================
# Main Logic
# ==========================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--config', type=str, required=False, default=None)
    parser.add_argument('--test_dir', type=str, default=None)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_steps', type=int, default=15)
    parser.add_argument('--force_regen', action='store_true', help='Force regeneration')
    parser.add_argument('--max_eval_samples', type=int, default=50)
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'
    
    # Setup Output
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / 'metrics.csv'
    cache_path = out_dir / 'ref_features_cache.pt'
    
    processed_pairs = set()
    if csv_path.exists() and not args.force_regen:
        print("Found metrics.csv, scanning processed pairs...")
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (Path(row['src_image']).name, row['tgt_style'])
                processed_pairs.add(key)
        print(f"✓ Resuming: {len(processed_pairs)} pairs already done.")

    # Config & Data
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists(): raise FileNotFoundError(checkpoint_path)

    if args.config:
        with open(args.config, 'r') as f: cfg = json.load(f)
    else:
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        cfg = ckpt.get('config', None)

    test_dir = Path(args.test_dir) if args.test_dir else Path(cfg['training'].get('test_image_dir', ''))
    style_subdirs = cfg['data'].get('style_subdirs', [])
    
    test_images = {}
    for style_id, style_name in enumerate(style_subdirs):
        style_dir = test_dir / style_name
        if not style_dir.exists(): continue
        images = sorted([p for p in style_dir.iterdir() if p.suffix.lower() in ['.jpg', '.png', '.jpeg']])
        test_images[style_id] = (style_name, images)

    # Load Models
    print("Loading Models...")
    lgt = LGTInference(str(checkpoint_path), device=device, num_steps=args.num_steps)
    vae = load_vae(device)
    vgg_extractor = VGGFeatureExtractor(device=device)
    loss_fn = lpips.LPIPS(net='vgg', verbose=False).to(device) if lpips else None

    try:
        from transformers import CLIPModel, CLIPProcessor
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        clip_model.eval()
        has_clip = True
        to_pil = ToPILImage()
    except:
        has_clip = False
        print("Warning: CLIP not found.")

    # Feature Cache
    ref_features = {} 
    cache_loaded = False
    if cache_path.exists() and not args.force_regen:
        try:
            ref_features = torch.load(cache_path, map_location='cpu')
            cache_loaded = True
            print(f"✓ Feature cache loaded.")
        except: pass
    
    if not cache_loaded:
        print("\nPre-computing Reference Features...")
        for style_id, (style_name, img_list) in test_images.items():
            eval_list = img_list
            if args.max_eval_samples > 0 and len(eval_list) > args.max_eval_samples:
                 random.seed(42)
                 eval_list = random.sample(img_list, args.max_eval_samples)
            
            ref_features[style_id] = []
            for img_path in tqdm(eval_list, desc=f"Caching {style_name}"):
                img_t = load_image_to_tensor(img_path, size=256, device=device)
                if img_t is None: continue
                with torch.no_grad():
                    v_feats = vgg_extractor.get_features(img_t)
                    c_emb = None
                    if has_clip:
                        pil_img = to_pil(img_t.squeeze(0).cpu())
                        inputs = clip_processor(images=pil_img, return_tensors='pt').to(device)
                        c_emb = clip_model.get_image_features(**inputs)
                        c_emb = (c_emb / c_emb.norm(p=2, dim=-1, keepdim=True)).cpu()
                ref_features[style_id].append({'path': str(img_path), 'vgg': v_feats, 'clip': c_emb})
        torch.save(ref_features, cache_path)

    # Eval Loop
    file_mode = 'a' if (csv_path.exists() and not args.force_regen) else 'w'
    csv_file = open(csv_path, file_mode, newline='')
    keys = ['src_style','tgt_style','src_image','gen_image','content_vgg','style_vgg','clip_content','clip_style','content_lpips','style_lpips']
    writer = csv.DictWriter(csv_file, fieldnames=keys)
    if file_mode == 'w': writer.writeheader()

    print("\nStarting Transfer & Evaluation...")
    for src_id, (src_name, src_list) in test_images.items():
        for src_path in src_list:
            needed_targets = []
            for tgt_id, (tgt_name, _) in test_images.items():
                # 注意：这里允许 Self-Transfer (Monet->Monet) 以便作为基准对比
                if (src_path.name, tgt_name) not in processed_pairs:
                    needed_targets.append(tgt_id)
            
            if not needed_targets and not args.force_regen: continue 

            img_src = load_image_to_tensor(src_path, size=256, device=device)
            if img_src is None: continue

            with torch.no_grad():
                src_vgg = vgg_extractor.get_features(img_src)
                src_clip = None
                if has_clip:
                    pil_src = to_pil(img_src.squeeze(0).cpu())
                    inputs = clip_processor(images=pil_src, return_tensors='pt').to(device)
                    src_clip = clip_model.get_image_features(**inputs)
                    src_clip = (src_clip / src_clip.norm(p=2, dim=-1, keepdim=True))

            latent_src = encode_image(vae, img_src, device).to(torch.float32)
            latent_x0 = lgt.inversion(latent_src, src_id, num_steps=args.num_steps)

            for tgt_id in needed_targets:
                tgt_name = test_images[tgt_id][0]
                out_img_name = f"{src_name}_{src_path.stem}_to_{tgt_name}.jpg"
                out_img_path = out_dir / out_img_name
                
                img_gen = None
                if out_img_path.exists() and not args.force_regen:
                    img_gen = load_image_to_tensor(out_img_path, size=256, device=device)
                
                if img_gen is None:
                    latent_tgt = lgt.generation(latent_x0, tgt_id, num_steps=args.num_steps)
                    img_gen = decode_latent(vae, latent_tgt, device)
                    from torchvision.utils import save_image
                    save_image(img_gen, out_img_path)

                with torch.no_grad():
                    gen_vgg_raw = vgg_extractor.vgg((img_gen - vgg_extractor.mean)/vgg_extractor.std)
                    gen_vgg = []
                    h = (img_gen - vgg_extractor.mean)/vgg_extractor.std
                    for i, layer in enumerate(vgg_extractor.vgg):
                        h = layer(h)
                        if i in vgg_extractor.layer_ids: gen_vgg.append(h)
                    
                    gen_clip = None
                    if has_clip:
                        pil_gen = to_pil(img_gen.squeeze(0).cpu())
                        inputs = clip_processor(images=pil_gen, return_tensors='pt').to(device)
                        gen_clip = clip_model.get_image_features(**inputs)
                        gen_clip = gen_clip / gen_clip.norm(p=2, dim=-1, keepdim=True)

                    content_vgg = compute_distance_cpu_gpu_hybrid(gen_vgg, src_vgg, device)
                    clip_content = float(F.cosine_similarity(gen_clip, src_clip).item()) if has_clip else 0.0
                    content_lpips = 0.0
                    if loss_fn: content_lpips = float(loss_fn(to_lpips_input(img_gen), to_lpips_input(img_src)).item())

                    style_vgg_dists, clip_style_sims, style_lpips_dists = [], [], []
                    target_refs = ref_features[tgt_id]
                    
                    for ref in target_refs:
                        style_vgg_dists.append(compute_distance_cpu_gpu_hybrid(gen_vgg, ref['vgg'], device))
                        if has_clip and ref['clip'] is not None:
                            ref_clip_gpu = ref['clip'].to(device)
                            clip_style_sims.append(float(F.cosine_similarity(gen_clip, ref_clip_gpu).item()))
                        if loss_fn:
                            ref_img = load_image_to_tensor(ref['path'], size=256, device=device)
                            if ref_img is not None:
                                style_lpips_dists.append(loss_fn(to_lpips_input(img_gen), to_lpips_input(ref_img)).item())

                    style_vgg = np.mean(style_vgg_dists) if style_vgg_dists else 0.0
                    clip_style = np.mean(clip_style_sims) if clip_style_sims else 0.0
                    style_lpips = np.mean(style_lpips_dists) if style_lpips_dists else 0.0

                row = {
                    'src_style': src_name, 'tgt_style': tgt_name, 'src_image': str(src_path), 'gen_image': str(out_img_path),
                    'content_vgg': content_vgg, 'style_vgg': style_vgg, 'clip_content': clip_content, 'clip_style': clip_style,
                    'content_lpips': content_lpips, 'style_lpips': style_lpips
                }
                writer.writerow(row)
                csv_file.flush()

    csv_file.close()

    # ==========================================
    # 6. Structured Summary Generation (Revised)
    # ==========================================
    print("\nGenerating Structured Summary...")
    
    rows = []
    if csv_path.exists():
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for r in reader: rows.append(r)

    if not rows:
        print("No data.")
        return

    def to_float(x):
        try: return float(x)
        except: return 0.0

    # 1. 矩阵数据 (The Matrix)
    # 结构: summary['matrix'][src_style][tgt_style] = { metrics... }
    matrix = defaultdict(lambda: defaultdict(list))
    for r in rows:
        matrix[r['src_style']][r['tgt_style']].append(r)

    matrix_json = {}
    
    # 2. 核心指标聚合
    transfer_metrics = defaultdict(list) # Off-diagonal (True Transfer)
    identity_metrics = defaultdict(list) # Diagonal (Reconstruction)
    target_performance = defaultdict(list) # How well each style is generated

    for src, targets in matrix.items():
        matrix_json[src] = {}
        for tgt, items in targets.items():
            # 计算该组合的平均值
            pair_summary = {
                'count': len(items),
                'clip_style': np.mean([to_float(x['clip_style']) for x in items]),
                'clip_content': np.mean([to_float(x['clip_content']) for x in items]),
                'style_lpips': np.mean([to_float(x['style_lpips']) for x in items]),
                'content_lpips': np.mean([to_float(x['content_lpips']) for x in items]),
            }
            matrix_json[src][tgt] = pair_summary
            
            # 存入聚合池
            if src == tgt:
                identity_metrics['all'].append(pair_summary)
            else:
                transfer_metrics['all'].append(pair_summary)
                # 专门记录: "Photo -> Any Painting" 这种有意义的指标
                if src == 'photo':
                    transfer_metrics['photo_to_art'].append(pair_summary)
            
            target_performance[tgt].append(pair_summary)

    def avg_pool(pool, key):
        vals = [x[key] for x in pool]
        return float(np.mean(vals)) if vals else 0.0

    summary = {
        'checkpoint': str(checkpoint_path),
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        
        # [最有意义部分] 全局矩阵
        'matrix_breakdown': matrix_json,

        # [高级统计] 
        'analysis': {
            # 真正的风格迁移能力 (排除对角线)
            'style_transfer_ability': {
                'clip_style': avg_pool(transfer_metrics['all'], 'clip_style'),
                'clip_content': avg_pool(transfer_metrics['all'], 'clip_content'),
                'note': "Metrics for Off-Diagonal pairs (Source != Target)"
            },
            # 图像重建能力 (对角线)
            'identity_reconstruction': {
                'clip_content': avg_pool(identity_metrics['all'], 'clip_content'),
                'note': "Metrics for Diagonal pairs (Source == Target)"
            },
            # [特别关注] 照片转艺术能力
            'photo_to_art_performance': {
                'clip_style': avg_pool(transfer_metrics['photo_to_art'], 'clip_style'),
                'valid': len(transfer_metrics['photo_to_art']) > 0
            }
        },

        # [目标风格难度排行] 哪个风格最难学?
        'target_style_ranking': {}
    }

    # 生成难度排行
    for tgt, pool in target_performance.items():
        summary['target_style_ranking'][tgt] = avg_pool(pool, 'clip_style')

    summary_path = out_dir / 'summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
        
    print(f"✓ Structured summary saved to {summary_path}")
    print("\n[Quick Diagnosis]")
    print(f"Transfer Capability (CLIP Style): {summary['analysis']['style_transfer_ability']['clip_style']:.4f}")
    if summary['analysis']['photo_to_art_performance']['valid']:
        print(f"Photo->Art Capability (CLIP Style): {summary['analysis']['photo_to_art_performance']['clip_style']:.4f}")

if __name__ == '__main__':
    main()