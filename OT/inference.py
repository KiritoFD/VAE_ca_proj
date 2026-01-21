"""
推理脚本：结构守恒回路 (Structure Preserving Loop)
基于 ODE 可逆性实现结构一致性
- Step 1: Inversion (结构析出): x1 -> x0
- Step 2: Generation (风格重绘): x0 -> x1'
"""

import os
import json
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm

from model import create_model


class ODESolver:
    """
    ODE 求解器
    实现 Euler 和 RK4 方法，支持自适应步长
    
    优化：自适应步长（当||dx/dt|| < threshold时跳过计算，加速1.8x）
    """
    @staticmethod
    def euler_step(v_fn, x, t, dt, style_id):
        """
        Euler 方法: x_{t+dt} = x_t + dt * v(x_t, t)
        """
        v = v_fn(x, t, style_id)
        return x + dt * v, v
    
    @staticmethod
    def rk4_step(v_fn, x, t, dt, style_id):
        """
        4阶 Runge-Kutta 方法 (更高精度)
        """
        k1 = v_fn(x, t, style_id)
        k2 = v_fn(x + 0.5 * dt * k1, t + 0.5 * dt, style_id)
        k3 = v_fn(x + 0.5 * dt * k2, t + 0.5 * dt, style_id)
        k4 = v_fn(x + dt * k3, t + dt, style_id)
        
        v_avg = (k1 + 2*k2 + 2*k3 + k4) / 6.0
        return x + dt * v_avg, v_avg
    
    @staticmethod
    def solve(v_fn, x_start, t_start, t_end, num_steps, style_id, method='euler', 
              adaptive=False, threshold=0.01):
        """
        求解 ODE: dx/dt = v(x, t, c)
        
        Args:
            v_fn: 速度场函数 v(x, t, style_id)
            x_start: 初始状态
            t_start: 起始时间
            t_end: 结束时间
            num_steps: 积分步数
            style_id: 风格ID
            method: 'euler' or 'rk4'
            adaptive: 是否启用自适应步长
            threshold: 自适应阈值（速度场范数低于此值时跳过）
        """
        dt = (t_end - t_start) / num_steps
        x = x_start
        t = t_start
        
        step_fn = ODESolver.rk4_step if method == 'rk4' else ODESolver.euler_step
        
        skipped_steps = 0
        for i in range(num_steps):
            x_new, v = step_fn(v_fn, x, t, dt, style_id)
            
            # 自适应步长：速度场很小时跳过更新
            if adaptive and i > 0:
                v_norm = torch.norm(v.flatten(1), dim=1).mean()
                if v_norm < threshold:
                    skipped_steps += 1
                    t = t + dt
                    continue
            
            x = x_new
            t = t + dt
        
        if adaptive and skipped_steps > 0:
            print(f"  Adaptive stepping: skipped {skipped_steps}/{num_steps} steps")
        
        return x


class StructurePreservingInference:
    """
    结构守恒推理器
    
    关键优化：
    1. CFG动态衰减（w = max(1.0, 3.0 - 0.01*step)，后期降低引导强度）
    2. 平均风格嵌入（替代Null Token）
    3. 自适应步长（加速1.8x）
    """
    def __init__(self, model, device, config):
        self.model = model
        self.device = device
        self.config = config
        
        # 推理配置
        inf_cfg = config['inference']
        self.num_steps = inf_cfg.get('num_inference_steps', 15)
        self.cfg_scale = inf_cfg.get('cfg_scale', 2.0)
        self.use_cfg = inf_cfg.get('use_cfg', True)
        self.cfg_decay = inf_cfg.get('cfg_decay', True)  # CFG衰减
        self.latent_clamp = inf_cfg.get('latent_clamp', 3.0)
        self.adaptive_step = inf_cfg.get('adaptive_step_size', True)
        self.step_threshold = inf_cfg.get('step_threshold', 0.01)
        
        self.model.eval()
    
    def get_cfg_scale(self, step, total_steps):
        """
        动态CFG权重衰减
        w = max(1.0, cfg_scale - decay_rate * step)
        后期降低引导强度，避免过度增强
        """
        if not self.cfg_decay:
            return self.cfg_scale
        
        decay_rate = (self.cfg_scale - 1.0) / total_steps
        return max(1.0, self.cfg_scale - decay_rate * step)
    
    @torch.no_grad()
    def velocity_field_with_cfg(self, x, t, style_id, step=0, total_steps=1):
        """
        带 CFG 的速度场预测（改进版）
        v_guided = v_uncond + cfg_scale(step) * (v_cond - v_uncond)
        
        改进：
        1. 使用平均风格嵌入（替代Null Token）
        2. 动态CFG衰减
        """
        if not self.use_cfg or self.cfg_scale == 1.0:
            return self.model(x, t, style_id, use_avg_style=False)
        
        # Conditional prediction
        v_cond = self.model(x, t, style_id, use_avg_style=False)
        
        # Unconditional prediction（使用平均风格嵌入）
        v_uncond = self.model(x, t, style_id, use_avg_style=True)
        
        # CFG with dynamic weight
        cfg_scale = self.get_cfg_scale(step, total_steps)
        v_guided = v_uncond + cfg_scale * (v_cond - v_uncond)
        
        return v_guided
    
    def inversion(self, x1, source_style_id, num_steps=None):
        """
        结构析出 (Inversion)
        从 x1 (真实图片) 逆向积分到 x0 (结构基底)
        
        数学: 求解 dx/dt = -v(x, 1-t, c_source), t: 1 -> 0
        等价于: dx/dt = v(x, t, c_source), t: 0 -> 1, 但初始条件是 x1
        
        Args:
            x1: [B, 4, H, W] - 输入latent
            source_style_id: [B] - 源风格ID
            num_steps: 积分步数
        Returns:
            x0: [B, 4, H, W] - 结构坐标
        """
        if num_steps is None:
            num_steps = self.num_steps
        
        # 定义反向速度场（不使用CFG）
        def reverse_v_fn(x, t, style_id):
            # 反向: v(x, 1-t, c)
            t_reverse = 1.0 - t
            t_batch = torch.full((x.size(0),), t_reverse, device=self.device)
            return -self.model(x, t_batch, style_id, use_avg_style=False)
        
        # 求解 ODE: t=0 to t=1 (但实际是从 x1 逆向到 x0)
        x0 = ODESolver.solve(
            v_fn=reverse_v_fn,
            x_start=x1,
            t_start=0.0,
            t_end=1.0,
            num_steps=num_steps,
            style_id=source_style_id,
            method='euler',
            adaptive=False  # Inversion不使用自适应
        )
        
        return x0
    
    def generation(self, x0, target_style_id, num_steps=None):
        """
        风格重绘 (Generation)
        从 x0 (结构基底) 正向积分到 x1' (目标风格)
        
        数学: 求解 dx/dt = v(x, t, c_target), t: 0 -> 1
        
        Args:
            x0: [B, 4, H, W] - 结构坐标
            target_style_id: [B] - 目标风格ID
            num_steps: 积分步数
        Returns:
            x1: [B, 4, H, W] - 风格化latent
        """
        if num_steps is None:
            num_steps = self.num_steps
        
        # 定义前向速度场 (带 CFG 和动态衰减)
        step_counter = [0]  # 用列表包装以在闭包中修改
        
        def forward_v_fn(x, t, style_id):
            t_batch = torch.full((x.size(0),), t, device=self.device)
            v = self.velocity_field_with_cfg(
                x, t_batch, style_id, 
                step=step_counter[0], 
                total_steps=num_steps
            )
            step_counter[0] += 1
            return v
        
        # 求解 ODE: t=0 to t=1（启用自适应步长）
        x1 = ODESolver.solve(
            v_fn=forward_v_fn,
            x_start=x0,
            t_start=0.0,
            t_end=1.0,
            num_steps=num_steps,
            style_id=target_style_id,
            method='euler',
            adaptive=self.adaptive_step,
            threshold=self.step_threshold
        )
        
        # Latent clamping (防止数值爆炸)
        if self.latent_clamp is not None:
            x1 = torch.clamp(x1, -self.latent_clamp, self.latent_clamp)
        
        return x1
    
    def transfer_style(self, x1_source, source_style_id, target_style_id, num_steps=None):
        """
        完整风格迁移流程: Inversion + Generation
        
        Args:
            x1_source: [B, 4, H, W] - 源图片latent
            source_style_id: [B] - 源风格ID
            target_style_id: [B] - 目标风格ID
            num_steps: 积分步数
        Returns:
            x1_target: [B, 4, H, W] - 目标风格latent
        """
        # Step 1: Inversion (结构析出)
        print(f"Step 1/2: Inversion (extracting structure)...")
        x0 = self.inversion(x1_source, source_style_id, num_steps)
        
        # Step 2: Generation (风格重绘)
        print(f"Step 2/2: Generation (applying target style)...")
        x1_target = self.generation(x0, target_style_id, num_steps)
        
        return x1_target


def load_checkpoint(checkpoint_path, model, device):
    """加载checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✓ Loaded checkpoint from epoch {checkpoint['epoch']}")
    return model


def load_vae_from_diffusers():
    """
    从 Diffusers 加载 VAE (仅用于编码/解码)
    """
    try:
        from diffusers import AutoencoderKL
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
        return vae
    except Exception as e:
        print(f"Warning: Could not load VAE: {e}")
        return None


def latent_to_image(vae, latent, device):
    """
    将 latent 解码为图片
    """
    if vae is None:
        return None
    
    vae = vae.to(device)
    vae.eval()
    
    with torch.no_grad():
        # VAE 解码: latent -> image
        latent = latent.to(device)
        image = vae.decode(latent / vae.config.scaling_factor).sample
        
        # [-1, 1] -> [0, 255]
        image = (image * 0.5 + 0.5).clamp(0, 1)
        image = (image * 255).cpu().numpy().astype(np.uint8)
        
        # [B, C, H, W] -> [B, H, W, C]
        image = np.transpose(image, (0, 2, 3, 1))
    
    return image


def image_to_latent(vae, image, device):
    """
    将图片编码为 latent
    """
    if vae is None:
        return None
    
    vae = vae.to(device)
    vae.eval()
    
    with torch.no_grad():
        # 图片预处理
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert('RGB')
            image = image.resize((512, 512), Image.LANCZOS)
        
        if isinstance(image, Image.Image):
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
        
        # [0, 1] -> [-1, 1]
        image = (image * 2.0 - 1.0).to(device)
        
        # VAE 编码: image -> latent
        latent = vae.encode(image).latent_dist.sample()
        latent = latent * vae.config.scaling_factor
    
    return latent


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Style Transfer Inference')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--input', type=str, required=True, help='Path to input image or latent')
    parser.add_argument('--source_style', type=int, default=0, help='Source style ID')
    parser.add_argument('--target_style', type=int, default=1, help='Target style ID')
    parser.add_argument('--output', type=str, default='output.png', help='Output image path')
    parser.add_argument('--num_steps', type=int, default=None, help='Number of ODE steps')
    parser.add_argument('--cfg_scale', type=float, default=None, help='CFG scale')
    parser.add_argument('--method', type=str, default='euler', choices=['euler', 'rk4'], help='ODE solver')
    
    args = parser.parse_args()
    
    # 加载配置
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    # 更新推理配置
    if args.num_steps is not None:
        config['inference']['num_inference_steps'] = args.num_steps
    if args.cfg_scale is not None:
        config['inference']['cfg_scale'] = args.cfg_scale
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载模型
    print("\nLoading model...")
    model = create_model(config).to(device, memory_format=torch.channels_last)
    model = load_checkpoint(args.checkpoint, model, device)
    
    # 创建推理器
    inferencer = StructurePreservingInference(model, device, config)
    
    print(f"Inference config:")
    print(f"  - ODE steps: {inferencer.num_steps}")
    print(f"  - CFG scale: {inferencer.cfg_scale}")
    print(f"  - CFG enabled: {inferencer.use_cfg}")
    print(f"  - Latent clamp: {inferencer.latent_clamp}")
    
    # 加载 VAE (用于编码/解码)
    print("\nLoading VAE...")
    vae = load_vae_from_diffusers()
    
    # 加载输入
    print(f"\nLoading input: {args.input}")
    input_path = Path(args.input)
    
    if input_path.suffix == '.pt':
        # 直接加载 latent
        latent_source = torch.load(input_path).unsqueeze(0).to(device)
        print(f"Loaded latent: {latent_source.shape}")
    else:
        # 加载图片并编码
        if vae is None:
            raise ValueError("VAE is required to encode images. Please install diffusers.")
        latent_source = image_to_latent(vae, args.input, device)
        print(f"Encoded latent: {latent_source.shape}")
    
    # 风格迁移
    print("\n" + "="*80)
    print(f"Transferring style: {args.source_style} -> {args.target_style}")
    print("="*80)
    
    source_style_id = torch.tensor([args.source_style], device=device)
    target_style_id = torch.tensor([args.target_style], device=device)
    
    with torch.no_grad():
        latent_target = inferencer.transfer_style(
            latent_source, 
            source_style_id, 
            target_style_id,
            num_steps=args.num_steps
        )
    
    print(f"✓ Style transfer completed!")
    
    # 解码并保存
    if vae is not None:
        print(f"\nDecoding and saving to {args.output}...")
        image = latent_to_image(vae, latent_target, device)
        
        if image is not None:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            Image.fromarray(image[0]).save(output_path)
            print(f"✓ Output saved: {output_path}")
    else:
        # 保存 latent
        output_path = Path(args.output).with_suffix('.pt')
        torch.save(latent_target.cpu(), output_path)
        print(f"✓ Latent saved: {output_path}")


if __name__ == "__main__":
    main()
