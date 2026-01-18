import torch
import torch.nn as nn
import os
import json
import numpy as np
from PIL import Image
from diffusers import AutoencoderKL
from SAFlow import SAFModel 

# ================= 配置 =================
CKPT_PATH = r"g:\GitHub\VAE_ca_proj\checkpoints\stage1_epoch10.pt"
OUTPUT_DIR = "./mnn_export_final"
CONFIG_PATH = "config.json"
TEST_IMG_PATH = "test.jpg" 
# =======================================

def load_config():
    with open(CONFIG_PATH, 'r') as f: return json.load(f)

class EncoderWrapper(nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae
    def forward(self, x):
        return self.vae.encode(x).latent_dist.mode() * 0.18215

class FlowWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x_t, x_cond, t, s):
        return self.model(x_t, x_cond, t, s)

class DecoderWrapper(nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae
    def forward(self, z):
        z = z / 0.18215
        out = self.vae.decode(z).sample
        return (out / 2.0) + 0.5 # 直接输出 0~1

def export():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cpu")
    
    cfg = load_config()
    saf_model = SAFModel(**cfg['model']).to(device)
    saf_model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
    saf_model.eval()
    
    vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae").to(device)
    vae.eval()
    
    # 包装模型
    encoder = EncoderWrapper(vae)
    flow_net = FlowWrapper(saf_model)
    decoder = DecoderWrapper(vae)
    
    # 准备 Dummy Inputs
    print("📷 正在构造 Dummy Inputs...")
    img = Image.open(TEST_IMG_PATH).convert("RGB").resize((512, 512))
    img_tensor = torch.from_numpy(np.array(img).astype(np.float32) / 127.5 - 1.0).permute(2, 0, 1).unsqueeze(0)
    
    with torch.no_grad():
        latent_c = encoder(img_tensor)
    
    # 【关键】x_t 和 x_cond 必须使用物理上不同的 Tensor 实例，防止被 ONNX 优化器合并
    dummy_xt = torch.randn(1, 4, 64, 64)
    dummy_xc = latent_c.clone() 
    dummy_t = torch.tensor([0.5]).float()
    dummy_s = torch.tensor([0]).int()

    # Opset 14 + Classic Mode (Training=False 强制触发旧版导出逻辑)
    common_args = {
        "opset_version": 14,
        "do_constant_folding": True,
        "keep_initializers_as_inputs": False,
        "training": torch.onnx.TrainingMode.EVAL
    }

    print(">>> 导出 Encoder...")
    torch.onnx.export(encoder, img_tensor, f"{OUTPUT_DIR}/Encoder.onnx",
                      input_names=['input'], output_names=['output'], **common_args)
    
    print(">>> 导出 Flow...")
    torch.onnx.export(flow_net, (dummy_xt, dummy_xc, dummy_t, dummy_s), f"{OUTPUT_DIR}/Flow.onnx",
                      input_names=['x_t', 'x_cond', 't', 's'], output_names=['output'], **common_args)
    
    print(">>> 导出 Decoder...")
    torch.onnx.export(decoder, dummy_xc, f"{OUTPUT_DIR}/Decoder.onnx",
                      input_names=['input'], output_names=['output'], **common_args)
    
    print(f"✅ 导出完成！")

if __name__ == "__main__":
    export()