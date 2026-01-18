#pip install onnx onnx-simplifier MNN
import torch
import torch.nn as nn
import os
import json
import numpy as np
import subprocess
from PIL import Image
from diffusers import AutoencoderKL
import onnx
from onnxsim import simplify

# 假设 SAFlow.py 在同一目录下，否则请调整路径
try:
    from SAFlow import SAFModel
except ImportError:
    print("❌ 错误: 找不到 SAFlow.py，请确保该文件在当前目录下或在 PYTHONPATH 中。")
    exit(1)

# ================= 配置 =================
CKPT_PATH = r"g:\GitHub\VAE_ca_proj\checkpoints\stage1_epoch10.pt"
OUTPUT_DIR = "./mnn_export_final"
CONFIG_PATH = "config.json"
TEST_IMG_PATH = "test.jpg"  # 必须存在，用于生成准确的 Trace
OPSET_VERSION = 14          # Opset 14 兼容性较好
# =======================================

def load_config():
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"找不到配置文件: {CONFIG_PATH}")
    with open(CONFIG_PATH, 'r') as f:
        return json.load(f)

# --- Wrappers (保持不变) ---
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
        # 确保输入类型匹配，防止 ONNX 类型推断错误
        return self.model(x_t, x_cond, t, s)

class DecoderWrapper(nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae
    def forward(self, z):
        z = z / 0.18215
        out = self.vae.decode(z).sample
        out = (out / 2.0) + 0.5
        return torch.clamp(out, 0.0, 1.0)

def load_real_image(path):
    if not os.path.exists(path):
        print(f"⚠️ 警告: 找不到 {path}，使用随机噪声代替（可能会导致量化层统计不准）！")
        return torch.randn(1, 3, 512, 512)
    
    print(f"📷 读取测试图片: {path}")
    img = Image.open(path).convert("RGB").resize((512, 512))
    img = np.array(img).astype(np.float32) / 127.5 - 1.0
    img = img.transpose(2, 0, 1)
    return torch.from_numpy(img).unsqueeze(0)

def onnx_simplify(onnx_path):
    """ 使用 onnx-simplifier 简化模型 """
    print(f"   Now Simplifying: {onnx_path} ...")
    try:
        model = onnx.load(onnx_path)
        model_sim, check = simplify(model)
        if not check:
            print("   ⚠️ onnx-simplifier 校验失败，跳过简化步骤")
            return
        onnx.save(model_sim, onnx_path)
        print("   ✅ Simplified.")
    except Exception as e:
        print(f"   ❌ Simplify 失败: {e}")

def convert_to_mnn(onnx_path, mnn_path):
    """ 调用 MNNConvert 命令行工具 """
    print(f"   Now Converting to MNN: {mnn_path} ...")
    
    # 按照你的要求：严禁加 --fp16
    # 如果未来需要 FP16，加上 --fp16 即可
    cmd = [
        "MNNConvert",
        "-f", "ONNX",
        "--modelFile", onnx_path,
        "--MNNModel", mnn_path,
        "--bizCode", "SAFlow",
        # "--fp16" # 你明确要求不加
    ]
    
    try:
        # Windows 下 shell=True 有时能解决找不到命令的问题，Linux 下通常不需要
        is_windows = os.name == 'nt'
        subprocess.check_call(cmd, shell=is_windows)
        print("   ✅ MNN Conversion Success.")
    except subprocess.CalledProcessError:
        print("   ❌ MNNConvert 失败！请检查是否安装了 MNN (pip install MNN) 并且添加到了环境变量。")
    except FileNotFoundError:
        print("   ❌ 找不到 MNNConvert 命令。")

def export_pipeline():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cpu") # 导出时建议用 CPU，避免 CUDA 算子带来的兼容性问题
    
    print("1. 加载 PyTorch 模型...")
    cfg = load_config()
    
    saf_model = SAFModel(**cfg['model']).to(device)
    if os.path.exists(CKPT_PATH):
        print(f"   Load Checkpoint: {CKPT_PATH}")
        saf_model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
    else:
        print(f"⚠️ Warning: Checkpoint not found at {CKPT_PATH}, using random weights.")
    saf_model.eval()
    
    print("   Load VAE...")
    vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae").to(device)
    vae.eval()
    
    encoder = EncoderWrapper(vae).to(device)
    flow_net = FlowWrapper(saf_model).to(device)
    decoder = DecoderWrapper(vae).to(device)
    
    # ================= 准备输入数据 =================
    real_img_tensor = load_real_image(TEST_IMG_PATH).to(device)
    
    print("2. 计算真实 Latent 用于 Flow 输入...")
    with torch.no_grad():
        real_latent = encoder(real_img_tensor) # [1, 4, 64, 64]
    
    # 标量输入
    dummy_t = torch.tensor([0.5]).float().to(device)
    dummy_s = torch.tensor([0]).int().to(device)

    # ================= 定义导出任务 =================
    tasks = [
        {
            "name": "Encoder",
            "model": encoder,
            "args": (real_img_tensor,),
            "input_names": ['input'],
            "output_names": ['output']
        },
        {
            "name": "Flow",
            "model": flow_net,
            "args": (real_latent, real_latent, dummy_t, dummy_s),
            "input_names": ['x_t', 'x_cond', 't', 's'],
            "output_names": ['output']
        },
        {
            "name": "Decoder",
            "model": decoder,
            "args": (real_latent,),
            "input_names": ['input'],
            "output_names": ['output']
        }
    ]

    # ================= 循环执行导出 =================
    for task in tasks:
        name = task["name"]
        onnx_file = os.path.join(OUTPUT_DIR, f"{name}.onnx")
        mnn_file = os.path.join(OUTPUT_DIR, f"{name}.mnn")
        
        print(f"\n>>> 处理 {name} ...")
        
        # 1. Torch -> ONNX
        torch.onnx.export(
            task["model"],
            task["args"],
            onnx_file,
            input_names=task["input_names"],
            output_names=task["output_names"],
            opset_version=OPSET_VERSION,
            do_constant_folding=True,
            keep_initializers_as_inputs=False,
            check_trace=False
        )
        
        # 2. Simplify ONNX
        onnx_simplify(onnx_file)
        
        # 3. ONNX -> MNN
        convert_to_mnn(onnx_file, mnn_file)

    print(f"\n🎉 全部完成！文件位于: {os.path.abspath(OUTPUT_DIR)}")

if __name__ == "__main__":
    with torch.no_grad():
        export_pipeline()