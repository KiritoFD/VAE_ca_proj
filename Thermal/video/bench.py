import torch
import torch.nn.functional as F
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
import cv2
import numpy as np
import argparse

def load_frames(video_path, max_frames=None):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if max_frames and len(frames) >= max_frames: break
    cap.release()
    # [T, 3, H, W], float 0-255
    return torch.tensor(np.array(frames)).permute(0, 3, 1, 2).float().cuda()

def compute_metrics(stylized_path, original_path):
    device = 'cuda'
    
    # Load RAFT Flow Model
    weights = Raft_Large_Weights.DEFAULT
    flow_model = raft_large(weights=weights).to(device).eval()
    transforms = weights.transforms()
    
    # Load Videos
    style_frames = load_frames(stylized_path)
    orig_frames = load_frames(original_path)
    
    # Resize to RAFT friendly size if needed, or keeping original
    # Metric 1: Temporal Warping Error
    warping_errs = []
    
    with torch.no_grad():
        for t in range(len(style_frames) - 1):
            # 1. Compute Flow on Original Video (Ground Truth Motion)
            img1, img2 = orig_frames[t:t+1], orig_frames[t+1:t+2]
            inputs = torch.cat([img1, img2], dim=0)
            inputs = transforms(inputs)
            flow = flow_model(inputs[0:1], inputs[1:2])[-1] # [1, 2, H, W]
            
            # 2. Warp Style[t] -> Style[t+1] using Flow
            # Create Grid
            B, C, H, W = flow.shape
            grid_y, grid_x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
            grid = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0).to(device) # [1, 2, H, W]
            
            # Apply Flow (Forward warping approximation for simplicity in monitoring)
            # Ideally backward flow + backward warp, but consistent fwd flow is OK for relative comparison
            grid_flow = grid + flow
            
            # Normalize to [-1, 1] for grid_sample
            grid_flow[:, 0] = 2.0 * grid_flow[:, 0] / (W - 1) - 1.0
            grid_flow[:, 1] = 2.0 * grid_flow[:, 1] / (H - 1) - 1.0
            grid_flow = grid_flow.permute(0, 2, 3, 1) # [1, H, W, 2]
            
            style_t = style_frames[t:t+1] / 255.0
            style_t_plus_1 = style_frames[t+1:t+2] / 255.0
            
            # Warp t to t+1
            warped_t = F.grid_sample(style_t, grid_flow, align_corners=True)
            
            # 3. Calculate L2 Error (ignore occlusion for quick check)
            diff = (warped_t - style_t_plus_1) ** 2
            warping_errs.append(diff.mean().item())
            
    avg_error = sum(warping_errs) / len(warping_errs)
    print(f"Warping Error (x1e3): {avg_error * 1000:.4f}")
    return avg_error

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--style", type=str, required=True)
    parser.add_argument("--orig", type=str, required=True)
    args = parser.parse_args()
    compute_metrics(args.style, args.orig)