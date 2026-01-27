import torch
import torch.nn as nn
import math

class StyleHyperLinear(nn.Module):
    """
    Hyper-LoRA Linear Layer optimized for RTX 4070.
    
    Theory:
    Instead of w = w_base + delta_w, we calculate:
    y = x @ w_base.T + alpha * (x @ A.T @ B.T)
    
    This avoids materializing the massive (Batch, Out, In) delta matrix.
    Complexity: O(Batch * Tokens * Dim * Rank) instead of O(Batch * Dim^2).
    """
    def __init__(self, in_features, out_features, style_dim=256, rank=8, alpha=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        
        # 1. Base Weights (Frozen or Trainable)
        self.base_linear = nn.Linear(in_features, out_features)
        
        # 2. HyperNetwork (The Generator)
        # Input: Style Embedding [B, style_dim]
        # Output: Parameters for Matrix A [rank, in] and B [out, rank]
        # Total params to generate: (in + out) * rank
        self.num_lora_params = (in_features + out_features) * rank
        
        # A simple MLP projector
        self.hyper_net = nn.Sequential(
            nn.Linear(style_dim, style_dim),
            nn.SiLU(),
            nn.Linear(style_dim, self.num_lora_params)
        )
        
        # === CRITICAL: Zero Initialization ===
        # We initialize the LAST layer of HyperNet to pure zeros.
        # This ensures that at step 0, A=0, B=0, so Delta_W = 0.
        # The model behaves EXACTLY like the base model initially.
        nn.init.zeros_(self.hyper_net[-1].weight)
        nn.init.zeros_(self.hyper_net[-1].bias)

    def forward(self, x, style_emb):
        """
        x: [Batch, Tokens, In_Dim]
        style_emb: [Batch, Style_Dim]
        """
        batch_size = x.shape[0]
        
        # 1. Standard Linear Pass
        base_out = self.base_linear(x)
        
        # 2. Hyper-LoRA Pass (Dynamic)
        # Generate raw params: [Batch, (in+out)*rank]
        raw_params = self.hyper_net(style_emb)
        
        # Split into A and B
        # A: [Batch, Rank, In]
        # B: [Batch, Out, Rank]
        split_idx = self.in_features * self.rank
        matrix_a = raw_params[:, :split_idx].view(batch_size, self.rank, self.in_features)
        matrix_b = raw_params[:, split_idx:].view(batch_size, self.out_features, self.rank)
        
        # 3. Optimized Computation: x @ A.T @ B.T
        # We use torch.bmm for batched matrix multiplication.
        # x needs to be viewed as [Batch, Tokens, In]
        
        # Step I: Down-projection (x @ A.T)
        # x: [B, T, In], A.transpose: [B, In, R] -> [B, T, R]
        # We handle arbitrary spatial dimensions by flattening
        x_shape = x.shape
        x_flat = x.view(batch_size, -1, self.in_features)
        
        down_proj = torch.bmm(x_flat, matrix_a.transpose(1, 2))
        
        # Step II: Up-projection (result @ B.T)
        # down: [B, T, R], B.transpose: [B, R, Out] -> [B, T, Out]
        up_proj = torch.bmm(down_proj, matrix_b.transpose(1, 2))
        
        # Reshape back to original layout
        lora_out = up_proj.view(*x_shape[:-1], self.out_features)
        
        return base_out + self.alpha * lora_out

def run_verification():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running Validation on: {device}")
    
    # Configuration simulating a Bottleneck layer in SD
    B, T, C = 4, 64, 512  # Batch=4, 8x8 tokens, 512 channels
    STYLE_DIM = 256
    RANK = 8
    
    model = StyleHyperLinear(C, C, STYLE_DIM, RANK).to(device)
    
    # Inputs
    x = torch.randn(B, T, C, device=device)
    style_codes = torch.randn(B, STYLE_DIM, device=device)
    
    print("-" * 50)
    
    # --- TEST 1: Zero Initialization (Safety Check) ---
    print("[Test 1] Zero Initialization Safety")
    with torch.no_grad():
        y_base = model.base_linear(x)
        y_hyper = model(x, style_codes)
        
        diff = torch.max(torch.abs(y_base - y_hyper)).item()
        print(f"   Max Difference (Base vs Hyper): {diff:.9f}")
        
        if diff < 1e-6:
            print("   ✅ PASSED: Module starts as identity.")
        else:
            print("   ❌ FAILED: Module disturbs initial weights!")
            return

    # --- TEST 2: Dynamic Weight Generation (Batch Independence) ---
    print("\n[Test 2] Batch Independence & Dynamicity")
    # Manually perturb HyperNet weights to simulate training
    with torch.no_grad():
        model.hyper_net[-1].weight.normal_(0, 0.02)
    
    # Create two identical inputs but DIFFERENT styles
    x_same = x[0:1].repeat(2, 1, 1) # [2, T, C] identical content
    style_diff = torch.randn(2, STYLE_DIM, device=device) # Different styles
    
    with torch.no_grad():
        out = model(x_same, style_diff)
        # Output 0 and Output 1 should be different
        discrepancy = torch.mean(torch.abs(out[0] - out[1])).item()
        
    print(f"   Output Discrepancy (Same Content, Diff Style): {discrepancy:.6f}")
    if discrepancy > 1e-5:
        print("   ✅ PASSED: Different styles produce different transforms.")
    else:
        print("   ❌ FAILED: Module is ignoring style input!")

    # --- TEST 3: Gradient Flow (Learnability) ---
    print("\n[Test 3] Gradient Flow")
    style_codes.requires_grad = True
    model.zero_grad()
    
    y = model(x, style_codes)
    loss = y.mean()
    loss.backward()
    
    grad_norm = style_codes.grad.norm().item()
    print(f"   Style Gradient Norm: {grad_norm:.6f}")
    
    if grad_norm > 0:
        print("   ✅ PASSED: Gradients flow back to style codes.")
    else:
        print("   ❌ FAILED: Broken computational graph.")

    # --- TEST 4: Overhead Analysis ---
    print("\n[Test 4] Memory Overhead")
    base_params = sum(p.numel() for p in model.base_linear.parameters())
    hyper_params = sum(p.numel() for p in model.hyper_net.parameters())
    
    print(f"   Base Layer Params: {base_params}")
    print(f"   HyperNet Params:   {hyper_params}")
    print(f"   Overhead Ratio:    {hyper_params/base_params:.2f}x")
    print("   Note: For a rank 8 LoRA, this overhead is expected.")
    print("         Since we only replace Bottleneck attention, total VRAM impact is negligible.")

if __name__ == "__main__":
    run_verification()