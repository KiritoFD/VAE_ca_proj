
import torch
import torch.nn as nn

class LearnableStructureExtractor(nn.Module):
    """
    A lightweight Latent-to-Edge translation network.
    Acts as a 'proxy' for VAE Decode + Canny.
    """
    def __init__(self):
        super().__init__()
        # Input: 4 channels (Latent) -> Output: 1 channel (Edge Probability)
        self.net = nn.Sequential(
            # Layer 1: Expand features
            nn.Conv2d(4, 32, kernel_size=3, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            
            # Layer 2: Dilated Conv to capture context (Receptive Field ↑)
            nn.Conv2d(32, 64, kernel_size=3, padding=2, dilation=2),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            
            # Layer 3: Feature consolidation
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),

            # Layer 4: Projection to Edge Map
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid() # Output [0, 1] probability
        )

    def forward(self, x):
        return self.net(x)

if __name__ == "__main__":
    # Test compilation
    net = LearnableStructureExtractor().cuda()
    x = torch.randn(2, 4, 32, 32).cuda()
    y = net(x)
    print(f"Structure Net Output: {y.shape}") # Should be [2, 1, 32, 32]