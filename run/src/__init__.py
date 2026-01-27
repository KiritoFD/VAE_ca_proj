"""
LGT: Latent Geometric Thermodynamics

Geometric free energy optimization for style transfer in VAE latent space.
"""

__version__ = "1.0.0"

from .model import LGTUNet, TimestepEmbedding, StyleEmbedding, count_parameters
from .losses import (
    PatchSlicedWassersteinLoss,
    MultiScaleSWDLoss,
    TrajectoryMSELoss,
    GeometricFreeEnergyLoss,
    VelocityRegularizationLoss
)
from .inference import (
    LangevinSampler,
    LGTInference,
    load_vae,
    encode_image,
    decode_latent,
    tensor_to_pil
)

__all__ = [
    # Model
    "LGTUNet",
    "TimestepEmbedding",
    "StyleEmbedding",
    "count_parameters",
    
    # Losses
    "PatchSlicedWassersteinLoss",
    "MultiScaleSWDLoss",
    "TrajectoryMSELoss",
    "GeometricFreeEnergyLoss",
    "VelocityRegularizationLoss",
    
    # Inference
    "LangevinSampler",
    "LGTInference",
    "load_vae",
    "encode_image",
    "decode_latent",
    "tensor_to_pil",
]
