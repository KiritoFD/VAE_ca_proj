"""Compatibility wrapper for the refactored training pipeline."""

from dataset import LatentDataset, elastic_deform
from physics import generate_latent, get_dynamic_epsilon, integrate_ode, invert_latent
from trainer import LGTTrainer
from main import main

__all__ = [
    'LatentDataset',
    'elastic_deform',
    'generate_latent',
    'get_dynamic_epsilon',
    'integrate_ode',
    'invert_latent',
    'LGTTrainer',
    'main',
]


if __name__ == "__main__":
    main()
