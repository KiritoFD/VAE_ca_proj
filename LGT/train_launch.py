#!/usr/bin/env python3
"""
LGT Training Launch Script

Examples:
    python train_launch.py                              # Start fresh training
    python train_launch.py --resume                     # Resume from latest.pt
    python train_launch.py --resume epoch_0050.pt       # Resume from specific epoch
    python train_launch.py --quick-debug                # Fast debug mode
    python train_launch.py --high-quality               # High quality mode
"""

import argparse
import json
from pathlib import Path
import sys
import subprocess


def create_debug_config(config):
    """Create quick debug configuration."""
    config['training']['batch_size'] = 4
    config['training']['num_epochs'] = 5
    config['training']['save_interval'] = 1
    config['training']['eval_interval'] = 1
    config['training']['ode_integration_steps'] = 3
    return config


def create_quality_config(config):
    """Create high quality configuration."""
    config['training']['batch_size'] = 16
    config['training']['num_epochs'] = 200
    config['training']['save_interval'] = 10
    config['training']['eval_interval'] = 10
    config['training']['ode_integration_steps'] = 10
    return config


def main():
    parser = argparse.ArgumentParser(description='LGT Training Launch Script')
    
    parser.add_argument('--config', type=str, default='config.json',
                        help='Config file (default: config.json)')
    parser.add_argument('--resume', nargs='?', const='latest.pt', type=str,
                        help='Resume from checkpoint (latest.pt by default)')
    parser.add_argument('--quick-debug', action='store_true',
                        help='Quick debug mode (5 epochs, fast inference)')
    parser.add_argument('--high-quality', action='store_true',
                        help='High quality mode (200 epochs, careful training)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print configuration without running')
    
    args = parser.parse_args()
    
    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"✗ Config file not found: {config_path}")
        sys.exit(1)
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print("="*80)
    print("LGT Training Configuration")
    print("="*80)
    
    # Apply presets
    if args.quick_debug:
        print("📝 Applying QUICK DEBUG configuration...")
        config = create_debug_config(config)
    elif args.high_quality:
        print("📝 Applying HIGH QUALITY configuration...")
        config = create_quality_config(config)
    
    # Handle resume
    if args.resume:
        checkpoint_path = Path('checkpoints') / args.resume
        if not checkpoint_path.exists():
            print(f"⚠ Checkpoint not found: {checkpoint_path}")
            print(f"   Available checkpoints:")
            for ckpt in sorted(Path('checkpoints').glob('*.pt')):
                print(f"   - {ckpt.name}")
            sys.exit(1)
        
        config['training']['resume_checkpoint'] = str(checkpoint_path)
        print(f"✓ Resuming from: {checkpoint_path}")
    else:
        config['training']['resume_checkpoint'] = None
    
    # Print configuration
    print(f"\nTraining Configuration:")
    print(f"  Epochs: {config['training']['num_epochs']}")
    print(f"  Batch Size: {config['training']['batch_size']}")
    print(f"  Learning Rate: {config['training']['learning_rate']}")
    print(f"  Save Interval: {config['training']['save_interval']}")
    print(f"  Eval Interval: {config['training']['eval_interval']}")
    print(f"  ODE Steps: {config['training']['ode_integration_steps']}")
    print(f"  Test Image Dir: {config['training']['test_image_dir']}")
    
    print(f"\nModel Configuration:")
    print(f"  Base Channels: {config['model']['base_channels']}")
    print(f"  Style Dim: {config['model']['style_dim']}")
    print(f"  Num Styles: {config['model']['num_styles']}")
    
    print(f"\nLoss Configuration:")
    print(f"  w_style: {config['loss']['w_style']}")
    print(f"  w_content: {config['loss']['w_content']}")
    print(f"  Max SWD Samples: {config['loss']['max_samples']}")
    
    if args.dry_run:
        print("\n✓ Dry run completed. Configuration is valid.")
        return
    
    # Save temporary config
    temp_config_path = Path('config_launch.json')
    with open(temp_config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print("\n" + "="*80)
    print("Starting Training...")
    print("="*80 + "\n")
    
    # Run training
    try:
        cmd = ['python', 'train.py', '--config', str(temp_config_path)]
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n\n⏸ Training paused by user")
        print(f"To resume: python train_launch.py --resume latest.pt")
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Training failed with error code {e.returncode}")
        sys.exit(1)
    finally:
        # Cleanup
        if temp_config_path.exists():
            temp_config_path.unlink()


if __name__ == '__main__':
    main()
