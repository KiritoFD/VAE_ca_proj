"""
Run multiple training experiments sequentially.

Usage:
    python run_experiments.py               # run built-in presets sequentially
    python run_experiments.py --list        # list presets
    python run_experiments.py --exp exp1    # run a named preset
    python run_experiments.py --dry-run     # show commands without executing

Behavior:
- For each experiment preset, the script creates a working folder under "experiments/{name}" and writes a config.json there
- It updates the "checkpoint.save_dir" in the config to "checkpoints/{name}" to avoid conflicts
- Runs `python train.py --config experiments/{name}/config.json`
- Captures stdout/stderr to experiments/{name}/train.log
- Stores exit code and runtime to experiments/{name}/result.json

The script runs experiments sequentially to avoid GPU contention.
"""

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

BASE_DIR = Path(__file__).parent
EXPERIMENTS_DIR = BASE_DIR / "experiments"
EXPERIMENTS_DIR.mkdir(exist_ok=True)

# =================================================================             
# 🧪 LGT 实验预设集 (LGT Experiment Presets)
# =================================================================             
PRESETS = {
    # -------------------------------------------------------------------------
    

    # -------------------------------------------------------------------------
    # 1. 基准线 (Baseline)
    # -------------------------------------------------------------------------
    "lgt_baseline": {
        "desc": "📏 Baseline LGT: Default config (w_style=1.0, w_content=1.0)",
        "overrides": {} # Use config.json defaults
    },

    # -------------------------------------------------------------------------
    # 2. 势能平衡探索 (Energy Landscape Balance)
    # 验证：到底多强的风格势能才能克服内容的拓扑刚性？
    # -------------------------------------------------------------------------
    "lgt_style_focus": {
        "desc": "🎨 Style Focus: Stronger style potential (w_style=10.0)",
        "overrides": {
            "loss": {
                "w_style": 10.0,
                "w_content": 1.0
            },
            "training": {
                "save_interval": 10,
                "eval_interval": 10
            }
        }
    },
    "lgt_content_lock": {
        "desc": "🔒 Content Lock: Stronger topology constraint (w_content=5.0)",
        "overrides": {
            "loss": {
                "w_style": 1.0,
                "w_content": 5.0
            },
            "training": {
                "save_interval": 10,
                "eval_interval": 10
            }
        }
    },

    # -------------------------------------------------------------------------
    # 3. 几何尺度探索 (Geometric Scale)
    # 验证：3x3 Patch 够吗？更大的 Patch 是否能捕捉更宏观的笔触？
    # -------------------------------------------------------------------------
    "lgt_large_patch": {
        "desc": "📐 Large Patch: Use 5x5 patches for SWD (Captures macro-texture)",
        "overrides": {
            "loss": {
                "patch_size": 5,  # 3 -> 5 (dim: 36 -> 100)
                # 大Patch计算量大，稍微减少采样数以防OOM
                "max_samples": 2048 
            }
        }
    },

    # -------------------------------------------------------------------------
    # 4. 分布精度探索 (Distribution Fidelity)
    # 验证：更密集的采样和更多的投影是否能让梯度更精准？
    # -------------------------------------------------------------------------
    "lgt_hifi_swd": {
        "desc": "🎯 Hi-Fi SWD: High precision distribution matching (128 proj, 8192 samples)",
        "overrides": {
            "loss": {
                "num_projections": 128, # default 64
                "max_samples": 8192     # default 4096
            },
            "training": {
                "batch_size": 12 # 降低BS以补偿显存
            }
        }
    },

    # -------------------------------------------------------------------------
    # 5. 网络容量探索 (Model Capacity)
    # 验证：更深的 Decoder (Dynamic Conv) 是否能生成更复杂的纹理？
    # -------------------------------------------------------------------------
    "lgt_deep_decoder": {
        "desc": "🧠 Deep Decoder: More dynamic conv layers for complex texture generation",
        "overrides": {
            "model": {
                "num_decoder_blocks": 5 # default 3
            },
            "training": {
                "batch_size": 12
            }
        }
    },

    # -------------------------------------------------------------------------
    # 6. 长时热力学演化 (Long-term Evolution)
    # 验证：给系统更多时间“退火”，是否能找到更优的全局极小值？
    # -------------------------------------------------------------------------
    "lgt_long_run": {
        "desc": "⏳ Long Run: 200 epochs for thorough annealing",
        "overrides": {
            "training": {
                "num_epochs": 200,
                "save_interval": 20,
                "eval_interval": 20,
                "learning_rate": 5e-5 # 稍低的学习率防止震荡
            }
        }
    }
}


def deep_update(orig, overrides):
    """Recursively update dict ``orig`` with ``overrides``."""
    for k, v in overrides.items():
        if isinstance(v, dict) and k in orig and isinstance(orig[k], dict):
            deep_update(orig[k], v)
        else:
            orig[k] = v


def prepare_experiment(name, base_config_path, overrides):
    exp_dir = EXPERIMENTS_DIR / name
    if exp_dir.exists():
        # keep existing but back it up
        backup = exp_dir.parent / f"{name}_bak_{int(time.time())}"
        shutil.move(str(exp_dir), str(backup))
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Load base config
    with open(base_config_path, 'r') as f:
        cfg = json.load(f)

    # Apply overrides
    if overrides:
        deep_update(cfg, overrides)

    # Ensure checkpoint save dir is namespaced
    ckpt_dir = Path(cfg.get('checkpoint', {}).get('save_dir', 'checkpoints'))
    cfg.setdefault('checkpoint', {})
    cfg['checkpoint']['save_dir'] = str(Path('checkpoints') / name)

    # Set default test_image_dir to current if missing
    cfg.setdefault('training', {})
    cfg['training'].setdefault('test_image_dir', cfg.get('training', {}).get('test_image_dir', "/mnt/f/monet2photo/test"))

    # Clear any resume flag for fresh experiments
    cfg['training']['resume_checkpoint'] = None

    # Write experiment config
    config_path = exp_dir / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(cfg, f, indent=2)

    return exp_dir, config_path


def run_experiment(name, config_path, dry_run=False):
    exp_dir = config_path.parent
    log_path = exp_dir / 'train.log'
    result_path = exp_dir / 'result.json'

    cmd = ['python', 'train.py', '--config', str(config_path)]
    print(f"\n=== Experiment: {name} ===")
    print(f"Command: {' '.join(cmd)}")
    print(f"Logs: {log_path}")

    if dry_run:
        print("Dry run: skipping execution")
        return 0

    start = time.time()
    with open(log_path, 'wb') as logfile:
        proc = subprocess.Popen(cmd, stdout=logfile, stderr=subprocess.STDOUT)
        ret = proc.wait()
    duration = time.time() - start

    # Record result
    result = {
        'name': name,
        'return_code': ret,
        'duration_seconds': duration,
        'log_path': str(log_path)
    }
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"Finished experiment: {name} (code={ret}, time={duration:.1f}s)")
    return ret


def list_presets():
    print("\nAvailable LGT Experiment Presets:")
    print("="*60)
    for k, v in PRESETS.items():
        print(f"🔹 {k:<20} : {v.get('desc', '')}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Run multiple LGT training experiments')
    parser.add_argument('--list', action='store_true', help='List presets')
    parser.add_argument('--exp', type=str, help='Run a specific preset by name')
    parser.add_argument('--all', action='store_true', help='Run ALL presets sequentially')
    parser.add_argument('--dry-run', action='store_true', help='Prepare configs but do not run')
    parser.add_argument('--base-config', type=str, default='config.json', help='Base config path')

    args = parser.parse_args()

    base_config_path = BASE_DIR / args.base_config
    if not base_config_path.exists():
        print(f"Base config not found: {base_config_path}")
        return 1

    if args.list:
        list_presets()
        return 0

    if args.exp:
        presets_to_run = [args.exp]
    elif args.all:
        presets_to_run = list(PRESETS.keys())
    else:
        print("Please specify --exp <name> or --all. Use --list to see available experiments.")
        return 1

    for name in presets_to_run:
        if name not in PRESETS:
            print(f"Unknown preset: {name}")
            continue
        exp_dir, config_path = prepare_experiment(name, base_config_path, PRESETS[name].get('overrides'))
        ret = run_experiment(name, config_path, dry_run=args.dry_run)
        if ret != 0:
            print(f"Experiment {name} failed (return code {ret}). Stopping further runs.")
            return ret
        # Run evaluation automatically on success
        eval_out = EXPERIMENTS_DIR / name / 'evaluation'
        eval_cmd = ['python', 'run_evaluation.py', '--checkpoint', str(Path('checkpoints')/name/'latest.pt'), '--config', str(config_path), '--output', str(eval_out)]
        print(f"Running evaluation: {' '.join(eval_cmd)}")
        if not args.dry_run:
            eval_proc = subprocess.run(eval_cmd)
            if eval_proc.returncode != 0:
                print(f"Evaluation for {name} failed (code {eval_proc.returncode})")
                # continue to next experiment, but log failure
            else:
                print(f"Evaluation for {name} completed. Results in: {eval_out}")
        # Update aggregate report after each experiment
        agg_cmd = ['python', 'run_aggregate.py', '--experiments_dir', 'experiments', '--output', 'experiments/aggregate']
        print(f"Updating aggregate report: {' '.join(agg_cmd)}")
        if not args.dry_run:
            subprocess.run(agg_cmd)

    print('\nAll requested experiments finished.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
