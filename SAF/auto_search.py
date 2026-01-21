import sys
import os
import torch
import gc
import time
import json
from pathlib import Path

# 🟢 修改：导入新的训练器
sys.path.append(str(Path(__file__).parent.parent))  # 添加项目根目录到路径
from train import OTCFMTrainer, LatentDataset, create_model
from torch.utils.data import DataLoader

# 🟢 新增：导入评估模块（如果存在）
try:
    from eval_lpips import Evaluator as LPIPSEvaluator
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print("⚠️  LPIPS Evaluator not found, skipping LPIPS evaluation")

try:
    from eval_clip import Evaluator as CLIPEvaluator
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("⚠️  CLIP Evaluator not found, skipping CLIP evaluation")

try:
    from eval_vgg import Evaluator as VGGEvaluator
    VGG_AVAILABLE = True
except ImportError:
    VGG_AVAILABLE = False
    print("⚠️  VGG Evaluator not found, skipping VGG evaluation")

# ==============================================================================
# 🎛️ 实验配置中心
# 这里列出的每个字典代表一次完整的训练任务。
# 字典中的结构与 config.json 完全一致，未列出的参数将使用 config.json 的默认值。
# ==============================================================================

EXPERIMENTS = [
    
    # -------------------------------------------------------------------------
    # 第一组：基准 (Baseline)
    # -------------------------------------------------------------------------
    {
        "name": "Exp1_Baseline",
        "description": "【基准线】各项参数中庸，用于对比其他实验的提升幅度。",
        "training": {
            "learning_rate": 1e-4,
            "label_drop_prob": 0.10,
            "stage1_epochs": 100,
        }
    },

    # -------------------------------------------------------------------------
    # 第二组：激进学习率 (Aggressive LR)
    # 验证：更快的收敛是否能保持风格质量
    # -------------------------------------------------------------------------
    {
        "name": "Exp2_HighLR",
        "description": "【高学习率】2倍LR，测试快速收敛是否影响风格质量。",
        "training": {
            "learning_rate": 2e-4,
            "label_drop_prob": 0.10,
            "stage1_epochs": 120,
        }
    },

    # -------------------------------------------------------------------------
    # 第三组：精细打磨 (Precision Mode) - 重点关注！
    # 验证：更小的学习率和更多epoch是否提升生成质量
    # -------------------------------------------------------------------------
    {
        "name": "Exp3_SlowCook",
        "description": "【慢工细活】极低LR + 长Epoch。旨在提升生成质量。",
        "training": {
            "learning_rate": 2e-5,
            "label_drop_prob": 0.05,  # 更少的label dropping
            "stage1_epochs": 300,
        }
    },

    # -------------------------------------------------------------------------
    # 第四组：高CFG dropout (Strong Unconditional)
    # 验证：更强的CFG训练是否提升推理时的控制力
    # -------------------------------------------------------------------------
    {
        "name": "Exp4_HighDropout",
        "description": "【强CFG训练】20%概率使用平均风格嵌入。",
        "training": {
            "learning_rate": 1e-4,
            "label_drop_prob": 0.20,
            "stage1_epochs": 150,
        }
    },

    # -------------------------------------------------------------------------
    # 第五组：动态epsilon关闭 (Static Epsilon)
    # 验证：固定epsilon是否影响训练稳定性
    # -------------------------------------------------------------------------
    {
        "name": "Exp5_StaticEpsilon",
        "description": "【固定epsilon】关闭动态epsilon，测试对训练的影响。",
        "training": {
            "learning_rate": 1e-4,
            "label_drop_prob": 0.10,
            "dynamic_epsilon": False,
            "stage1_epochs": 100,
        }
    },

    # -------------------------------------------------------------------------
    # 第六组：大Batch Size (High Stability)
    # 验证：更大的batch size是否带来更稳定的梯度
    # 注意：需要根据显存调整
    # -------------------------------------------------------------------------
    {
        "name": "Exp6_HighBS",
        "description": "【高稳定性】大Batch Size，梯度估计更准确。",
        "training": {
            "learning_rate": 1e-4,
            "batch_size": 32,  # 根据8G显存调整
            "label_drop_prob": 0.10,
            "stage1_epochs": 120,
        }
    },

    # -------------------------------------------------------------------------
    # 第七组：快速epsilon预热 (Fast Warmup)
    # 验证：更快达到直线路径是否加速收敛
    # -------------------------------------------------------------------------
    {
        "name": "Exp7_FastWarmup",
        "description": "【快速预热】50 epoch达到最大epsilon。",
        "training": {
            "learning_rate": 1e-4,
            "label_drop_prob": 0.10,
            "epsilon_warmup_epochs": 50,
            "stage1_epochs": 100,
        }
    }
]

# ==============================================================================
# 🟢 修改：评估函数 - 统一输出到实验目录
# ==============================================================================
def run_evaluations(ckpt_path, exp_name, exp_ckpt_dir, config_path="config.json"):
    """
    运行所有三个评估脚本并记录结果
    所有评估结果统一保存到 exp_ckpt_dir/evaluation/ 下
    """
    eval_dir = Path(exp_ckpt_dir) / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print(f"📊 Running Evaluations for: {exp_name}")
    print(f"📂 Results will be saved to: {eval_dir}")
    print("="*60)
    
    # 读取配置获取参考目录
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    
    ref_dir = cfg.get("data", {}).get("raw_data_root", None)
    target_dir = cfg.get("inference", {}).get("image_path", "")
    
    # 🟢 汇总结果字典
    results_summary = {
        "experiment_name": exp_name,
        "checkpoint_path": str(ckpt_path),
        "evaluation_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": {}
    }
    
    # 1. LPIPS Evaluation
    if LPIPS_AVAILABLE:
        try:
            print("\n🔹 [1/3] Running LPIPS Evaluation...")
            if not target_dir:
                raise ValueError("Target directory not configured in config.json")
            lpips_eval = LPIPSEvaluator(str(ckpt_path), config_path)
            lpips_results = lpips_eval.evaluate(target_dir, batch_size=2, save_dir=str(eval_dir))
            results_summary["metrics"]["lpips"] = lpips_results
            del lpips_eval
            gc.collect()
            torch.cuda.empty_cache()
            print("✅ LPIPS Evaluation Complete")
        except Exception as e:
            print(f"❌ LPIPS Evaluation Failed: {e}")
            results_summary["metrics"]["lpips"] = {"error": str(e)}
    
    # 2. CLIP Evaluation
    if CLIP_AVAILABLE:
        try:
            print("\n🔹 [2/3] Running CLIP Evaluation...")
            if not target_dir:
                raise ValueError("Target directory not configured in config.json")
            clip_eval = CLIPEvaluator(str(ckpt_path), config_path)
            clip_results = clip_eval.evaluate(target_dir, batch_size=2, save_dir=str(eval_dir))
            results_summary["metrics"]["clip"] = clip_results
            del clip_eval
            gc.collect()
            torch.cuda.empty_cache()
            print("✅ CLIP Evaluation Complete")
        except Exception as e:
            print(f"❌ CLIP Evaluation Failed: {e}")
            results_summary["metrics"]["clip"] = {"error": str(e)}
    
    # 3. VGG Style Evaluation
    if VGG_AVAILABLE:
        try:
            print("\n🔹 [3/3] Running VGG Style Evaluation...")
            vgg_eval = VGGEvaluator(str(ckpt_path), ref_root=ref_dir, config_path=config_path)
            vgg_results = vgg_eval.evaluate(bs=1, save_dir=str(eval_dir))
            results_summary["metrics"]["vgg"] = vgg_results
            del vgg_eval
            gc.collect()
            torch.cuda.empty_cache()
            print("✅ VGG Evaluation Complete")
        except Exception as e:
            print(f"❌ VGG Evaluation Failed: {e}")
            results_summary["metrics"]["vgg"] = {"error": str(e)}
    
    # 🟢 保存汇总结果到 JSON
    summary_path = eval_dir / "metrics_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, indent=4, ensure_ascii=False)
    
    print("\n" + "="*60)
    print(f"📊 All Evaluations Finished for: {exp_name}")
    print(f"📄 Summary saved to: {summary_path}")
    print("="*60)
    
    return results_summary

# ==============================================================================
# 🟢 新增：合并配置的辅助函数
# ==============================================================================
def merge_config(base_config, override):
    """递归合并配置字典"""
    result = base_config.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_config(result[key], value)
        else:
            result[key] = value
    return result

# ==============================================================================
# 自动化引擎 (Auto-Pilot)
# ==============================================================================
def run_grid_search():
    # 加载基础配置
    base_config_path = Path(__file__).parent.parent / "config.json"
    with open(base_config_path, 'r', encoding='utf-8') as f:
        base_config = json.load(f)
    
    # 结果总根目录
    ROOT_SAVE_DIR = Path("AutoSearch_Results")
    ROOT_SAVE_DIR.mkdir(exist_ok=True)
    
    print(f"🚀 Starting Grid Search: {len(EXPERIMENTS)} Experiments Queued.")
    print(f"📂 Root Output: {ROOT_SAVE_DIR.absolute()}")

    for i, exp in enumerate(EXPERIMENTS):
        exp_name = exp['name']
        print("\n" + "#"*60)
        print(f"▶️  [{i+1}/{len(EXPERIMENTS)}] Running Experiment: {exp_name}")
        print(f"ℹ️  Description: {exp.get('description', 'N/A')}")
        print("#"*60)

        # 1. 构造本次实验的专属目录
        exp_dir = ROOT_SAVE_DIR / exp_name
        ckpt_dir = exp_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        
        # 2. 构造配置覆盖
        config_override = merge_config(base_config, {
            "checkpoint": {
                "save_dir": str(ckpt_dir)
            }
        })
        
        # 将用户定义的参数合并进去
        for k, v in exp.items():
            if k not in ["name", "description"]:
                if k in config_override and isinstance(config_override[k], dict):
                    config_override[k] = merge_config(config_override[k], v)
                else:
                    config_override[k] = v
        
        # 保存本次实验的配置到文件
        exp_config_path = exp_dir / "experiment_config.json"
        with open(exp_config_path, 'w', encoding='utf-8') as f:
            json.dump(config_override, f, indent=4, ensure_ascii=False)
        print(f"📝 Experiment config saved to: {exp_config_path}")

        start_time = time.time()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        try:
            # 3. 创建数据集和DataLoader
            print("\n📦 Loading dataset...")
            dataset = LatentDataset(
                data_root=config_override['data']['data_root'],
                num_styles=config_override['data']['num_classes']
            )
            
            dataloader = DataLoader(
                dataset,
                batch_size=config_override['training']['batch_size'],
                shuffle=True,
                num_workers=config_override['training'].get('num_workers', 4),
                pin_memory=True,
                persistent_workers=True if config_override['training'].get('num_workers', 4) > 0 else False
            )
            
            # 4. 创建模型
            print("\n🏗️  Creating model...")
            model = create_model(config_override).to(device, memory_format=torch.channels_last)
            num_params = sum(p.numel() for p in model.parameters()) / 1e6
            print(f"Model parameters: {num_params:.2f}M")
            
            # 5. 创建训练器
            print("\n🚂 Initializing trainer...")
            trainer = OTCFMTrainer(config_override, model, device)
            
            # 6. 运行训练
            print("\n🎓 Starting training...")
            trainer.train(dataloader)
            
            print(f"✅ Experiment [{exp_name}] Training Completed in {(time.time() - start_time)/60:.1f} mins.")
            
            # 7. 运行评估（如果有最终checkpoint）
            final_ckpt = ckpt_dir / f"stage1_epoch{config_override['training']['stage1_epochs']}.pt"
            if not final_ckpt.exists():
                # 尝试找最新的checkpoint
                ckpts = sorted(ckpt_dir.glob("stage1_epoch*.pt"))
                if ckpts:
                    final_ckpt = ckpts[-1]
            
            if final_ckpt.exists():
                print(f"\n📊 Using checkpoint: {final_ckpt.name}")
                # 先清理训练器释放显存
                del trainer, model, dataloader, dataset
                gc.collect()
                torch.cuda.empty_cache()
                
                # 运行评估
                run_evaluations(final_ckpt, exp_name, ckpt_dir, str(exp_config_path))
            else:
                print("⚠️  No checkpoint found for evaluation")

        except KeyboardInterrupt:
            print("\n🛑 User Interrupted. Exiting...")
            sys.exit(0)
        except Exception as e:
            print(f"\n❌ Experiment [{exp_name}] Failed!")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # 8. 显存清理
            gc.collect()
            torch.cuda.empty_cache()
            print("🧹 GPU Memory Cleared.")

    print("\n" + "="*60)
    print("🎉 All Experiments Finished!")
    print(f"📂 Results saved to: {ROOT_SAVE_DIR.absolute()}")
    print("="*60)

if __name__ == "__main__":
    run_grid_search()
