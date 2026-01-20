import sys
import os
import torch
import gc
import time
from pathlib import Path
from train import LSFMTrainer

# 🟢 新增：导入评估模块
from eval_lpips import Evaluator as LPIPSEvaluator
from eval_clip import Evaluator as CLIPEvaluator
from eval_vgg import Evaluator as VGGEvaluator

# ==============================================================================
# 🎛️ 实验配置中心
# 这里列出的每个字典代表一次完整的训练任务。
# 字典中的结构与 config.json 完全一致，未列出的参数将使用 config.json 的默认值。
# ==============================================================================

EXPERIMENTS = [
    # --- 实验 1: 基准对照组 (大火慢炖) ---
    {
        "name": "Exp1_Baseline_LR1e4_W5",
        "description": "标准 LR，权重 5，跑 100 轮看收敛",
        "training": {
            "learning_rate": 1e-4,
            "transfer_loss_weight": 5.0,
            "stage1_epochs": 100,
            "batch_size": 64
        }
    },
    
    # --- 实验 2: 激进权重组 (强迫症模式) ---
    {
        "name": "Exp2_HighWeight_W20",
        "description": "极大增加转换权重，看是否能产生更强烈的风格",
        "training": {
            "learning_rate": 8e-5,
            "transfer_loss_weight": 20.0,
            "stage1_epochs": 100,
        }
    },

    # --- 实验 3: 小火慢炖 (细节打磨) ---
    {
        "name": "Exp3_LowLR_LongRun",
        "description": "极低 LR，跑久一点，防止错过最优解",
        "training": {
            "learning_rate": 2e-5,
            "transfer_loss_weight": 15.0,
            "stage1_epochs": 200,
        }
    },
    
    # --- 实验 4: 甚至可以改模型参数 (如果显存允许) ---
    # {
    #     "name": "Exp4_DeeperModel",
    #     "model": {
    #         "depth": 10,
    #         "dim": 768
    #     },
    #     "training": {
    #         "batch_size": 32
    #     }
    # }
]

# ==============================================================================
# 🟢 新增：评估函数
# ==============================================================================
def run_evaluations(ckpt_path, exp_name, config_path="config.json"):
    """
    运行所有三个评估脚本并记录结果
    """
    import json
    
    print("\n" + "="*60)
    print(f"📊 Running Evaluations for: {exp_name}")
    print("="*60)
    
    # 读取配置获取参考目录
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    
    ref_dir = cfg.get("data", {}).get("data_root", None)
    if ref_dir:
        ref_dir = ref_dir.strip('"').strip("'")
    
    results = {}
    
    # 1. LPIPS Evaluation
    try:
        print("\n🔹 [1/3] Running LPIPS Evaluation...")
        lpips_eval = LPIPSEvaluator(str(ckpt_path), config_path)
        target_dir = cfg.get("inference", {}).get("image_path", "").strip('"').strip("'")
        if target_dir:
            lpips_eval.evaluate(target_dir, batch_size=2)
        del lpips_eval
        gc.collect()
        torch.cuda.empty_cache()
        print("✅ LPIPS Evaluation Complete")
    except Exception as e:
        print(f"❌ LPIPS Evaluation Failed: {e}")
    
    # 2. CLIP Evaluation
    try:
        print("\n🔹 [2/3] Running CLIP Evaluation...")
        clip_eval = CLIPEvaluator(str(ckpt_path), config_path)
        target_dir = cfg.get("inference", {}).get("image_path", "").strip('"').strip("'")
        if target_dir:
            clip_eval.evaluate(target_dir, batch_size=2)
        del clip_eval
        gc.collect()
        torch.cuda.empty_cache()
        print("✅ CLIP Evaluation Complete")
    except Exception as e:
        print(f"❌ CLIP Evaluation Failed: {e}")
    
    # 3. VGG Style Evaluation
    try:
        print("\n🔹 [3/3] Running VGG Style Evaluation...")
        vgg_eval = VGGEvaluator(str(ckpt_path), ref_root=ref_dir, config_path=config_path)
        vgg_eval.evaluate(bs=1)
        del vgg_eval
        gc.collect()
        torch.cuda.empty_cache()
        print("✅ VGG Evaluation Complete")
    except Exception as e:
        print(f"❌ VGG Evaluation Failed: {e}")
    
    print("\n" + "="*60)
    print(f"📊 All Evaluations Finished for: {exp_name}")
    print("="*60)

# ==============================================================================
# 自动化引擎 (Auto-Pilot)
# ==============================================================================
def run_grid_search():
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
        vis_dir = exp_dir / "visualizations"
        
        # 2. 构造配置覆盖 (Override)
        config_override = {
            "checkpoint": {
                "save_dir": str(ckpt_dir)
            },
            "inference": {
                "save_dir": str(vis_dir),
                "num_inference_steps": 4 
            }
        }
        
        # 将用户定义的参数合并进去 (training, model, data 等)
        for k, v in exp.items():
            if k not in ["name", "description"]:
                config_override[k] = v

        start_time = time.time()
        trainer = None

        try:
            # 3. 实例化训练器 (传入覆盖参数)
            trainer = LSFMTrainer(config_override=config_override)
            
            # 4. 运行训练 (只跑 Stage 1 即可快速验证风格)
            trainer.run_stage1()
            
            # 5. 强制执行一次最终推理
            print("🎨 Running Final Inference...")
            final_model = trainer.get_model()
            final_ckpt = trainer.ckpt_dir / "stage1_final.pt"
            if final_ckpt.exists():
                trainer.safe_load(final_model, torch.load(final_ckpt))
                trainer.do_inference(final_model, "final", "stage1_final")
            
            print(f"✅ Experiment [{exp_name}] Training Completed in {(time.time() - start_time)/60:.1f} mins.")
            
            # 🟢 6. 运行评估脚本
            if final_ckpt.exists():
                # 先清理训练器释放显存
                del trainer
                del final_model
                gc.collect()
                torch.cuda.empty_cache()
                trainer = None  # 标记已删除
                
                run_evaluations(final_ckpt, exp_name)

        except KeyboardInterrupt:
            print("\n🛑 User Interrupted. Exiting...")
            sys.exit(0)
        except Exception as e:
            print(f"\n❌ Experiment [{exp_name}] Failed!")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # 7. 显存清理 (至关重要)
            if trainer:
                del trainer
            gc.collect()
            torch.cuda.empty_cache()
            print("🧹 GPU Memory Cleared.")

    print("\n" + "="*60)
    print("🎉 All Experiments Finished!")
    print("="*60)

if __name__ == "__main__":
    run_grid_search()
