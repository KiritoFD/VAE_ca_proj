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
    
    # -------------------------------------------------------------------------
    # 第一组：基准 (Baseline)
    # -------------------------------------------------------------------------
    {
        "name": "Exp1_Baseline",
        "description": "【基准线】各项参数中庸，用于对比其他实验的提升幅度。",
        "training": {
            "learning_rate": 1e-4,
            "transfer_loss_weight": 5.0,
            "stage1_epochs": 100,
        }
    },

    # -------------------------------------------------------------------------
    # 第二组：激进风格 (Aggressive Style)
    # 验证：是否只有加大惩罚权重，才能逼出明显的照片风格？
    # 风险：画面可能出现高频噪点或色彩溢出。
    # -------------------------------------------------------------------------
    {
        "name": "Exp2_HighForce",
        "description": "【高压策略】20倍权重，强迫模型大幅度修改原图。LR略降防止跑飞。",
        "training": {
            "learning_rate": 8e-5,
            "transfer_loss_weight": 20.0, 
            "stage1_epochs": 120,
        }
    },

    # -------------------------------------------------------------------------
    # 第三组：精细打磨 (Precision Mode) - 重点关注！
    # 验证：之前的褪色/模糊是否因为步子太大？小步慢跑能否画出高清细节？
    # 预期：LPIPS 分数应该最低（最好），但训练最慢。
    # -------------------------------------------------------------------------
    {
        "name": "Exp3_SlowCook",
        "description": "【慢工细活】极低LR + 长Epoch + 高权重。旨在解决褪色和模糊。",
        "training": {
            "learning_rate": 2e-5,       # 只有基准的 1/5
            "transfer_loss_weight": 15.0, # 权重较高，保证方向
            "stage1_epochs": 300,         # 时间换质量
        }
    },

    # -------------------------------------------------------------------------
    # 第四组：快速收敛 (Fast Convergence)
    # 验证：模型是否其实前50轮就学完了？是不是后面都在过拟合？
    # -------------------------------------------------------------------------
    {
        "name": "Exp4_SpeedRun",
        "description": "【极速版】高LR + 低Epoch。测试模型的学习上限速度。",
        "training": {
            "learning_rate": 2e-4,       # 基准的 2 倍
            "transfer_loss_weight": 8.0,
            "stage1_epochs": 80,
        }
    },

    # -------------------------------------------------------------------------
    # 第五组：极端权重测试 (Stress Test)
    # 验证：如果给 50 倍权重，模型是会画出完美的照片，还是会彻底崩坏成噪声？
    # 目的：寻找权重的“崩溃临界点”。
    # -------------------------------------------------------------------------
    {
        "name": "Exp5_WeightStress",
        "description": "【压力测试】50倍权重。探索模型的鲁棒性边界。",
        "training": {
            "learning_rate": 5e-5,
            "transfer_loss_weight": 50.0, # 极端的惩罚
            "stage1_epochs": 100,
        }
    },

    # -------------------------------------------------------------------------
    # 第六组：松弛控制 (Relaxed Control)
    # 验证：如果只给一点点压力，模型是否会保留更多原图结构（Identity）但画质更自然？
    # -------------------------------------------------------------------------
    {
        "name": "Exp6_Gentle",
        "description": "【微调模式】低权重。测试是否能仅改变光影而不破坏结构。",
        "training": {
            "learning_rate": 1e-4,
            "transfer_loss_weight": 2.0,  # 非常温和
            "stage1_epochs": 150,
        }
    },

    # -------------------------------------------------------------------------
    # 第七组：大 Batch Size (High Stability)
    # 验证：显存允许的情况下，更大的 Batch Size 是否能带来更稳定的梯度下降？
    # 注意：4070 8G 跑 BS=96 可能会 OOM，如果炸了请跳过。
    # -------------------------------------------------------------------------
    {
        "name": "Exp7_HighBS",
        "description": "【高稳定性】大Batch Size。梯度估计更准，理论上色彩更正。",
        "training": {
            "learning_rate": 1e-4,
            "transfer_loss_weight": 10.0,
            "batch_size": 80,             # 挑战显存极限
            "stage1_epochs": 120,
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
    import json
    
    # 🟢 创建统一的评估结果目录
    eval_dir = Path(exp_ckpt_dir) / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print(f"📊 Running Evaluations for: {exp_name}")
    print(f"📂 Results will be saved to: {eval_dir}")
    print("="*60)
    
    # 读取配置获取参考目录
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    
    ref_dir = cfg.get("data", {}).get("data_root", None)
    if ref_dir:
        ref_dir = ref_dir.strip('"').strip("'")
    
    target_dir = cfg.get("inference", {}).get("image_path", "").strip('"').strip("'")
    
    # 🟢 汇总结果字典
    results_summary = {
        "experiment_name": exp_name,
        "checkpoint_path": str(ckpt_path),
        "evaluation_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": {}
    }
    
    # 1. LPIPS Evaluation
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
    try:
        print("\n🔹 [3/3] Running VGG Style Evaluation...")
        vgg_eval = VGGEvaluator(str(ckpt_path), ref_root=ref_dir, config_path=config_path)
        # 🟢 修改：传入保存路径
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
        
        # 🟢 新增：如果发现旧格式 checkpoint，清理掉防止冲突
        if ckpt_dir.exists():
            old_ckpts = list(ckpt_dir.glob("stage1_epoch*.pt"))
            if old_ckpts:
                # 检查第一个是否为旧格式
                try:
                    test_ckpt = torch.load(old_ckpts[0], map_location='cpu')
                    if 'model_state_dict' not in test_ckpt:
                        print(f"⚠️  Found old format checkpoints in {ckpt_dir.name}")
                        print(f"🗑️  Cleaning up {len(old_ckpts)} old checkpoints...")
                        for old in old_ckpts:
                            old.unlink()
                        print("✅ Cleanup complete")
                except:
                    pass
        
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
            
            # 5. 🟢 修复：强制执行一次最终推理
            print("🎨 Running Final Inference...")
            final_ckpt = trainer.ckpt_dir / "stage1_final.pt"
            if final_ckpt.exists():
                final_model = trainer.get_model()
                
                # 🟢 正确加载：先读取checkpoint，提取model_state_dict
                ckpt_data = torch.load(final_ckpt, map_location=trainer.device)
                if 'model_state_dict' in ckpt_data:
                    trainer.safe_load(final_model, ckpt_data['model_state_dict'])
                else:
                    trainer.safe_load(final_model, ckpt_data)
                
                trainer.do_inference(final_model, "final", "stage1_final")
                
                # 清理
                del final_model
                gc.collect()
                torch.cuda.empty_cache()
            
            print(f"✅ Experiment [{exp_name}] Training Completed in {(time.time() - start_time)/60:.1f} mins.")
            
            # 🟢 6. 运行评估脚本 - 传入 ckpt_dir
            if final_ckpt.exists():
                # 先清理训练器释放显存
                del trainer
                gc.collect()
                torch.cuda.empty_cache()
                trainer = None  # 标记已删除
                
                run_evaluations(final_ckpt, exp_name, ckpt_dir)

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
