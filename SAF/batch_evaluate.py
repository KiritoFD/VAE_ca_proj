"""
批量评估脚本
自动扫描 AutoSearch_Results 目录，对所有已完成的实验运行评估并汇总结果
"""
import sys
import json
import torch
import gc
from pathlib import Path
import time
from tqdm import tqdm

# 导入评估模块
from eval_lpips import Evaluator as LPIPSEvaluator
from eval_clip import Evaluator as CLIPEvaluator
from eval_vgg import Evaluator as VGGEvaluator


def find_all_experiments(root_dir="AutoSearch_Results"):
    """
    扫描根目录，找到所有包含 stage1_final.pt 的实验
    """
    root = Path(root_dir)
    if not root.exists():
        print(f"❌ Directory not found: {root}")
        return []
    
    experiments = []
    for exp_dir in root.iterdir():
        if not exp_dir.is_dir():
            continue
        
        ckpt_dir = exp_dir / "checkpoints"
        final_ckpt = ckpt_dir / "stage1_final.pt"
        
        if final_ckpt.exists():
            experiments.append({
                "name": exp_dir.name,
                "exp_dir": exp_dir,
                "ckpt_dir": ckpt_dir,
                "checkpoint": final_ckpt
            })
    
    return experiments


def evaluate_single_experiment(exp_info, config_path="config.json"):
    """
    对单个实验运行所有评估
    """
    exp_name = exp_info["name"]
    ckpt_path = exp_info["checkpoint"]
    ckpt_dir = exp_info["ckpt_dir"]
    
    # 创建评估目录
    eval_dir = ckpt_dir / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"📊 Evaluating: {exp_name}")
    print(f"📂 Checkpoint: {ckpt_path}")
    print(f"💾 Results: {eval_dir}")
    print(f"{'='*60}")
    
    # 读取配置
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    
    ref_dir = cfg.get("data", {}).get("data_root", "").strip('"').strip("'")
    target_dir = cfg.get("inference", {}).get("image_path", "").strip('"').strip("'")
    
    # 汇总结果
    results_summary = {
        "experiment_name": exp_name,
        "checkpoint_path": str(ckpt_path),
        "evaluation_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": {}
    }
    
    # ========== 1. LPIPS Evaluation ==========
    try:
        print("\n🔹 [1/3] Running LPIPS Evaluation...")
        lpips_eval = LPIPSEvaluator(str(ckpt_path), config_path)
        
        # 🟢 修复：直接捕获评估结果
        lpips_eval.evaluate(target_dir, batch_size=2)
        
        # 🟢 从评估器内部属性或生成的文件读取结果
        # 方案1: 检查评估器是否有results属性
        if hasattr(lpips_eval, 'results'):
            lpips_data = lpips_eval.results
        else:
            # 方案2: 读取默认输出的CSV文件
            import pandas as pd
            csv_candidates = [
                Path("lpips_results.csv"),
                Path("eval_results/lpips_results.csv"),
                eval_dir / "lpips_results.csv"
            ]
            
            lpips_data = None
            for csv_path in csv_candidates:
                if csv_path.exists():
                    df = pd.read_csv(csv_path)
                    lpips_data = {
                        "mean": float(df['lpips'].mean()),
                        "std": float(df['lpips'].std()),
                        "per_style": {}
                    }
                    
                    # 按风格统计
                    if 'target_style' in df.columns:
                        for style in df['target_style'].unique():
                            style_df = df[df['target_style'] == style]
                            lpips_data["per_style"][f"style_{style}"] = {
                                "mean": float(style_df['lpips'].mean()),
                                "count": len(style_df)
                            }
                    
                    # 移动到评估目录
                    target_path = eval_dir / "lpips_results.csv"
                    if csv_path != target_path:
                        csv_path.rename(target_path)
                    lpips_data["results_file"] = str(target_path)
                    break
            
            if lpips_data is None:
                lpips_data = {"error": "No output file found"}
        
        results_summary["metrics"]["lpips"] = lpips_data
        
        del lpips_eval
        gc.collect()
        torch.cuda.empty_cache()
        print("✅ LPIPS Complete")
        
    except Exception as e:
        print(f"❌ LPIPS Failed: {e}")
        import traceback
        traceback.print_exc()
        results_summary["metrics"]["lpips"] = {"error": str(e)}
    
    # ========== 2. CLIP Evaluation ==========
    try:
        print("\n🔹 [2/3] Running CLIP Evaluation...")
        clip_eval = CLIPEvaluator(str(ckpt_path), config_path)
        
        clip_eval.evaluate(target_dir, batch_size=2)
        
        import pandas as pd
        csv_candidates = [
            Path("clip_scores.csv"),
            Path("eval_results/clip_scores.csv"),
            eval_dir / "clip_scores.csv"
        ]
        
        clip_data = None
        for csv_path in csv_candidates:
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                clip_data = {
                    "mean_similarity": float(df['clip_similarity'].mean()),
                    "std_similarity": float(df['clip_similarity'].std()),
                    "per_style": {}
                }
                
                if 'target_style' in df.columns:
                    for style in df['target_style'].unique():
                        style_df = df[df['target_style'] == style]
                        clip_data["per_style"][f"style_{style}"] = {
                            "mean": float(style_df['clip_similarity'].mean()),
                            "count": len(style_df)
                        }
                
                target_path = eval_dir / "clip_scores.csv"
                if csv_path != target_path:
                    csv_path.rename(target_path)
                clip_data["results_file"] = str(target_path)
                break
        
        if clip_data is None:
            clip_data = {"error": "No output file found or unrecognized format"}
        
        results_summary["metrics"]["clip"] = clip_data
        
        del clip_eval
        gc.collect()
        torch.cuda.empty_cache()
        print("✅ CLIP Complete")
        
    except Exception as e:
        print(f"❌ CLIP Failed: {e}")
        import traceback
        traceback.print_exc()
        results_summary["metrics"]["clip"] = {"error": str(e)}
    
    # ========== 3. VGG Style Evaluation ==========
    try:
        print("\n🔹 [3/3] Running VGG Style Evaluation...")
        vgg_eval = VGGEvaluator(str(ckpt_path), ref_root=ref_dir, config_path=config_path)
        
        vgg_eval.evaluate(bs=1)
        
        import pandas as pd
        csv_candidates = [
            Path("vgg_style_distances.csv"),
            Path("eval_results/vgg_style_distances.csv"),
            eval_dir / "vgg_style_distances.csv"
        ]
        
        vgg_data = None
        for csv_path in csv_candidates:
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                
                # 🟢 VGG的列名可能不同，检查所有可能的列
                distance_col = None
                for col in ['vgg_distance', 'style_distance', 'gram_distance']:
                    if col in df.columns:
                        distance_col = col
                        break
                
                if distance_col:
                    vgg_data = {
                        "mean_distance": float(df[distance_col].mean()),
                        "std_distance": float(df[distance_col].std()),
                        "per_style": {}
                    }
                    
                    if 'target_style' in df.columns:
                        for style in df['target_style'].unique():
                            style_df = df[df['target_style'] == style]
                            vgg_data["per_style"][f"style_{style}"] = {
                                "mean": float(style_df[distance_col].mean()),
                                "count": len(style_df)
                            }
                    
                    target_path = eval_dir / "vgg_style_distances.csv"
                    if csv_path != target_path:
                        csv_path.rename(target_path)
                    vgg_data["results_file"] = str(target_path)
                    break
        
        if vgg_data is None:
            vgg_data = {"error": "No output file found or unrecognized format"}
        
        results_summary["metrics"]["vgg"] = vgg_data
        
        del vgg_eval
        gc.collect()
        torch.cuda.empty_cache()
        print("✅ VGG Complete")
        
    except Exception as e:
        print(f"❌ VGG Failed: {e}")
        import traceback
        traceback.print_exc()
        results_summary["metrics"]["vgg"] = {"error": str(e)}
    
    # 保存汇总
    summary_path = eval_dir / "metrics_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, indent=4, ensure_ascii=False)
    
    print(f"\n✅ Evaluation complete for {exp_name}")
    print(f"📄 Summary: {summary_path}\n")
    
    return results_summary


def generate_comparison_table(all_results, output_path="AutoSearch_Results/comparison_table.csv"):
    """
    生成所有实验的对比表格
    """
    import pandas as pd
    
    rows = []
    for result in all_results:
        row = {
            "Experiment": result["experiment_name"],
            "Eval_Time": result["evaluation_time"]
        }
        
        # 🟢 改进：更健壮的数据提取
        # LPIPS
        lpips = result["metrics"].get("lpips", {})
        if "mean" in lpips:
            row["LPIPS_Mean"] = f"{lpips['mean']:.5f}"
            row["LPIPS_Std"] = f"{lpips['std']:.5f}"
        elif "error" in lpips:
            row["LPIPS_Mean"] = f"ERROR: {lpips['error']}"
            row["LPIPS_Std"] = "-"
        else:
            row["LPIPS_Mean"] = "NO DATA"
            row["LPIPS_Std"] = "-"
        
        # CLIP
        clip = result["metrics"].get("clip", {})
        if "mean_similarity" in clip:
            row["CLIP_Similarity"] = f"{clip['mean_similarity']:.5f}"
            row["CLIP_Std"] = f"{clip.get('std_similarity', 0):.5f}"
        elif "error" in clip:
            row["CLIP_Similarity"] = f"ERROR: {clip['error']}"
            row["CLIP_Std"] = "-"
        else:
            row["CLIP_Similarity"] = "NO DATA"
            row["CLIP_Std"] = "-"
        
        # VGG
        vgg = result["metrics"].get("vgg", {})
        if "mean_distance" in vgg:
            row["VGG_Distance"] = f"{vgg['mean_distance']:.5f}"
            row["VGG_Std"] = f"{vgg.get('std_distance', 0):.5f}"
        elif "error" in vgg:
            row["VGG_Distance"] = f"ERROR: {vgg['error']}"
            row["VGG_Std"] = "-"
        else:
            row["VGG_Distance"] = "NO DATA"
            row["VGG_Std"] = "-"
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    
    print(f"\n{'='*60}")
    print(f"📊 Comparison Table Generated")
    print(f"📄 Saved to: {output_path}")
    print(f"{'='*60}\n")
    print(df.to_string(index=False))
    
    return df


def main():
    print("="*60)
    print("🔍 Batch Evaluation Script")
    print("="*60)
    
    # 1. 扫描实验
    experiments = find_all_experiments()
    
    if not experiments:
        print("❌ No experiments found with stage1_final.pt")
        return
    
    print(f"\n✅ Found {len(experiments)} completed experiments:")
    for i, exp in enumerate(experiments, 1):
        print(f"  {i}. {exp['name']}")
    
    # 2. 批量评估
    all_results = []
    for exp in tqdm(experiments, desc="Evaluating Experiments"):
        try:
            result = evaluate_single_experiment(exp)
            all_results.append(result)
        except Exception as e:
            print(f"\n❌ Failed to evaluate {exp['name']}: {e}")
            import traceback
            traceback.print_exc()
    
    # 3. 生成对比表
    if all_results:
        generate_comparison_table(all_results)
    
    print("\n" + "="*60)
    print("🎉 Batch Evaluation Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
