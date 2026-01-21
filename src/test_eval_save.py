"""
测试评估器的保存功能
验证三个指标是否正确保存为 JSON 文件
"""
import json
from pathlib import Path
import sys

def test_evaluation_files(eval_dir):
    """
    检查评估目录中的文件
    """
    eval_dir = Path(eval_dir)
    
    print("\n" + "="*60)
    print(f"📂 检查评估目录: {eval_dir}")
    print("="*60)
    
    if not eval_dir.exists():
        print("❌ 评估目录不存在！")
        return False
    
    # 检查三个必需的文件
    required_files = [
        "lpips_results.json",
        "clip_results.json", 
        "vgg_results.json",
        "metrics_summary.json"
    ]
    
    all_good = True
    
    for filename in required_files:
        filepath = eval_dir / filename
        if filepath.exists():
            print(f"✅ {filename} 存在")
            
            # 读取并验证 JSON 格式
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                # 验证数据结构
                if "metric" in data:
                    print(f"   └─ Metric: {data.get('metric')}")
                    if "overall_average" in data:
                        print(f"   └─ Overall averages: {list(data['overall_average'].keys())}")
                elif "metrics" in data:
                    print(f"   └─ Contains metrics: {list(data['metrics'].keys())}")
                    
            except json.JSONDecodeError as e:
                print(f"   ❌ JSON 格式错误: {e}")
                all_good = False
            except Exception as e:
                print(f"   ⚠️  读取错误: {e}")
        else:
            print(f"❌ {filename} 不存在！")
            all_good = False
    
    print("="*60)
    
    if all_good:
        print("✅ 所有评估文件都正确保存！")
        
        # 显示汇总信息
        summary_path = eval_dir / "metrics_summary.json"
        if summary_path.exists():
            with open(summary_path, 'r', encoding='utf-8') as f:
                summary = json.load(f)
            
            print("\n📊 评估汇总：")
            print(f"实验名称: {summary.get('experiment_name', 'N/A')}")
            print(f"评估时间: {summary.get('evaluation_time', 'N/A')}")
            
            metrics = summary.get('metrics', {})
            for metric_name, metric_data in metrics.items():
                if isinstance(metric_data, dict) and 'overall_average' in metric_data:
                    print(f"\n{metric_name.upper()}:")
                    for style, values in metric_data['overall_average'].items():
                        if 'mean' in values:
                            print(f"  {style}: {values['mean']:.6f}")
    else:
        print("❌ 部分评估文件缺失或损坏！")
    
    return all_good


def find_latest_evaluation():
    """
    查找最新的评估结果目录
    """
    results_root = Path("AutoSearch_Results")
    
    if not results_root.exists():
        print("❌ AutoSearch_Results 目录不存在")
        return None
    
    # 查找所有实验的 evaluation 目录
    eval_dirs = list(results_root.glob("*/checkpoints/evaluation"))
    
    if not eval_dirs:
        print("❌ 未找到任何评估目录")
        return None
    
    # 按修改时间排序，返回最新的
    eval_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    
    print(f"\n找到 {len(eval_dirs)} 个评估目录")
    print(f"最新的评估目录: {eval_dirs[0]}")
    
    return eval_dirs[0]


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # 指定目录
        test_dir = sys.argv[1]
        test_evaluation_files(test_dir)
    else:
        # 自动查找最新的
        latest = find_latest_evaluation()
        if latest:
            test_evaluation_files(latest)
        else:
            print("\n使用方法:")
            print("  python test_eval_save.py [评估目录路径]")
            print("  或者不带参数自动查找最新的评估结果")
