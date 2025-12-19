"""
运行对比实验：有/无 PageRank 的准确率对比。

用途：
- 运行无 PageRank 的训练（baseline）
- 运行有 PageRank 的训练（with PageRank）
- 从日志中提取准确率并生成对比报告
"""

import subprocess
import sys
import json
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime

def extract_accuracies_from_output(output: str) -> List[float]:
    """从控制台输出中提取准确率"""
    accuracies = []
    # 匹配 "Global Model Accuracy: XX.XX%"
    pattern = r"Global Model Accuracy:\s*(\d+\.\d+)%"
    matches = re.findall(pattern, output)
    for match in matches:
        accuracies.append(float(match) / 100.0)  # 转换为0-1之间的值
    return accuracies

def extract_accuracies_from_logs(log_file: Path) -> List[float]:
    """从日志文件中提取准确率"""
    accuracies = []
    if not log_file.exists():
        return accuracies
    
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    event = json.loads(line.strip())
                    if event.get('event_type') == 'GLOBAL_EVAL':
                        details = event.get('details', {})
                        if 'global_accuracy' in details:
                            accuracies.append(float(details['global_accuracy']))
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"Warning: Could not read log file {log_file}: {e}")
    
    return accuracies

def run_training(script_name: str, description: str) -> Tuple[bool, str, List[float]]:
    """运行训练脚本并提取准确率"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Script: {script_name}")
    print(f"{'='*60}\n")
    
    script_path = Path(__file__).parent / "Serverlesscase" / script_name
    
    if not script_path.exists():
        print(f"Error: Script not found: {script_path}")
        return False, "", []
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour timeout
        )
        
        print("STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        # 从控制台输出提取准确率
        accuracies = extract_accuracies_from_output(result.stdout)
        
        # 如果控制台没有提取到，尝试从日志文件提取
        if not accuracies:
            log_file = Path(__file__).parent.parent / "logs" / "blockchain_log.jsonl"
            accuracies = extract_accuracies_from_logs(log_file)
        
        success = result.returncode == 0
        return success, result.stdout, accuracies
    except subprocess.TimeoutExpired:
        print(f"Error: Training script timed out after 1 hour")
        return False, "", []
    except Exception as e:
        print(f"Error running script: {e}")
        return False, "", []

def generate_comparison_report(
    baseline_accuracies: List[float],
    pagerank_accuracies: List[float],
    output_file: Path
):
    """生成对比报告"""
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("PageRank 对比实验报告")
    report_lines.append("=" * 80)
    report_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    # Baseline 结果
    report_lines.append("【Baseline (无 PageRank)】")
    if baseline_accuracies:
        report_lines.append(f"  准确率列表: {[f'{acc*100:.2f}%' for acc in baseline_accuracies]}")
        report_lines.append(f"  Round 1: {baseline_accuracies[0]*100:.2f}%")
        if len(baseline_accuracies) > 1:
            report_lines.append(f"  Round 2: {baseline_accuracies[1]*100:.2f}%")
        if len(baseline_accuracies) > 2:
            report_lines.append(f"  Round 3: {baseline_accuracies[2]*100:.2f}%")
        report_lines.append(f"  最终准确率: {baseline_accuracies[-1]*100:.2f}%")
    else:
        report_lines.append("  未找到准确率数据")
    report_lines.append("")
    
    # PageRank 结果
    report_lines.append("【With PageRank (有 PageRank)】")
    if pagerank_accuracies:
        report_lines.append(f"  准确率列表: {[f'{acc*100:.2f}%' for acc in pagerank_accuracies]}")
        report_lines.append(f"  Round 1: {pagerank_accuracies[0]*100:.2f}%")
        if len(pagerank_accuracies) > 1:
            report_lines.append(f"  Round 2: {pagerank_accuracies[1]*100:.2f}%")
        if len(pagerank_accuracies) > 2:
            report_lines.append(f"  Round 3: {pagerank_accuracies[2]*100:.2f}%")
        report_lines.append(f"  最终准确率: {pagerank_accuracies[-1]*100:.2f}%")
    else:
        report_lines.append("  未找到准确率数据")
    report_lines.append("")
    
    # 对比分析
    report_lines.append("【对比分析】")
    if baseline_accuracies and pagerank_accuracies:
        min_len = min(len(baseline_accuracies), len(pagerank_accuracies))
        for i in range(min_len):
            diff = pagerank_accuracies[i] - baseline_accuracies[i]
            diff_pct = diff * 100
            symbol = "↑" if diff > 0 else "↓" if diff < 0 else "="
            report_lines.append(f"  Round {i+1}: {symbol} {abs(diff_pct):.2f}% ({baseline_accuracies[i]*100:.2f}% → {pagerank_accuracies[i]*100:.2f}%)")
        
        final_diff = pagerank_accuracies[-1] - baseline_accuracies[-1]
        final_diff_pct = final_diff * 100
        symbol = "↑" if final_diff > 0 else "↓" if final_diff < 0 else "="
        report_lines.append(f"  最终准确率变化: {symbol} {abs(final_diff_pct):.2f}%")
        
        if final_diff > 0:
            report_lines.append(f"  ✅ PageRank 提升了模型准确率")
        elif final_diff < 0:
            report_lines.append(f"  ⚠️  PageRank 降低了模型准确率")
        else:
            report_lines.append(f"  ➡️  PageRank 对准确率影响不大")
    else:
        report_lines.append("  无法进行对比（缺少数据）")
    report_lines.append("")
    
    report_lines.append("=" * 80)
    
    # 写入文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    # 打印报告
    print('\n'.join(report_lines))
    print(f"\n对比报告已保存到: {output_file}")

def main():
    print("="*60)
    print("PageRank Comparison Experiment")
    print("="*60)
    
    root_dir = Path(__file__).parent.parent
    
    # 1. 运行无 PageRank 的训练（baseline）
    print("\n[Step 1/2] Running baseline training (without PageRank)...")
    baseline_success, baseline_output, baseline_accuracies = run_training(
        "Serverless_NonIID_Medical_transcriptions.py",
        "Baseline Training (without PageRank)"
    )
    
    if not baseline_success:
        print("Warning: Baseline training failed or had errors")
    
    # 2. 运行有 PageRank 的训练
    print("\n[Step 2/2] Running training with PageRank...")
    pagerank_success, pagerank_output, pagerank_accuracies = run_training(
        "Serverless_NonIID_Medical_transcriptions_with_pagerank.py",
        "Training with PageRank"
    )
    
    if not pagerank_success:
        print("Warning: PageRank training failed or had errors")
    
    # 3. 生成对比报告
    print("\n" + "="*60)
    print("Generating Comparison Report...")
    print("="*60)
    
    report_file = root_dir / "pagerank_comparison_report.txt"
    generate_comparison_report(
        baseline_accuracies,
        pagerank_accuracies,
        report_file
    )
    
    # 4. 总结
    print("\n" + "="*60)
    print("Experiment Summary")
    print("="*60)
    print(f"Baseline (without PageRank): {'✓ Success' if baseline_success else '✗ Failed'}")
    print(f"With PageRank: {'✓ Success' if pagerank_success else '✗ Failed'}")
    print("\nOutput files:")
    print(f"  - {report_file}: Comparison report")
    print("  - logs/blockchain_log.jsonl: Training logs")
    print("  - pagerank_scores.txt: PageRank scores")
    print("\nNext steps:")
    print("  1. Check the comparison report for accuracy comparison")
    print("  2. Take screenshots of console output showing client removal logs")
    print("  3. Verify pagerank_scores.txt contains PageRank scores")


if __name__ == "__main__":
    main()

