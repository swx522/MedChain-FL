"""
快速验证 PageRank 集成是否正常工作。

用途：
- 检查所有必需文件是否存在
- 验证 PageRank 函数是否正常工作
- 检查日志文件格式
- 生成验证报告
"""

import sys
from pathlib import Path
from typing import Tuple
import json

def check_file_exists(file_path: Path, description: str) -> Tuple[bool, str]:
    """检查文件是否存在"""
    if file_path.exists():
        return True, f"✓ {description}: {file_path}"
    else:
        return False, f"✗ {description}: {file_path} (不存在)"

def check_pagerank_import():
    """检查 PageRank 模块是否可以导入"""
    try:
        # 由于本脚本位于 src 目录下，直接使用同级模块导入
        from pagerank_utils import compute_pagerank_scores, sort_scores_desc
        from graph_extractor import extract_graph_from_rounds
        return True, "✓ PageRank 模块导入成功"
    except ImportError as e:
        return False, f"✗ PageRank 模块导入失败: {e}"

def check_pagerank_functionality():
    """测试 PageRank 函数是否正常工作"""
    try:
        # 同级模块导入
        from pagerank_utils import compute_pagerank_scores
        
        # 简单测试
        clients = [0, 1, 2, 3]
        edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
        scores = compute_pagerank_scores(clients, edges)
        
        if len(scores) == 4 and all(0 <= score <= 1 for score in scores.values()):
            return True, f"✓ PageRank 函数测试通过 (分数: {scores})"
        else:
            return False, f"✗ PageRank 函数返回异常结果: {scores}"
    except Exception as e:
        return False, f"✗ PageRank 函数测试失败: {e}"

def check_log_format(log_file: Path) -> Tuple[bool, str]:
    """检查日志文件格式"""
    if not log_file.exists():
        return False, f"✗ 日志文件不存在: {log_file}"
    
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            if len(lines) == 0:
                return True, "✓ 日志文件为空（正常，如果还没运行训练）"
            
            # 检查前几行是否是有效的 JSON
            valid_count = 0
            for line in lines[:5]:
                try:
                    json.loads(line.strip())
                    valid_count += 1
                except json.JSONDecodeError:
                    pass
            
            if valid_count > 0:
                return True, f"✓ 日志文件格式正确 ({valid_count}/{min(5, len(lines))} 行有效)"
            else:
                return False, "✗ 日志文件格式错误（无法解析 JSON）"
    except Exception as e:
        return False, f"✗ 读取日志文件失败: {e}"

def main():
    print("=" * 80)
    print("PageRank 集成验证")
    print("=" * 80)
    print()
    
    root_dir = Path(__file__).parent.parent
    results = []
    
    # 1. 检查必需文件
    print("【1/4】检查必需文件...")
    files_to_check = [
        (root_dir / "src" / "pagerank_utils.py", "PageRank 工具模块"),
        (root_dir / "src" / "graph_extractor.py", "图提取模块"),
        (root_dir / "src" / "Serverlesscase" / "Serverless_NonIID_Medical_transcriptions.py", "原始训练脚本"),
        (root_dir / "src" / "Serverlesscase" / "Serverless_NonIID_Medical_transcriptions_with_pagerank.py", "集成 PageRank 的训练脚本"),
        (root_dir / "src" / "run_comparison.py", "对比脚本"),
    ]
    
    for file_path, description in files_to_check:
        exists, msg = check_file_exists(file_path, description)
        print(f"  {msg}")
        results.append(exists)
    print()
    
    # 2. 检查模块导入
    print("【2/4】检查模块导入...")
    import_ok, import_msg = check_pagerank_import()
    print(f"  {import_msg}")
    results.append(import_ok)
    print()
    
    # 3. 检查 PageRank 功能
    print("【3/4】检查 PageRank 功能...")
    func_ok, func_msg = check_pagerank_functionality()
    print(f"  {func_msg}")
    results.append(func_ok)
    print()
    
    # 4. 检查日志文件
    print("【4/4】检查日志文件...")
    log_file = root_dir / "logs" / "blockchain_log.jsonl"
    log_ok, log_msg = check_log_format(log_file)
    print(f"  {log_msg}")
    results.append(log_ok)
    print()
    
    # 总结
    print("=" * 80)
    print("验证结果总结")
    print("=" * 80)
    
    all_passed = all(results)
    if all_passed:
        print("✅ 所有检查通过！PageRank 集成正常。")
        print("\n下一步：")
        print("  1. 运行训练脚本测试：python src/Serverlesscase/Serverless_NonIID_Medical_transcriptions_with_pagerank.py")
        print("  2. 检查输出文件：pagerank_scores.txt")
        print("  3. 检查控制台是否有剔除日志（如 'client X removed'）")
        print("  4. 运行对比实验：python src/run_comparison.py")
    else:
        print("⚠️  部分检查未通过，请检查上述错误。")
        failed_count = sum(1 for r in results if not r)
        print(f"   失败项数: {failed_count}/{len(results)}")
    
    print()

if __name__ == "__main__":
    main()

