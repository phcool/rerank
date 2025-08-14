#!/usr/bin/env python3
"""
从大的parquet文件中提取小部分数据用于测试
"""
import pandas as pd
import os
import argparse
from pathlib import Path


def extract_small_dataset(input_file, output_file, num_samples=10, random_seed=42):
    """
    从输入文件中提取指定数量的样本
    
    Args:
        input_file: 输入的parquet文件路径
        output_file: 输出的parquet文件路径
        num_samples: 要提取的样本数量
        random_seed: 随机种子
    """
    print(f"正在读取文件: {input_file}")
    
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"❌ 输入文件不存在: {input_file}")
        return False
    
    try:
        # 读取parquet文件
        df = pd.read_parquet(input_file)
        print(f"✓ 原始数据集大小: {len(df)} 条记录")
        print(f"✓ 数据列: {list(df.columns)}")
        
        # 显示前几行数据信息
        print(f"\n数据预览:")
        for i, row in df.head(3).iterrows():
            print(f"  样本 {i+1}:")
            if 'query' in row:
                print(f"    查询: {row['query'][:100]}...")
            if 'prompt' in row:
                print(f"    Prompt长度: {len(str(row['prompt']))} 字符")
            print()
        
        # 随机采样
        if len(df) <= num_samples:
            print(f"⚠️ 原始数据量({len(df)})小于等于请求的样本数({num_samples})，使用全部数据")
            sampled_df = df
        else:
            print(f"正在随机采样 {num_samples} 条记录...")
            sampled_df = df.sample(n=num_samples, random_state=random_seed)
        
        # 创建输出目录
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # 保存采样后的数据
        sampled_df.to_parquet(output_file, index=False)
        print(f"✓ 已保存 {len(sampled_df)} 条记录到: {output_file}")
        
        # 显示文件大小对比
        input_size = os.path.getsize(input_file) / (1024 * 1024)  # MB
        output_size = os.path.getsize(output_file) / (1024 * 1024)  # MB
        print(f"✓ 文件大小: {input_size:.2f}MB -> {output_size:.2f}MB")
        
        return True
        
    except Exception as e:
        print(f"❌ 处理文件时出错: {e}")
        return False


def find_parquet_files(data_dir):
    """查找数据目录中的parquet文件"""
    parquet_files = []
    
    if os.path.exists(data_dir):
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.endswith('.parquet'):
                    parquet_files.append(os.path.join(root, file))
    
    return parquet_files


def main():
    parser = argparse.ArgumentParser(description='从parquet文件中提取小部分数据用于测试')
    parser.add_argument('--input', '-i', type=str, help='输入的parquet文件路径')
    parser.add_argument('--output', '-o', type=str, help='输出的parquet文件路径')
    parser.add_argument('--samples', '-n', type=int, default=2, help='要提取的样本数量 (默认: 2)')
    parser.add_argument('--seed', '-s', type=int, default=42, help='随机种子 (默认: 42)')
    parser.add_argument('--list', '-l', action='store_true', help='列出data目录中的所有parquet文件')
    
    args = parser.parse_args()
    
    # 如果指定了--list，列出所有parquet文件
    if args.list:
        print("正在查找data目录中的parquet文件...")
        data_dir = "data"
        parquet_files = find_parquet_files(data_dir)
        
        if parquet_files:
            print(f"找到 {len(parquet_files)} 个parquet文件:")
            for i, file in enumerate(parquet_files, 1):
                size = os.path.getsize(file) / (1024 * 1024)  # MB
                print(f"  {i}. {file} ({size:.2f}MB)")
        else:
            print("未找到任何parquet文件")
        return
    
    # 如果没有指定输入文件，尝试自动查找
    if not args.input:
        print("未指定输入文件，正在自动查找...")
        data_dir = "data"
        parquet_files = find_parquet_files(data_dir)
        
        if not parquet_files:
            print("❌ 未找到任何parquet文件")
            print("请使用 --list 查看可用文件，或使用 --input 指定文件路径")
            return
        
        # 优先选择elimination_sort文件
        elimination_files = [f for f in parquet_files if 'elimination_sort' in f]
        if elimination_files:
            args.input = elimination_files[0]
            print(f"✓ 自动选择: {args.input}")
        else:
            args.input = parquet_files[0]
            print(f"✓ 自动选择: {args.input}")
    
    # 如果没有指定输出文件，自动生成
    if not args.output:
        input_path = Path(args.input)
        output_name = input_path.stem + f"_small_{args.samples}" + input_path.suffix
        args.output = str(input_path.parent / output_name)
        print(f"✓ 自动生成输出文件名: {args.output}")
    
    # 执行提取
    print(f"\n开始提取数据:")
    print(f"  输入文件: {args.input}")
    print(f"  输出文件: {args.output}")
    print(f"  样本数量: {args.samples}")
    print(f"  随机种子: {args.seed}")
    print()
    
    success = extract_small_dataset(
        input_file=args.input,
        output_file=args.output,
        num_samples=args.samples,
        random_seed=args.seed
    )
    
    if success:
        print(f"\n✅ 数据提取完成!")
        print(f"可以使用以下命令测试:")
        print(f"  python test_elimination_sort.py")
        print(f"  # 或修改训练脚本使用: {args.output}")
    else:
        print(f"\n❌ 数据提取失败!")


if __name__ == "__main__":
    main()
