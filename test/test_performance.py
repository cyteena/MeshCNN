#!/usr/bin/env python3
"""
简单的性能测试脚本
"""

import torch
import time
import numpy as np

def test_matrix_multiplication_performance():
    """测试矩阵乘法性能"""
    
    # 启用性能优化
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing on device: {device}")
    
    # 测试不同大小的矩阵乘法
    sizes = [(1000, 1000), (2000, 2000), (5000, 1000)]
    
    for m, n in sizes:
        print(f"\nTesting matrix size: {m}x{n}")
        
        # 创建随机矩阵
        a = torch.randn(m, n, device=device)
        b = torch.randn(n, m, device=device)
        
        # 预热
        for _ in range(5):
            _ = torch.matmul(a, b)
        
        # 性能测试
        start_time = time.time()
        num_iterations = 100
        
        for _ in range(num_iterations):
            result = torch.matmul(a, b)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.time()
        
        avg_time = (end_time - start_time) / num_iterations
        print(f"Average time per multiplication: {avg_time:.4f} seconds")
        
        # 测试数据类型转换性能
        print("Testing data type conversion...")
        start_time = time.time()
        
        for _ in range(num_iterations):
            a_fp16 = a.to(dtype=torch.float16)
            result = torch.matmul(a_fp16, b.to(dtype=torch.float16))
            result = result.to(dtype=torch.float32)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.time()
        
        avg_time_fp16 = (end_time - start_time) / num_iterations
        print(f"Average time with FP16: {avg_time_fp16:.4f} seconds")
        
        speedup = avg_time / avg_time_fp16
        print(f"Speedup: {speedup:.2f}x")

if __name__ == "__main__":
    test_matrix_multiplication_performance()
