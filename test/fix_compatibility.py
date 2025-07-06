#!/usr/bin/env python3
"""
RTX 5090 + PyTorch 1.3.1 兼容性解决方案
"""

import torch
import time

def apply_compatibility_fixes():
    """应用兼容性修复"""
    print("=== 应用兼容性修复 ===")
    
    if not torch.cuda.is_available():
        print("CUDA不可用")
        return False
    
    # 1. 设置计算模式
    print("设置计算模式...")
    try:
        # 对于新GPU，可能需要特殊设置
        torch.backends.cudnn.benchmark = False  # 关闭自动调优
        torch.backends.cudnn.deterministic = True  # 使用确定性算法
        print("✓ 计算模式设置完成")
    except Exception as e:
        print(f"✗ 计算模式设置失败: {e}")
    
    # 2. 强制使用特定的CUDA设备
    print("设置CUDA设备...")
    try:
        torch.cuda.set_device(0)
        print("✓ CUDA设备设置完成")
    except Exception as e:
        print(f"✗ CUDA设备设置失败: {e}")
    
    # 3. 预热GPU
    print("预热GPU...")
    try:
        # 从小到大逐步预热
        for size in [2, 5, 10, 20, 30]:
            a = torch.randn(size, size).cuda()
            b = torch.randn(size, size).cuda()
            # 使用CPU计算后复制到GPU，避免cuBLAS
            c_cpu = torch.matmul(a.cpu(), b.cpu())
            c_gpu = c_cpu.cuda()
            
        print("✓ GPU预热完成")
        return True
    except Exception as e:
        print(f"✗ GPU预热失败: {e}")
        return False

def fallback_matmul(a, b):
    """回退的矩阵乘法实现"""
    try:
        # 尝试直接GPU计算
        return torch.matmul(a, b)
    except RuntimeError as e:
        if "CUBLAS_STATUS_EXECUTION_FAILED" in str(e):
            print(f"Warning: GPU matmul failed, falling back to CPU")
            # 回退到CPU计算
            a_cpu = a.cpu()
            b_cpu = b.cpu()
            result_cpu = torch.matmul(a_cpu, b_cpu)
            return result_cpu.to(a.device)
        else:
            raise e

def test_fixed_operations():
    """测试修复后的操作"""
    print("\n=== 测试修复后的操作 ===")
    
    sizes = [50, 100, 200, 500, 750]  # MeshCNN相关的大小
    
    for size in sizes:
        try:
            print(f"测试 {size}x{size} 矩阵乘法...")
            
            a = torch.randn(size, size).cuda()
            b = torch.randn(size, size).cuda()
            
            start = time.time()
            c = fallback_matmul(a, b)
            end = time.time()
            
            print(f"✓ {size}x{size} 成功 - 时间: {end-start:.4f}s")
            
        except Exception as e:
            print(f"✗ {size}x{size} 失败: {e}")

def patch_mesh_union():
    """为mesh_union.py创建补丁"""
    print("\n=== 创建mesh_union.py补丁 ===")
    
    patch_code = '''
# 在 mesh_union.py 的 rebuild_features_average 方法中替换 torch.matmul 调用

def safe_matmul(a, b):
    """安全的矩阵乘法，处理CUBLAS错误"""
    try:
        return torch.matmul(a, b)
    except RuntimeError as e:
        if "CUBLAS_STATUS_EXECUTION_FAILED" in str(e):
            print("Warning: CUBLAS error, falling back to CPU computation")
            a_cpu = a.cpu()
            b_cpu = b.cpu()
            result_cpu = torch.matmul(a_cpu, b_cpu)
            return result_cpu.to(a.device)
        else:
            raise e

# 在 rebuild_features_average 中将：
# fe = torch.matmul(features_squeezed, self.groups)
# 替换为：
# fe = safe_matmul(features_squeezed, self.groups)
'''
    
    with open('mesh_union_patch.py', 'w') as f:
        f.write(patch_code)
    
    print("✓ 补丁文件已创建: mesh_union_patch.py")

def main():
    """主函数"""
    print("RTX 5090 + PyTorch 1.3.1 兼容性解决方案")
    print("=" * 50)
    
    # 应用修复
    success = apply_compatibility_fixes()
    
    if success:
        # 测试修复后的操作
        test_fixed_operations()
    
    # 创建补丁
    patch_mesh_union()
    
    print("\n=== 总结 ===")
    print("主要问题：RTX 5090 (计算能力 12.0) 与 PyTorch 1.3.1 的 cuBLAS 不兼容")
    print("解决方案：")
    print("1. 在出现 CUBLAS 错误时自动回退到 CPU 计算")
    print("2. 关闭 cuDNN 自动调优以避免不稳定的算法")
    print("3. 使用确定性算法")
    print("4. 考虑升级到更新的 PyTorch 版本")
    print("\n建议：")
    print("- 短期：使用 CPU 回退方案")
    print("- 长期：升级到 PyTorch 1.8+ 以获得更好的新GPU支持")

if __name__ == "__main__":
    main()
