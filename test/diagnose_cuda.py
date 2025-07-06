#!/usr/bin/env python3
"""
CUDA 环境诊断工具
"""

import torch
import time

def diagnose_cuda_environment():
    """诊断CUDA环境"""
    print("=== CUDA 环境诊断 ===")
    
    # 1. 基本信息
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    
    if not torch.cuda.is_available():
        print("CUDA不可用，退出诊断")
        return
    
    print(f"CUDA设备数量: {torch.cuda.device_count()}")
    print(f"当前设备: {torch.cuda.current_device()}")
    print(f"设备名称: {torch.cuda.get_device_name(0)}")
    
    # 2. 设备属性
    props = torch.cuda.get_device_properties(0)
    print(f"GPU内存: {props.total_memory / 1024 / 1024:.0f} MB")
    print(f"多处理器数量: {props.multi_processor_count}")
    print(f"计算能力: {props.major}.{props.minor}")
    
    # 3. 测试基本张量创建
    print("\n=== 测试基本张量操作 ===")
    
    try:
        print("创建CPU张量...")
        cpu_tensor = torch.randn(10, 10)
        print("✓ CPU张量创建成功")
        
        print("创建GPU张量...")
        gpu_tensor = torch.randn(10, 10).cuda()
        print("✓ GPU张量创建成功")
        
        print("测试GPU张量运算...")
        result = gpu_tensor + gpu_tensor
        print("✓ GPU张量运算成功")
        
        print("测试CPU到GPU转换...")
        transferred = cpu_tensor.cuda()
        print("✓ CPU到GPU转换成功")
        
        print("测试GPU到CPU转换...")
        back_to_cpu = transferred.cpu()
        print("✓ GPU到CPU转换成功")
        
    except Exception as e:
        print(f"✗ 基本张量操作失败: {e}")
        return
    
    # 4. 测试矩阵乘法
    print("\n=== 测试矩阵乘法 ===")
    
    sizes = [2, 5, 10, 50, 100]
    
    for size in sizes:
        try:
            print(f"测试 {size}x{size} 矩阵乘法...")
            
            # CPU版本
            a_cpu = torch.randn(size, size)
            b_cpu = torch.randn(size, size)
            start = time.time()
            c_cpu = torch.matmul(a_cpu, b_cpu)
            cpu_time = time.time() - start
            
            # GPU版本
            a_gpu = a_cpu.cuda()
            b_gpu = b_cpu.cuda()
            start = time.time()
            c_gpu = torch.matmul(a_gpu, b_gpu)
            torch.cuda.synchronize()
            gpu_time = time.time() - start
            
            print(f"✓ {size}x{size} - CPU: {cpu_time:.6f}s, GPU: {gpu_time:.6f}s")
            
        except Exception as e:
            print(f"✗ {size}x{size} 矩阵乘法失败: {e}")
            break
    
    # 5. 测试不同数据类型
    print("\n=== 测试不同数据类型 ===")
    
    dtypes = [torch.float32, torch.float16, torch.int32, torch.int64]
    
    for dtype in dtypes:
        try:
            print(f"测试数据类型 {dtype}...")
            tensor = torch.randn(5, 5, dtype=dtype).cuda()
            print(f"✓ {dtype} 成功")
        except Exception as e:
            print(f"✗ {dtype} 失败: {e}")
    
    # 6. 内存测试
    print("\n=== 内存测试 ===")
    
    try:
        print("测试内存分配...")
        tensors = []
        for i in range(10):
            tensor = torch.randn(1000, 1000).cuda()
            tensors.append(tensor)
        
        print(f"✓ 分配了 {len(tensors)} 个 1000x1000 张量")
        
        # 清理内存
        del tensors
        torch.cuda.empty_cache()
        print("✓ 内存清理成功")
        
    except Exception as e:
        print(f"✗ 内存测试失败: {e}")

def test_specific_operations():
    """测试特定操作"""
    print("\n=== 测试特定操作 ===")
    
    if not torch.cuda.is_available():
        return
    
    # 测试类似MeshCNN的操作
    try:
        print("测试特征张量操作...")
        
        # 模拟MeshCNN的特征维度
        features = torch.randn(1, 5, 750).cuda()  # batch, features, edges
        print(f"✓ 特征张量创建成功: {features.shape}")
        
        # 模拟卷积操作
        conv_weight = torch.randn(64, 5, 1).cuda()
        conv_result = torch.nn.functional.conv1d(features, conv_weight)
        print(f"✓ 卷积操作成功: {conv_result.shape}")
        
        # 模拟矩阵乘法
        matrix_a = torch.randn(64, 750).cuda()
        matrix_b = torch.randn(750, 600).cuda()
        matmul_result = torch.matmul(matrix_a, matrix_b)
        print(f"✓ 矩阵乘法成功: {matmul_result.shape}")
        
    except Exception as e:
        print(f"✗ 特定操作失败: {e}")

def main():
    """主函数"""
    diagnose_cuda_environment()
    test_specific_operations()
    
    print("\n=== 诊断完成 ===")
    print("如果出现CUBLAS错误，可能的原因：")
    print("1. CUDA驱动版本不兼容")
    print("2. cuBLAS库损坏")
    print("3. GPU硬件问题")
    print("4. PyTorch版本与CUDA版本不匹配")
    print("5. 内存不足或碎片化")

if __name__ == "__main__":
    main()
