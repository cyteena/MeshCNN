#!/usr/bin/env python3
"""
专门分析数据转换瓶颈的简化版本
"""

import time
import torch
import os
import argparse
import pickle
from models.layers.mesh import Mesh
from util.util import pad

def analyze_data_conversion_bottleneck():
    """专门分析数据转换瓶颈"""
    print("=== 分析数据转换瓶颈 ===")
    
    # 测试基本的CUDA操作
    print("\n1. 测试基本CUDA操作...")
    
    # 测试简单的GPU操作
    start = time.time()
    if torch.cuda.is_available():
        x = torch.randn(100, 100).cuda()
        y = torch.randn(100, 100).cuda()
        z = torch.matmul(x, y)
        torch.cuda.synchronize()
    end = time.time()
    print(f"基本CUDA操作时间: {end-start:.4f}s")
    
    # 测试CPU到GPU的数据传输
    print("\n2. 测试CPU到GPU数据传输...")
    
    # 创建一个较大的数组
    cpu_data = torch.randn(5, 750)  # 模拟特征数据大小
    
    start = time.time()
    gpu_data = cpu_data.cuda()
    torch.cuda.synchronize()
    end = time.time()
    print(f"CPU到GPU传输时间: {end-start:.4f}s")
    
    # 测试不同大小的数据传输
    print("\n3. 测试不同大小数据传输...")
    sizes = [100, 1000, 10000, 100000]
    
    for size in sizes:
        data = torch.randn(size)
        start = time.time()
        gpu_data = data.cuda()
        torch.cuda.synchronize()
        end = time.time()
        print(f"大小 {size:6d}: {end-start:.6f}s")
    
    # 测试首次CUDA初始化
    print("\n4. 测试CUDA上下文初始化...")
    
    # 重置CUDA
    torch.cuda.empty_cache()
    
    print("创建新的CUDA张量...")
    start = time.time()
    new_tensor = torch.randn(1000, 1000).cuda()
    torch.cuda.synchronize()
    end = time.time()
    print(f"新CUDA张量创建时间: {end-start:.4f}s")

def analyze_mesh_loading_detailed():
    """详细分析网格加载过程"""
    print("\n=== 详细分析网格加载过程 ===")
    
    # 基本参数
    opt = argparse.Namespace()
    opt.num_aug = 20
    opt.ninput_edges = 750
    opt.export_folder = ''
    
    test_obj = './datasets/shrec_16/alien/train/T1.obj'
    
    if not os.path.exists(test_obj):
        print(f"测试文件不存在: {test_obj}")
        return None, None
    
    # 分步骤计时
    print("步骤1: 创建Mesh对象...")
    start = time.time()
    mesh = Mesh(file=test_obj, opt=opt, hold_history=True, export_folder='')
    end = time.time()
    print(f"Mesh创建时间: {end-start:.4f}s")
    
    print("步骤2: 访问特征...")
    start = time.time()
    features = mesh.features
    end = time.time()
    print(f"特征访问时间: {end-start:.4f}s")
    print(f"特征形状: {features.shape}")
    
    print("步骤3: 加载归一化参数...")
    start = time.time()
    with open('./datasets/shrec_16/mean_std_cache.p', 'rb') as f:
        transform_dict = pickle.load(f)
        mean = transform_dict['mean']
        std = transform_dict['std']
    end = time.time()
    print(f"归一化参数加载时间: {end-start:.4f}s")
    
    print("步骤4: 特征预处理...")
    start = time.time()
    normalized_features = (features - mean) / std
    padded_features = pad(normalized_features, 750)
    end = time.time()
    print(f"特征预处理时间: {end-start:.4f}s")
    
    print("步骤5: 转换为PyTorch张量...")
    start = time.time()
    tensor_features = torch.from_numpy(padded_features).float()
    end = time.time()
    print(f"张量转换时间: {end-start:.4f}s")
    
    print("步骤6: 添加batch维度...")
    start = time.time()
    batched_features = tensor_features.unsqueeze(0)
    end = time.time()
    print(f"添加batch维度时间: {end-start:.4f}s")
    
    print("步骤7: 移动到GPU...")
    start = time.time()
    gpu_features = batched_features.cuda()
    torch.cuda.synchronize()
    end = time.time()
    print(f"GPU转换时间: {end-start:.4f}s")
    
    return mesh, gpu_features

def main():
    """主函数"""
    print("开始详细性能分析...")
    
    # 显示GPU信息
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"PyTorch版本: {torch.__version__}")
    
    # 分析数据转换瓶颈
    analyze_data_conversion_bottleneck()
    
    # 详细分析网格加载
    mesh, features = analyze_mesh_loading_detailed()
    
    if mesh is not None:
        print(f"\n最终特征形状: {features.shape}")
        print(f"网格边数: {mesh.edges_count}")
    
    print("\n=== 性能分析完成 ===")

if __name__ == "__main__":
    main()
