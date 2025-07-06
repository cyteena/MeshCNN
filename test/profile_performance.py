#!/usr/bin/env python3
"""
MeshCNN 性能分析工具
用于识别具体的性能瓶颈
"""

import time
import torch
import os
import argparse
import pickle
from contextlib import contextmanager
from models import create_model
from models.layers.mesh import Mesh
from util.util import pad

class PerformanceProfiler:
    def __init__(self):
        self.timing_data = {}
        self.memory_data = {}
        
    @contextmanager
    def time_block(self, name):
        """计时上下文管理器"""
        start_time = time.time()
        start_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        yield
        
        end_time = time.time()
        end_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        self.timing_data[name] = end_time - start_time
        self.memory_data[name] = end_memory - start_memory
        
    def print_results(self):
        """打印性能分析结果"""
        print("\n=== 性能分析结果 ===")
        print("\n--- 时间消耗 (秒) ---")
        total_time = sum(self.timing_data.values())
        for name, time_taken in sorted(self.timing_data.items(), key=lambda x: x[1], reverse=True):
            percentage = (time_taken / total_time) * 100 if total_time > 0 else 0
            print(f"{name:30s}: {time_taken:8.4f}s ({percentage:5.1f}%)")
            
        print(f"\n总时间: {total_time:.4f}s")
        
        if torch.cuda.is_available():
            print("\n--- 内存消耗 (MB) ---")
            for name, memory_used in sorted(self.memory_data.items(), key=lambda x: x[1], reverse=True):
                print(f"{name:30s}: {memory_used / 1024 / 1024:8.2f} MB")

def analyze_mesh_loading_performance(obj_path, opt):
    """分析网格加载性能"""
    profiler = PerformanceProfiler()
    
    print("=== 分析网格加载性能 ===")
    
    # 1. 测试网格加载
    with profiler.time_block("mesh_loading"):
        mesh = Mesh(file=obj_path, opt=opt, hold_history=True, export_folder='')
    
    # 2. 测试特征访问
    with profiler.time_block("feature_access"):
        initial_features = mesh.features
    
    # 3. 测试特征预处理
    with open('./datasets/shrec_16/mean_std_cache.p', 'rb') as f:
        transform_dict = pickle.load(f)
        mean = transform_dict['mean']
        std = transform_dict['std']
    
    with profiler.time_block("feature_preprocessing"):
        features = (initial_features - mean) / std
        features = pad(features, 750)
    
    # 4. 测试数据转换
    with profiler.time_block("data_conversion"):
        features_tensor = torch.from_numpy(features).float().unsqueeze(0)
        if torch.cuda.is_available():
            features_tensor = features_tensor.cuda()
    
    profiler.print_results()
    return mesh, features_tensor

def analyze_model_performance(model, features_tensor, mesh):
    """分析模型推理性能"""
    profiler = PerformanceProfiler()
    
    print("\n=== 分析模型推理性能 ===")
    
    net = model.net
    if isinstance(net, torch.nn.DataParallel):
        net = net.module
    
    x = features_tensor
    
    with torch.no_grad():
        # 分析每一层的性能
        for i in range(len(net.k) - 1):
            layer_name = f"layer_{i}"
            
            # 卷积层
            with profiler.time_block(f"{layer_name}_conv"):
                x = getattr(net, f'conv{i}')(x, [mesh])
            
            # 归一化层
            with profiler.time_block(f"{layer_name}_norm"):
                x = torch.nn.functional.relu(getattr(net, f'norm{i}')(x))
            
            # 池化层
            with profiler.time_block(f"{layer_name}_pool"):
                x = getattr(net, f'pool{i}')(x, [mesh])
    
    profiler.print_results()
    return x

def analyze_memory_usage():
    """分析内存使用情况"""
    if not torch.cuda.is_available():
        print("CUDA 不可用，无法分析GPU内存")
        return
    
    print("\n=== GPU 内存使用分析 ===")
    
    # 获取GPU内存信息 (兼容PyTorch 1.3.1)
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024  # MB
    allocated_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
    
    # PyTorch 1.3.1 可能没有 memory_reserved，使用 memory_cached
    try:
        cached_memory = torch.cuda.memory_cached() / 1024 / 1024  # MB
    except AttributeError:
        cached_memory = allocated_memory  # 如果没有cached，使用allocated作为近似
    
    print(f"GPU总内存: {total_memory:.0f} MB")
    print(f"已分配内存: {allocated_memory:.2f} MB")
    print(f"缓存内存: {cached_memory:.2f} MB")
    print(f"可用内存: {total_memory - cached_memory:.2f} MB")
    
    # 内存使用百分比
    memory_usage = (allocated_memory / total_memory) * 100
    print(f"内存使用率: {memory_usage:.1f}%")

def profile_mesh_operations(mesh):
    """分析网格操作性能"""
    profiler = PerformanceProfiler()
    
    print("\n=== 分析网格操作性能 ===")
    
    # 测试网格属性访问
    with profiler.time_block("mesh_edges_access"):
        _ = mesh.edges
    
    # 测试其他可用的属性
    with profiler.time_block("mesh_vs_access"):
        _ = mesh.vs
    
    with profiler.time_block("mesh_gemm_access"):
        _ = mesh.gemm_edges
    
    # 测试边数量
    with profiler.time_block("mesh_edges_count"):
        _ = mesh.edges_count
    
    profiler.print_results()

def main():
    """主函数"""
    # 设置基本参数
    opt = argparse.Namespace()
    opt.arch = 'mconvnet'
    opt.resblocks = 1
    opt.fc_n = 100
    opt.ncf = [64, 128, 256, 256]
    opt.pool_res = [600, 450, 300, 180]
    opt.norm = 'group'
    opt.num_groups = 16
    opt.num_aug = 20
    opt.ninput_edges = 750
    opt.batch_size = 1
    opt.export_folder = ''
    opt.is_train = False
    opt.checkpoints_dir = './checkpoints'
    opt.name = 'shrec16'
    opt.gpu_ids = '0'
    opt.serial_batches = True
    opt.dataset_mode = 'classification'
    opt.init_type = 'normal'
    opt.init_gain = 0.02
    opt.nclasses = 30
    opt.input_nc = 5
    opt.which_epoch = 'latest'
    
    # 测试文件路径
    test_obj = './datasets/shrec_16/alien/train/T5.obj'
    
    if not os.path.exists(test_obj):
        print(f"测试文件不存在: {test_obj}")
        print("请确保数据集已正确下载")
        return
    
    print("开始性能分析...")
    
    # 分析GPU环境
    print("=== GPU 环境信息 ===")
    if torch.cuda.is_available():
        print(f"GPU型号: {torch.cuda.get_device_name(0)}")
        print(f"PyTorch版本: {torch.__version__}")
        analyze_memory_usage()
    else:
        print("CUDA不可用，运行在CPU上")
    
    # 1. 分析网格加载性能
    mesh, features_tensor = analyze_mesh_loading_performance(test_obj, opt)
    
    # 2. 分析网格操作性能
    profile_mesh_operations(mesh)
    
    # 3. 创建模型并分析推理性能
    print("\n=== 创建模型 ===")
    model_start = time.time()
    model = create_model(opt)
    model.load_network('latest')
    model.net.eval()
    model_time = time.time() - model_start
    print(f"模型加载时间: {model_time:.4f}s")
    
    # 4. 分析模型推理性能
    try:
        features = analyze_model_performance(model, features_tensor, mesh)
        print(f"\n最终特征形状: {features.shape}")
    except Exception as e:
        print(f"模型推理失败: {e}")
    
    # 5. 最终内存分析
    analyze_memory_usage()
    
    print("\n=== 性能分析完成 ===")

if __name__ == "__main__":
    main()
