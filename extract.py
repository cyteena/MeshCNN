# feature_extractor.py

import torch
import numpy as np
import os
import argparse
import pickle

# 确保所有必要的 MeshCNN 模块都能被导入
from options.base_options import BaseOptions
from models import create_model
from models.layers.mesh import Mesh
from util.util import pad

class MeshCNNFeatureExtractor:
    """
    一个封装好的类，用于从 3D 网格文件 (.obj) 中提取学习到的几何特征。
    """
    def __init__(self, checkpoint_path, mean_std_path, gpu_ids='-1'):
        """
        初始化特征提取器。

        参数:
            checkpoint_path (str): 预训练模型权重文件 (.pth) 的路径。
            mean_std_path (str): 训练数据归一化所用的均值和标准差缓存文件 (mean_std_cache.p) 的路径。
            gpu_ids (str): 使用的 GPU ID，例如 '0' 或 '0,1'。使用 '-1' 表示 CPU。
        """
        print("Initializing MeshCNNFeatureExtractor...")
        self.opt = self._create_options(checkpoint_path, gpu_ids)
        
        # 加载归一化所需的均值和标准差
        with open(mean_std_path, 'rb') as f:
            transform_dict = pickle.load(f)
            self.mean = transform_dict['mean']
            self.std = transform_dict['std']
        
        # 创建并加载模型
        self.model = create_model(self.opt)
        
        # 从 checkpoint_path 推断 epoch
        which_epoch = os.path.basename(checkpoint_path).split('_')[0]
        self.model.load_network(which_epoch)
        self.model.net.eval() # 必须设置为评估模式

        # 将模型移动到指定设备
        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = [int(str_id) for str_id in str_ids if int(str_id) >= 0]
        if len(self.opt.gpu_ids) > 0:
            self.device = torch.device(f'cuda:{self.opt.gpu_ids[0]}')
        else:
            self.device = torch.device('cpu')
            
        self.model.net.to(self.device)

        print("Initialization complete.")

    def _create_options(self, checkpoint_path, gpu_ids):
        """根据您提供的训练参数创建一个最小化的配置对象。"""
        # 使用 argparse.Namespace 来模拟命令行参数
        opt = argparse.Namespace()

        # --- 关键网络参数 (必须与预训练模型匹配) ---
        opt.arch = 'mconvnet'
        opt.resblocks = 1
        opt.fc_n = 100  # <--- 添加了这行缺失的参数 (使用默认值)
        opt.ncf = [64, 128, 256, 256]
        opt.pool_res = [600, 450, 300, 180]
        opt.norm = 'group'
        opt.num_groups = 16 # group norm 的默认值或需要指定
        
        # --- 数据和加载参数 ---
        opt.num_aug = 20 # 这会影响缓存文件的查找路径，但我们直接加载obj，所以影响不大
        opt.ninput_edges = 750 # 网络输入所期望的固定边数
        opt.batch_size = 1 # 我们一次只处理一个模型
        opt.export_folder = '' # 不导出中间过程
        
        # --- 模型加载和设备参数 ---
        opt.is_train = False
        opt.checkpoints_dir = './checkpoints'  # 假设您使用的是 SHREC16 数据集
        opt.name = 'shrec16'
        opt.gpu_ids = gpu_ids
        opt.serial_batches = True
        opt.dataset_mode = 'classification' # 假设是分类模型
        opt.init_type = 'normal'
        opt.init_gain = 0.02

        # 确保 opt 对象包含 ClassifierModel 需要的其他属性
        opt.nclasses = 30 # SHREC数据集的类别数，对于特征提取不重要但初始化需要
        opt.input_nc = 5 # 初始特征维度

        # other
        opt.which_epoch = 'latest'  # 加载最新的模型
        
        return opt

    def extract(self, obj_path):
        """
        从单个 .obj 文件提取特征。

        参数:
            obj_path (str): .obj 文件的路径。

        返回:
            tuple: (learned_features, simplified_mesh)
                   - learned_features (torch.Tensor): 学习到的高维特征张量，形状为 [特征维度, 简化后的边数]。
                   - simplified_mesh (Mesh): 经过网络池化后的简化网格对象。
        """
        # 1. 加载网格并自动提取初始5维特征
        mesh = Mesh(file=obj_path, opt=self.opt, hold_history=True, export_folder=self.opt.export_folder)
        initial_features = mesh.features # 形状: (5, num_edges)

        # 2. 预处理特征: 归一化 和 填充
        features = (initial_features - self.mean) / self.std
        features = pad(features, self.opt.ninput_edges) # 填充到网络期望的输入大小
        
        # 3. 转换为 PyTorch Tensor 并添加 batch 维度
        features = torch.from_numpy(features).float().unsqueeze(0)
        features = features.to(self.device)

        # 4. 执行部分前向传播
        # 我们直接调用 model.net (即 MeshConvNet 实例) 的层
        net = self.model.net

        if isinstance(net, torch.nn.DataParallel):
            # 如果是 DataParallel 模型，获取实际的模型
            net = net.module

        x = features
        
        # 手动执行卷积和池化块，直到最后一个池化层
        for i in range(len(net.k) - 1):
            x = getattr(net, 'conv{}'.format(i))(x, [mesh])
            x = torch.nn.functional.relu(getattr(net, 'norm{}'.format(i))(x))
            x = getattr(net, 'pool{}'.format(i))(x, [mesh])

        # x 现在就是我们想要的高维特征，在全局池化之前
        learned_features = x.squeeze(0) # 移除 batch 维度

        # 返回特征和简化后的网格对象
        # mesh 对象在池化过程中被动态修改，现在它代表了简化后的网格
        return learned_features, mesh


if __name__ == '__main__':
    # ==================== 使用示例 ====================

    # 1. 设置路径
    # 假设你已经运行了 get_data.sh 和 get_pretrained.sh for shrec
    # 或者用您自己的模型替换这些路径
    CHECKPOINT_PATH = './checkpoints/shrec16/latest_net.pth'
    MEAN_STD_PATH = './datasets/shrec_16/mean_std_cache.p'
    # 选择一个要测试的 obj 文件
    OBJ_FILE_PATH = './datasets/shrec_16/alien/train/T5.obj'

    # 检查文件是否存在
    if not os.path.exists(CHECKPOINT_PATH) or not os.path.exists(MEAN_STD_PATH):
        print("错误: 找不到模型权重或归一化缓存。")
        print("请确保您已经运行了 'bash ./scripts/shrec/get_data.sh' 和 'bash ./scripts/shrec/get_pretrained.sh'")
        print("或者提供了您自己的模型文件路径。")
        exit()

    if not os.path.exists(OBJ_FILE_PATH):
         print(f"错误: 找不到测试文件 {OBJ_FILE_PATH}。请确保数据集已正确下载。")
         exit()

    # 2. 初始化提取器 (将在 CPU 上运行)
    extractor = MeshCNNFeatureExtractor(checkpoint_path=CHECKPOINT_PATH, 
                                        mean_std_path=MEAN_STD_PATH, 
                                        gpu_ids='0')

    # 3. 提取特征
    learned_features, simplified_mesh = extractor.extract(OBJ_FILE_PATH)

    # 4. 查看结果
    print("\n--- 特征提取结果 ---")
    print(f"输入文件: {OBJ_FILE_PATH}")
    print(f"学习到的特征张量形状: {learned_features.shape}")
    print(f"  - 特征维度: {learned_features.shape[0]}")
    print(f"  - 简化后网格的边数: {learned_features.shape[1]}")
    
    print("\n--- 简化后网格信息 ---")
    print(f"初始边数: ~{extractor.opt.ninput_edges}") # 这是填充后的数量
    print(f"最终边数: {simplified_mesh.edges_count}")

    # 您现在可以将 `learned_features` 用于您的下游任务，
    # 例如，将其整合到您的医学图像修复网络中。
    # `simplified_mesh` 对象包含了这些特征对应的几何结构信息（顶点、边、邻接关系）。