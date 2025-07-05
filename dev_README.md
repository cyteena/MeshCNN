# 使用 MeshCNN 提取三维网格几何特征

本文档旨在指导如何使用 `feature_extractor.py` 脚本，从一个三维网格文件（`.obj`）中提取高级几何特征。这些特征由一个预训练的 MeshCNN 模型生成，可以被整合到我们现有的医学图像修复网络中，以提供丰富的几何先验信息。

## 核心思想

特征提取分为两个阶段：

1.  **初始特征提取**：首先，脚本会为网格的每一条边计算一个5维的几何特征向量，包括二面角、对角等。
2.  **学习特征提取**：然后，这些初始特征被送入一个预训练的 `MeshConvNet` 网络。我们不执行完整的网络，而是在最后一个池化层之后、全局分类头之前，截取中间层的输出。这个输出就是我们需要的“学习特征”，它包含了网络学到的关于模型形状的抽象表示。

## 先决条件

在运行脚本之前，您需要准备好两样东西：

1.  **预训练模型权重 (`.pth` 文件)**：这是一个已经在大型数据集（如SHREC）上训练好的模型。您可以运行项目自带的 `bash ./scripts/shrec/get_pretrained.sh` 来下载一个示例模型，它会被存放在 `./checkpoints/shrec/latest_net.pth`。
2.  **归一化统计文件 (`mean_std_cache.p`)**：模型在训练时，输入特征都经过了标准化处理（减去均值，除以标准差）。我们在提取特征时必须使用**完全相同**的均值和标准差。这个文件是在训练集上计算得到的。运行 `bash ./scripts/shrec/get_data.sh` 会生成这个文件，它位于 `./datasets/shrec/mean_std_cache.p`。

## 文件结构

将我们提供的 `feature_extractor.py` 脚本放置在 MeshCNN 项目的根目录下，与 `train.py` 和 `models` 文件夹同级。

```
MeshCNN/
├── checkpoints/
├── datasets/
├── models/
├── options/
├── util/
├── feature_extractor.py  <-- 放置在这里
├── train.py
└── ...
```

## 核心代码解析 (`feature_extractor.py`)

脚本的核心是 `MeshCNNFeatureExtractor` 类。

-   **`__init__(checkpoint_path, mean_std_path, gpu_ids='-1')`**
    -   **功能**：初始化提取器。它会根据您的训练参数配置，搭建一个 `MeshConvNet` 模型，加载指定的预训练权重，并准备好归一化所需的均值/标准差。
    -   **参数**：
        -   `checkpoint_path`: 指向 `.pth` 权重文件的路径。
        -   `mean_std_path`: 指向 `mean_std_cache.p` 文件的路径。
        -   `gpu_ids`: 指定使用哪个GPU，'-1' 代表使用CPU。

-   **`extract(obj_path)`**
    -   **功能**：这是执行特征提取的主要方法。
    -   **参数**：`obj_path` - 您想要处理的 `.obj` 文件的路径。
    -   **返回**：一个元组 `(learned_features, simplified_mesh)`
        -   `learned_features`: 一个 PyTorch 张量，包含了最终学习到的高维特征。
        -   `simplified_mesh`: 一个 `Mesh` 对象，代表了经过网络池化（简化）后的网格。**这个对象的边与 `learned_features` 的第二维是一一对应的。**

## 使用方法

以下是一个完整的使用示例，展示了如何调用该脚本来提取特征。

```python
import os
from feature_extractor import MeshCNNFeatureExtractor

# 1. 定义必要的路径
CHECKPOINT_PATH = './checkpoints/shrec/latest_net.pth'
MEAN_STD_PATH = './datasets/shrec/mean_std_cache.p'
OBJ_FILE_PATH = './datasets/shrec/test/T1.obj' 

# 2. 检查文件是否存在，确保环境设置正确
# ... (省略示例代码中的检查逻辑) ...

# 3. 初始化特征提取器
# 如果你想在GPU上运行，可以修改 gpu_ids='0'
print("正在初始化提取器...")
extractor = MeshCNNFeatureExtractor(checkpoint_path=CHECKPOINT_PATH, 
                                    mean_std_path=MEAN_STD_PATH, 
                                    gpu_ids='-1')
print("初始化完成。")

# 4. 从你的.obj文件提取特征
print(f"正在从 {OBJ_FILE_PATH} 提取特征...")
learned_features, simplified_mesh = extractor.extract(OBJ_FILE_PATH)
print("提取完成。")

# 5. 查看并使用结果
print("\n--- 输出结果 ---")
print(f"学习特征的形状: {learned_features.shape}")

# 对于shrec的默认模型，输出形状应该是 [256, 180]
# 256 是最后一个卷积层的特征通道数
# 180 是经过4次池化后剩余的边数

print(f"简化后网格的边数: {simplified_mesh.edges_count}")

# 现在，你可以将 `learned_features` 这个张量输入到你的医学图像修复网络中了。
```

## 附录：关于 `MeshEncoderDecoder` (用于分割任务)

虽然我们当前使用的 `mconvnet` 架构，但如果您未来需要处理分割任务，可能会用到 `meshunet` 架构，它对应的是 `MeshEncoderDecoder` 类。

-   **结构**: `MeshEncoderDecoder` 类似于图像中的U-Net，包含一个编码器（降采样）和一个解码器（上采样）。
-   **特征提取点**: 对于这种架构，最能代表整个网格全局信息的特征位于**编码器的最末端**，也就是"U"型结构的**瓶颈(bottleneck)**部分。提取方法与上述类似，只需执行完所有编码器（`DownConv`）的模块即可。