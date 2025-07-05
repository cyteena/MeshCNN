
# 🧩 MeshCNN 初始特征提取与学习特征提取流程说明

## 第一步：初始几何特征提取

### 📁 核心文件定位

* `models/layers/mesh.py`
  定义了 `Mesh` 类，这是代表一个三维模型的核心数据结构。加载 `.obj` 文件时即是实例化该类。

* `models/layers/mesh_prepare.py`
  包含了从零开始处理 `.obj` 文件的所有核心逻辑，包括读取顶点/面、构建边、计算邻接关系以及提取 5 维几何特征。

---

### 🔄 代码执行流程

```python
# 入口：构建 Mesh 对象
mesh = Mesh(file='your_mesh.obj', opt=...)

# 构造函数 __init__ 中自动调用
fill_mesh(self, file, opt)
```

* 如果没有缓存 `.npz` 文件，`fill_mesh` 会调用 `from_scratch(file, opt)`：

  * `fill_from_file`: 读取顶点 (`vs`) 和面 (`faces`)
  * `build_gemm`: 构建所有边 (`edges`) 以及邻接关系 (`gemm_edges`)
  * `extract_features`: 提取每条边的 5 维几何特征

---

### 🧠 特征提取逻辑：`extract_features`

特征由以下三部分组成：

| 函数名                         | 含义                | 维度 |
| --------------------------- | ----------------- | -- |
| `dihedral_angle`            | 每条边的二面角           | 1  |
| `symmetric_opposite_angles` | 相邻两个三角面中，对应角的角度   | 2  |
| `symmetric_ratios`          | 每个三角面中顶点到边的高与边长比值 | 2  |

最终输出 `(5, N)` 的 `numpy` 数组（N 是边数），保存于 `mesh.features` 属性中。

---

### ✅ 小结

只需创建一个 `Mesh` 实例，即可自动完成初始几何特征提取：

```python
mesh = Mesh(file='your_mesh.obj', opt=...)
features = mesh.features  # shape: (5, N)
```

---

## 第二步：加载预训练网络并提取学习特征

### 💡 核心思想

加载预训练的 MeshCNN 模型，将第一步的几何特征作为输入，执行部分前向传播，截取 **卷积 + 池化部分之后、全连接部分之前** 的中间特征。

这些特征即为模型学习得到的高层抽象表示。

---

### 📁 核心文件与函数

* `models/networks.py`
  定义了网络结构，如 `MeshConvNet`（分类）和 `MeshEncoderDecoder`（分割）。

* `models/mesh_classifier.py`
  封装为 `ClassifierModel` 类：

  * 初始化网络结构
  * 加载预训练权重
  * 提供 `set_input()`、`forward()` 接口

---

### 🧬 提取中间特征位置（以 `MeshConvNet` 为例）

```python
# networks.py
for i in range(len(self.k) - 1):
    x = getattr(self, 'conv{}'.format(i))(x, mesh)
    x = F.relu(getattr(self, 'norm{}'.format(i))(x))
    x = getattr(self, 'pool{}'.format(i))(x, mesh)

# 🚩 此处的 x 即为我们想要提取的学习特征
```

---

### ⚙️ 执行流程解析

#### 1. 下载预训练模型

```bash
bash ./scripts/shrec/get_pretrained.sh
# 默认保存在 ./checkpoints/shrec/
```

---

#### 2. 创建并加载模型

```python
from models.mesh_classifier import ClassifierModel

model = ClassifierModel(opt)
model.load_network('latest')
```

其中 `opt` 包含架构、通道数、池化参数等（需与预训练模型参数一致）。

---

#### 3. 设置输入

```python
data = {
    'edge_features': torch.tensor(mesh.features).unsqueeze(0),  # shape: (1, 5, N)
    'mesh': [mesh],  # 列表形式
    'label': torch.tensor([0])  # 任意标签即可
}
model.set_input(data)
```

---

#### 4. 手动前向传播至中间层

```python
# 访问 model.net 手动运行卷积/池化层
x = data['edge_features']
mesh = data['mesh'][0]

for i in range(len(model.net.k) - 1):
    x = getattr(model.net, f'conv{i}')(x, mesh)
    x = F.relu(getattr(model.net, f'norm{i}')(x))
    x = getattr(model.net, f'pool{i}')(x, mesh)

# 🚩 此处的 x 为提取到的学习特征
print(x.shape)  # e.g., (1, 32, 580)
```

---

## ✅ 总结流程图

```
.obj 文件
   ↓
Mesh 对象构建 (mesh.py)
   ↓
fill_from_file → build_gemm → extract_features (mesh_prepare.py)
   ↓
初始几何特征 (5 × N)
   ↓
预训练网络（MeshConvNet）
   ↓
卷积 + 池化
   ↓
🎯 学习特征 (如 32 × M，M 是池化后的边数)
```


